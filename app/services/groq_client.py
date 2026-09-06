"""Thin Groq HTTP client.

Groq is the *only* AI provider in this project. There is deliberately no
fallback to Anthropic/OpenAI or any other paid API: a quota problem must surface
as a clear error and an owner notification, never as a silent bill.

Retry policy:
  * 429 and 5xx are retryable, honouring ``Retry-After`` when Groq sends it;
  * 4xx other than 429 are permanent (bad key, unsupported file, bad request);
  * the number of attempts is bounded by ``GROQ_MAX_RETRIES``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.services.token_budget import request_tokens
from app.utils.retry import RetryableError, retry_async

logger = logging.getLogger(__name__)


class GroqError(RuntimeError):
    """Any non-recoverable Groq failure."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GroqSchemaError(GroqError):
    """Invalid schema configuration; never downgrade."""


class GroqCompatibilityError(GroqError):
    """Provider explicitly reports structured output unavailable for this model."""


class GroqGenerationError(GroqError):
    """The model could not finish a valid structured response."""


class GroqQuotaError(GroqError):
    """Rate limit or quota exhausted after all bounded retries."""


class GroqClient:
    """Async wrapper over the Groq OpenAI-compatible endpoints."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.groq_base_url,
            timeout=httpx.Timeout(settings.groq_timeout_seconds, connect=30.0),
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        )

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(raw) - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                return None

    def _raise_for_status(self, response: httpx.Response, label: str) -> None:
        if response.status_code < 400:
            return
        body = response.text[:800]
        try:
            code = response.json().get("error", {}).get("code")
        except (ValueError, AttributeError):
            code = None
        if response.status_code in {400, 404, 422}:
            if code in {"invalid_json_schema", "invalid_schema", "schema_validation_error"}:
                raise GroqSchemaError(
                    f"{label}: invalid structured-output schema: {body}", response.status_code
                )
            try:
                detail = response.json().get("error", {})
                message = str(detail.get("message", "")).lower()
                param = detail.get("param")
            except (ValueError, AttributeError):
                message, param = "", None
            if (
                isinstance(param, str) and param.startswith("response_format.json_schema.schema")
            ) or any(
                phrase in message
                for phrase in (
                    "invalid schema",
                    "invalid json_schema",
                    "schema keyword",
                    "additionalproperties",
                    "required properties",
                )
            ):
                raise GroqSchemaError(
                    f"{label}: invalid structured-output schema: {body}", response.status_code
                )
            # Require explicit feature/model incompatibility, not an arbitrary 4xx.
            feature = "json_schema" in message or "structured output" in message
            unavailable = (
                "not supported" in message
                or "does not support" in message
                or "unavailable" in message
            )
            if (
                code not in {"json_validate_failed"}
                and feature
                and unavailable
                and "model" in message
                and param in {None, "response_format", "response_format.type"}
            ):
                raise GroqCompatibilityError(
                    f"{label}: structured output unavailable: {body}", response.status_code
                )
        if response.status_code == 400 and code == "json_validate_failed":
            raise GroqGenerationError(
                f"{label}: Groq could not generate valid JSON within the request constraints",
                status_code=400,
            )
        if response.status_code == 429:
            sizes = re.search(r"Limit\s+([\d,]+).*?Requested\s+([\d,]+)", body, re.I)
            if sizes and int(sizes[2].replace(",", "")) > int(sizes[1].replace(",", "")):
                raise GroqError(f"{label}: request exceeds TPM capacity: {body}", status_code=429)
            delay = self._retry_after(response)
            if delay is None and ("TPM" in body or "tokens per minute" in body.lower()):
                delay = 60.0
            raise RetryableError(
                f"{label}: Groq rate limit / quota (429): {body}",
                retry_after=delay,
            )
        if response.status_code >= 500:
            raise RetryableError(f"{label}: Groq server error ({response.status_code}): {body}")
        raise GroqError(
            f"{label}: Groq rejected the request ({response.status_code}): {body}",
            status_code=response.status_code,
        )

    async def _with_retries(self, label: str, operation: Any) -> Any:
        settings = self._settings
        try:
            return await retry_async(
                operation,
                attempts=settings.groq_max_retries + 1,
                base_delay=settings.groq_retry_base_delay,
                max_delay=settings.groq_retry_max_delay,
                label=label,
            )
        except RetryableError as error:
            raise GroqQuotaError(
                f"{error} (gave up after {settings.groq_max_retries + 1} attempts)"
            ) from error
        except httpx.HTTPError as error:
            raise GroqError(f"{label}: network failure talking to Groq: {error}") from error

    # -------------------------------------------------------------------- audio
    async def transcribe_file(
        self,
        path: Path,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "verbose_json",
    ) -> dict[str, Any]:
        """Call ``/audio/transcriptions`` and return the parsed JSON payload."""

        # Telegram uses .oga for Ogg audio; Groq requires the .ogg filename alias.
        upload_name = path.with_suffix(".ogg").name if path.suffix.lower() == ".oga" else path.name

        async def call() -> dict[str, Any]:
            data: dict[str, str] = {
                "model": model,
                "response_format": response_format,
                "temperature": "0",
            }
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            with path.open("rb") as handle:
                files = {"file": (upload_name, handle, "application/octet-stream")}
                response = await self._client.post(
                    "/audio/transcriptions",
                    data={
                        **data,
                        "timestamp_granularities[]": "segment",
                    },
                    files=files,
                )
            self._raise_for_status(response, f"transcribe({path.name})")
            return response.json()

        logger.info(
            "Transcribing %s (%.2f MB) with %s",
            path.name,
            path.stat().st_size / 1024 / 1024,
            model,
        )
        return await self._with_retries(f"transcribe({path.name})", call)

    # ---------------------------------------------------------------------- llm
    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "chat",
    ) -> dict[str, Any]:
        """Chat completion constrained to a JSON object.

        Use strict schemas; JSON mode is only an explicit compatibility fallback.
        GPT-OSS 120B always uses strict output when a schema is supplied.
        """
        settings = self._settings
        target_model = model or settings.groq_llm_model
        use_schema = bool(schema) and (
            target_model == "openai/gpt-oss-120b" or settings.groq_use_json_schema
        )
        output_budget = max_tokens if max_tokens is not None else settings.groq_llm_max_tokens
        input_estimate = request_tokens(system, user, schema)
        logger.info(
            "LLM %s model=%s approximate_input_tokens=%s output_budget=%s tpm_limit=%s",
            label,
            target_model,
            input_estimate,
            output_budget,
            settings.groq_tpm_limit,
        )
        if input_estimate + output_budget > settings.groq_tpm_limit:
            raise GroqError(
                f"{label}: estimated request size {input_estimate} + output budget "
                f"{output_budget} exceeds TPM limit {settings.groq_tpm_limit}; split input"
            )

        def payload(with_schema: bool) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": (
                    settings.groq_llm_temperature if temperature is None else temperature
                ),
                "max_tokens": output_budget,
            }
            if target_model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
                body["reasoning_effort"] = "low"
            if not with_schema and schema:
                # JSON mode guarantees syntax only; retain the output contract.
                body["messages"][0]["content"] += (
                    "\nReturn JSON matching this schema: "
                    + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
                )
            if with_schema and schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                }
            else:
                body["response_format"] = {"type": "json_object"}
            return body

        async def call(with_schema: bool) -> dict[str, Any]:
            response = await self._client.post("/chat/completions", json=payload(with_schema))
            self._raise_for_status(response, label)
            return response.json()

        try:
            raw = await self._with_retries(label, lambda: call(use_schema))
        except GroqCompatibilityError:
            if not use_schema:
                raise
            logger.warning(
                "%s: structured output explicitly unavailable on %s; compatibility fallback to json_object",
                label,
                target_model,
            )
            raw = await self._with_retries(label, lambda: call(False))

        return self._extract_json(raw, label)

    @staticmethod
    def _extract_json(raw: dict[str, Any], label: str) -> dict[str, Any]:
        try:
            choice = raw["choices"][0]
            logger.info(
                "LLM %s finish_reason=%s usage=%s",
                label,
                choice.get("finish_reason"),
                raw.get("usage", {}),
            )
            if choice.get("finish_reason") == "length":
                raise GroqGenerationError(
                    f"{label}: output token budget exhausted before JSON completed"
                )
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GroqError(f"{label}: unexpected Groq response shape: {raw}") from error
        if isinstance(content, list):  # some models return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = (content or "").strip()
        if not text:
            raise GroqGenerationError(f"{label}: Groq returned an empty completion")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            snippet = _slice_json_object(text)
            if snippet is None:
                raise GroqError(f"{label}: Groq did not return valid JSON: {text[:500]}") from error
            parsed = json.loads(snippet)
        if not isinstance(parsed, dict):
            raise GroqError(f"{label}: expected a JSON object, got {type(parsed).__name__}")
        return parsed


def _slice_json_object(text: str) -> str | None:
    """Recover a JSON object wrapped in prose or a markdown fence."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate
