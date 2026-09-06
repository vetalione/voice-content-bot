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
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.utils.retry import RetryableError, retry_async

logger = logging.getLogger(__name__)


class GroqError(RuntimeError):
    """Any non-recoverable Groq failure."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
            return None

    def _raise_for_status(self, response: httpx.Response, label: str) -> None:
        if response.status_code < 400:
            return
        body = response.text[:800]
        if response.status_code == 429:
            raise RetryableError(
                f"{label}: Groq rate limit / quota (429): {body}",
                retry_after=self._retry_after(response),
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

        Tries ``response_format=json_schema`` when enabled and falls back to
        ``json_object`` if the model or endpoint rejects it.
        """
        settings = self._settings
        target_model = model or settings.groq_llm_model
        use_schema = bool(schema) and settings.groq_use_json_schema

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
                "max_tokens": max_tokens or settings.groq_llm_max_tokens,
            }
            if with_schema and schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": False,
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
        except GroqError as error:
            if not use_schema or error.status_code not in {400, 404, 422}:
                raise
            logger.warning(
                "%s: json_schema rejected by %s, retrying with json_object",
                label,
                target_model,
            )
            raw = await self._with_retries(label, lambda: call(False))

        return self._extract_json(raw, label)

    @staticmethod
    def _extract_json(raw: dict[str, Any], label: str) -> dict[str, Any]:
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GroqError(f"{label}: unexpected Groq response shape: {raw}") from error
        if isinstance(content, list):  # some models return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = (content or "").strip()
        if not text:
            raise GroqError(f"{label}: Groq returned an empty completion")
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
