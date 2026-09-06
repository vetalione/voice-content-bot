"""OpenRouter text-only client. No provider/model switch can escape free-only mode."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.config import Settings
from app.services.llm import LLMError, LLMGenerationError
from app.services.token_budget import request_tokens
from app.services.usage import current_usage
from app.utils.retry import RetryableError, retry_async

logger = logging.getLogger(__name__)


def is_free_model(model: str) -> bool:
    return model == "openrouter/free" or model.endswith(":free")


def retry_after(headers) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0, float(raw))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(raw) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class OpenRouterClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._check_model(settings.openrouter_model)
        self._own_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", timeout=settings.openrouter_timeout_seconds
        )

    def _check_model(self, model):
        if not self.settings.openrouter_allow_paid and not is_free_model(model):
            raise LLMError(
                "Paid OpenRouter model blocked: explicitly set OPENROUTER_ALLOW_PAID=true to opt in"
            )

    async def aclose(self):
        if self._own_client:
            await self.client.aclose()

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        schema_name="response",
        temperature: float | None = None,
        max_tokens: int | None = None,
        label="chat",
    ) -> dict[str, Any]:
        target = model or self.settings.openrouter_model
        self._check_model(target)
        if not self.settings.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is missing")
        input_estimate = request_tokens(system, user, schema)
        if input_estimate > self.settings.text_max_input_tokens:
            raise LLMError(
                f"{label}: text input exceeds TEXT_MAX_INPUT_TOKENS; shorten/split input"
            )
        provider = {"require_parameters": True}
        if is_free_model(target) or not self.settings.openrouter_allow_paid:
            provider["max_price"] = {"prompt": 0, "completion": 0, "request": 0, "image": 0}
        body = {
            "model": target,
            "provider": provider,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens or 2000,
        }
        # Do not send model-specific reasoning/temperature parameters to the free router.
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        else:
            body["messages"][0]["content"] += "\nReturn a JSON object only."
        compatibility_used = False
        attempt_count = 0

        async def call():
            nonlocal compatibility_used, attempt_count
            usage = current_usage.get()
            if usage and usage.requests >= self.settings.openrouter_max_requests_per_recording:
                raise LLMError(
                    "OpenRouter per-recording request limit reached; no more requests sent"
                )
            if usage:
                usage.requests += 1
            attempt_count += 1
            count = usage.requests if usage else attempt_count
            logger.info(
                "LLM request provider=openrouter stage=%s model=%s request_count=%s approximate_input_tokens=%s output_budget=%s",
                label,
                target,
                count,
                input_estimate,
                body["max_tokens"],
            )
            response = await self.client.post(
                "/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.settings.openrouter_api_key}"},
            )
            try:
                raw = response.json()
            except ValueError as error:
                if response.status_code >= 400:
                    # Rate limits and proxy failures need backoff even with an HTML body.
                    raw = {"error": {"message": "Non-JSON API error response"}}
                else:
                    raise LLMGenerationError(
                        f"{label}: OpenRouter returned non-JSON response"
                    ) from error
            if not isinstance(raw, dict):
                if response.status_code >= 400:
                    raw = {"error": {"message": "Invalid API error response"}}
                else:
                    raise LLMGenerationError(f"{label}: invalid OpenRouter response shape")
            meta = raw.get("usage")
            meta = meta if isinstance(meta, dict) else {}
            actual = raw.get("model")
            details = meta.get("completion_tokens_details")
            details = details if isinstance(details, dict) else {}

            def token_count(value):
                return value if isinstance(value, int) and value >= 0 else 0

            inp, out, reasoning = (
                token_count(meta.get("prompt_tokens")),
                token_count(meta.get("completion_tokens")),
                token_count(details.get("reasoning_tokens")),
            )
            cost = meta.get("cost")
            if usage:
                usage.input_tokens += inp
                usage.output_tokens += out
                usage.reasoning_tokens += reasoning
                if isinstance(actual, str) and actual:
                    usage.models.add(actual)
                if isinstance(cost, int | float):
                    usage.reported_cost += cost
                    usage.cost_reports += 1
            logger.info(
                "LLM usage provider=openrouter routed_provider=%s actual_model=%s stage=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s request_count=%s reported_cost=%s status=%s",
                raw.get("provider"),
                actual,
                label,
                inp,
                out,
                reasoning,
                count,
                cost,
                response.status_code,
            )
            if "error" in raw or response.status_code >= 400:
                err = raw.get("error")
                err = err if isinstance(err, dict) else {"message": str(err or "API error")}
                message = str(err.get("message", "OpenRouter request failed"))
                # Avoid leaking a key if the upstream repeats one in an error.
                message = message.replace(self.settings.openrouter_api_key, "[REDACTED]")
                code = response.status_code if response.status_code >= 400 else err.get("code", 502)
                if isinstance(code, str) and code.isdigit():
                    code = int(code)
                if code == 429 or code in {500, 502, 503, 504}:
                    delay = retry_after(response.headers)
                    raise RetryableError(
                        f"{label}: OpenRouter {code}: {message}",
                        retry_after=60 if code == 429 and delay is None else delay,
                    )
                if code in {400, 422} and (err.get("code") == "json_validate_failed"):
                    raise LLMGenerationError(f"{label}: OpenRouter JSON generation failed", code)
                explicit_unsupported = err.get("param") in {
                    "response_format",
                    "response_format.type",
                } and ("not supported" in message.lower() or "unsupported" in message.lower())
                if (
                    schema
                    and not compatibility_used
                    and code in {400, 422}
                    and explicit_unsupported
                ):
                    compatibility_used = True
                    body.pop("response_format", None)
                    body["messages"][0]["content"] += "\nReturn only JSON matching: " + json.dumps(
                        schema, ensure_ascii=False, separators=(",", ":")
                    )
                    logger.warning(
                        "%s: response_format unavailable; local JSON validation on SAME model with SAME price guard",
                        label,
                    )
                    return await call()
                raise LLMError(f"{label}: OpenRouter {code}: {message}", code)
            return raw

        try:
            raw = await retry_async(
                call,
                attempts=self.settings.openrouter_max_retries + 1,
                base_delay=5,
                max_delay=60,
                label=label,
            )
        except RetryableError as error:
            raise LLMError(f"{label}: OpenRouter retries exhausted: {error}") from error
        except httpx.HTTPError as error:
            raise LLMError(f"{label}: OpenRouter network error ({type(error).__name__})") from error
        try:
            choice = raw["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMGenerationError(f"{label}: OpenRouter output truncated")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("missing text")
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("expected object")
            return parsed
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise LLMGenerationError(f"{label}: OpenRouter returned invalid JSON object") from error
