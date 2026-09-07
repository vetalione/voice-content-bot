"""Explicit model selection and live capability discovery; no implicit escalation."""

from __future__ import annotations

import logging
import time

import httpx

from app.services.llm import LLMError
from app.services.openrouter_client import OpenRouterClient, is_free_model

logger = logging.getLogger(__name__)


class ModelCatalog:
    def __init__(self, client=None):
        self.client = client or httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", timeout=20
        )
        self.models = {}
        self.loaded_at = 0.0

    async def refresh(self, force=False):
        if self.loaded_at and not force and time.monotonic() - self.loaded_at < 300:
            return
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            self.models = {m["id"]: m for m in response.json()["data"]}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise LLMError(
                "Cannot verify OpenRouter model catalog; no random model selected"
            ) from None
        self.loaded_at = time.monotonic()

    async def aclose(self):
        await self.client.aclose()


class TextRouter:
    """One client for writers/miner; audit_view explicitly selects the quality role."""

    def __init__(self, settings, catalog=None, claude=None, client_factory=OpenRouterClient):
        self.settings = settings
        self.catalog = catalog or ModelCatalog()
        self.claude = claude
        self.factory = client_factory
        self.clients = {}
        self.health = {"status": "unchecked"}
        self.cache_identity = [
            "semantic-text-v2",
            settings.openrouter_max_output_tokens,
            settings.openrouter_primary_model,
            settings.openrouter_primary_fallback_model,
            settings.openrouter_escalation_model,
            settings.semantic_reasoning_effort,
            settings.writing_reasoning_effort,
            settings.claude_model,
            settings.writing_provider,
        ]

    def audit_view(self, role):
        return RoutedView(self, role)

    async def preflight(self):
        try:
            await self.catalog.refresh(force=True)
            model = self._select("primary")
            self.health = {
                "status": "catalog_available",
                "primary": self.settings.openrouter_primary_model,
                "selected": model,
                "models": {
                    m: {
                        "exists": m in self.catalog.models,
                        "pricing": self.catalog.models.get(m, {}).get("pricing"),
                        "expiration_date": self.catalog.models.get(m, {}).get("expiration_date"),
                    }
                    for m in [
                        self.settings.openrouter_primary_model,
                        self.settings.openrouter_primary_fallback_model,
                        self.settings.openrouter_escalation_model,
                    ]
                    if m
                },
                "note": "Catalog existence is not an inference or credit check",
            }
        except LLMError as error:
            self.health = {"status": "unavailable", "error": str(error)}
            logger.error("Text provider preflight: %s", error)
        return self.health

    def _select(self, role):
        s = self.settings
        if role == "fallback":
            model = s.openrouter_primary_fallback_model
            if not model or model not in self.catalog.models:
                raise LLMError("Configured primary fallback unavailable")
        elif role == "escalation":
            if not s.openrouter_allow_escalation:
                raise LLMError("OpenRouter escalation is disabled")
            model = s.openrouter_escalation_model
            if not model or model not in self.catalog.models:
                raise LLMError(f"Configured escalation model unavailable: {model or '(unset)'}")
        else:
            model = s.openrouter_primary_model
            if not model:
                raise LLMError(
                    "Set OPENROUTER_PRIMARY_MODEL; legacy OPENROUTER_MODEL is not a semantic primary"
                )
            if model not in self.catalog.models:
                logger.error("Configured primary model unavailable: %s", model)
                model = s.openrouter_primary_fallback_model
                if not model or model not in self.catalog.models:
                    raise LLMError(
                        "Primary unavailable; configure an available OPENROUTER_PRIMARY_FALLBACK_MODEL explicitly"
                    )
                logger.warning("Using explicitly configured primary fallback: %s", model)
        # A free router can be selected explicitly; never use it as an implicit
        # replacement for an unavailable configured paid model.
        if not s.openrouter_allow_paid and not is_free_model(model):
            raise LLMError(
                "Set OPENROUTER_ALLOW_PAID=true to explicitly enable configured paid models"
            )
        if model == s.openrouter_primary_fallback_model and model != s.openrouter_primary_model:
            primary = self.catalog.models.get(s.openrouter_primary_model, {}).get("pricing", {})
            fallback = self.catalog.models[model].get("pricing", {})
            if not primary and not s.openrouter_allow_escalation:
                raise LLMError(
                    "Cannot verify fallback price against unavailable primary; explicit OPENROUTER_ALLOW_ESCALATION=true is required"
                )
            if (
                primary
                and not s.openrouter_allow_escalation
                and any(
                    float(fallback.get(k, 0)) > float(primary.get(k, 0))
                    for k in ("prompt", "completion")
                )
            ):
                raise LLMError(
                    "More expensive primary fallback requires OPENROUTER_ALLOW_ESCALATION=true"
                )
        return model

    async def ensure_primary(self):
        await self.catalog.refresh()
        return self._select("primary")

    async def monthly_usage(self):
        try:
            response = await self.catalog.client.get(
                "/key", headers={"Authorization": f"Bearer {self.settings.openrouter_api_key}"}
            )
            response.raise_for_status()
            value = response.json()["data"].get("usage_monthly")
            return float(value) if value is not None else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.warning("OpenRouter authoritative monthly usage unavailable")
            return None

    async def chat_json(self, **kwargs):
        return await self._chat("chat_json", **kwargs)

    async def chat_text(self, **kwargs):
        return await self._chat("chat_text", **kwargs)

    async def _chat(self, method, **kwargs):
        writing = kwargs.get("label", "").startswith(
            ("threads_editor", "reels_editor", "channel_teaser")
        )
        return await self.request(
            self.settings.writing_provider if writing else "primary", method=method, **kwargs
        )

    async def request(self, role, method="chat_json", **kwargs):
        if role == "claude":
            if not self.claude:
                raise LLMError("Claude adapter is not enabled")
            return await getattr(self.claude, method)(**kwargs)
        await self.catalog.refresh()
        target = self._select(role)
        if kwargs.get("model") and kwargs["model"] != target:
            raise LLMError("Model overrides must use explicit primary/escalation configuration")
        stage = kwargs.get("label", "")
        deep = stage.startswith(("semantic_", "relationship_", "coverage_"))
        effort = (
            self.settings.semantic_reasoning_effort
            if deep
            else self.settings.writing_reasoning_effort
        )
        info = self.catalog.models[target]
        reasoning = info.get("reasoning") or {}
        accepted = reasoning.get("supported_efforts")
        if reasoning.get("mandatory") and effort == "none":
            effort = "low"
        if accepted and effort not in accepted:
            # Do not silently promote medium to maximum on K3.
            effort = "low" if "low" in accepted else accepted[-1]
        key = (target, effort)
        if key not in self.clients:
            configured = self.settings.model_copy(
                update={
                    "openrouter_model": target,
                    "openrouter_reasoning_effort": effort,
                }
            )
            self.clients[key] = self.factory(
                configured,
                supported_parameters=(
                    None
                    if target == "openrouter/free"
                    else set(info.get("supported_parameters", []))
                ),
            )
        kwargs["model"] = target
        try:
            return await getattr(self.clients[key], method)(**kwargs)
        except LLMError as error:
            # A missing endpoint is not permission to use the escalation model.
            if error.status_code == 404:
                logger.error(
                    "Configured model has no usable endpoint: %s; primary fallback is not escalation",
                    target,
                )
                if (
                    role == "primary"
                    and self.settings.openrouter_primary_fallback_model
                    and target != self.settings.openrouter_primary_fallback_model
                ):
                    logger.warning(
                        "Trying explicit primary fallback once: %s",
                        self.settings.openrouter_primary_fallback_model,
                    )
                    kwargs.pop("model", None)
                    return await self.request("fallback", method=method, **kwargs)
            raise

    async def aclose(self):
        for client in self.clients.values():
            await client.aclose()
        await self.catalog.aclose()
        if self.claude:
            await self.claude.aclose()


class RoutedView:
    def __init__(self, router, role):
        self.router, self.role = router, role
        self.cache_identity = [router.cache_identity, role]

    async def chat_json(self, **kwargs):
        return await self.router.request(self.role, **kwargs)

    async def aclose(self):
        pass
