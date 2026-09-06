"""Owner-wide text selection, persisted separately from recording checkpoints."""

from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation

from app.services.checkpoints import CheckpointError, NullStore
from app.services.llm import LLMError
from app.services.openrouter_client import is_free_model


def model_key(model: str) -> str:
    return hashlib.sha256(model.encode()).hexdigest()[:16]


def free_text_model(model: dict) -> bool:
    if not is_free_model(model.get("id", "")):
        return False
    if "text" not in model.get("architecture", {}).get("output_modalities", ["text"]):
        return False
    pricing = model.get("pricing", {})
    try:
        return all(Decimal(pricing[k]) == 0 for k in ("prompt", "completion")) and all(
            Decimal(pricing.get(k, "0")) == 0 for k in ("request", "image")
        )
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return False


class ModelPreferences:
    def __init__(self, settings, store, catalog):
        self.settings, self.store, self.catalog = settings, store, catalog
        self.key = f"owner-settings:{settings.owner_telegram_id}"

    async def current(self):
        return await self.store.get(self.key, "text-model") or {"mode": "configured"}

    async def choices(self):
        await self.catalog.refresh()
        return sorted(
            (
                m["id"]
                for m in self.catalog.models.values()
                if free_text_model(m) and m["id"] != "openrouter/free"
            ),
        )

    def paid_models(self):
        return list(
            dict.fromkeys(
                m
                for m in (
                    self.settings.openrouter_primary_model,
                    self.settings.openrouter_escalation_model,
                )
                if m and not is_free_model(m)
            )
        )

    async def select(self, mode, model=""):
        if isinstance(self.store, NullStore):
            raise CheckpointError("Для сохранения выбора подключи Supabase или SQLite.")
        if mode == "free":
            if model != "openrouter/free" and model not in await self.choices():
                raise LLMError("Бесплатная модель больше недоступна. Обнови список.")
        elif mode == "paid":
            if not self.settings.openrouter_allow_paid:
                raise LLMError("Платные модели выключены: нужен OPENROUTER_ALLOW_PAID=true.")
            await self.catalog.refresh()
            if model not in self.paid_models() or model not in self.catalog.models:
                raise LLMError("Настроенная платная модель недоступна.")
        else:
            raise LLMError("Неизвестный режим модели.")
        await self.store.put(self.key, "text-model", {"mode": mode, "model": model})

    async def snapshot(self):
        choice = await self.current()
        mode, model = choice.get("mode"), choice.get("model", "")
        if mode == "configured":
            return self.settings.model_copy(deep=True)
        if mode not in ("free", "paid") or not model:
            raise LLMError("Сохранённые настройки модели некорректны. Открой /settings.")
        if mode == "free" and not is_free_model(model):
            raise LLMError("Платная модель запрещена в бесплатном режиме.")
        if mode == "paid" and (
            not self.settings.openrouter_allow_paid or model not in self.paid_models()
        ):
            raise LLMError("Выбранная платная модель отключена в настройках сервера.")
        updates = {
            "text_provider": "openrouter",
            "semantic_pipeline_enabled": True,
            "openrouter_primary_model": model,
            "openrouter_model": model,
        }
        if mode == "free":
            updates.update(
                openrouter_allow_paid=False,
                openrouter_primary_fallback_model="",
                openrouter_escalation_model="",
                openrouter_allow_escalation=False,
                quality_auditor="none",
                writing_provider="primary",
                claude_enabled=False,
                force_quality_audit=False,
                semantic_reasoning_effort="low",
                writing_reasoning_effort="low",
            )
        return self.settings.model_copy(update=updates, deep=True)

    async def description(self):
        s = await self.snapshot()
        model = s.openrouter_primary_model if s.semantic_pipeline_enabled else s.openrouter_model
        if s.text_provider == "groq":
            return f"Groq: {s.groq_llm_model}"
        return f"OpenRouter: {model} ({'бесплатно' if is_free_model(model) else 'платно'})"
