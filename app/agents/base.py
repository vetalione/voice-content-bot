"""Shared plumbing for the LLM agents.

Each agent is a thin object that:
  1. renders an editable markdown prompt from ``prompts/``,
  2. asks the selected text provider for a JSON object constrained by the target Pydantic schema,
  3. validates the result, retrying once with the validation error attached.

Machine-readable stages validate JSON. Writers can keep plain text in the same
persistent stage store, with source links assigned locally.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.services.llm import LLMClient, LLMError, LLMGenerationError, LLMTruncationError
from app.services.prompts import PromptLibrary
from app.services.token_budget import request_tokens

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class AgentError(RuntimeError):
    """The agent could not produce a valid structured result."""


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Closed strict schema; Pydantic retains bounds as defense in depth.

    Defaulted strings/lists remain required strings/lists (empty is meaningful).
    Only explicitly nullable annotations accept null.
    """

    def convert(node):
        result = {}
        for key, value in node.items():
            if key in {
                "title",
                "description",
                "default",
                "$schema",
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "minLength",
                "maxLength",
                "pattern",
                "format",
                "minItems",
                "maxItems",
            }:
                continue
            if key in {"properties", "$defs"}:
                result[key] = {name: convert(child) for name, child in value.items()}
            elif key == "items":
                result[key] = convert(value)
            elif key == "anyOf":
                result[key] = [convert(child) for child in value]
            elif key in {"type", "$ref", "enum", "required", "additionalProperties"}:
                result[key] = value
            else:
                raise ValueError(f"Unsupported strict JSON Schema keyword: {key}")
        if result.get("type") == "object" or "properties" in result:
            if isinstance(result.get("additionalProperties"), dict):
                raise ValueError("Open mappings cannot be represented as strict objects")
            result.setdefault("properties", {})
            result["additionalProperties"] = False
            result["required"] = list(result["properties"])
        return result

    return convert(model.model_json_schema(mode="serialization"))


class StructuredAgent:
    """Base class: prompt file + response model + one validation repair pass."""

    #: markdown file inside ``prompts/``
    prompt_file: str = ""
    #: label used in logs
    name: str = "agent"

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptLibrary,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._settings = settings

    # ------------------------------------------------------------------ helpers
    @property
    def prompts(self) -> PromptLibrary:
        return self._prompts

    @property
    def settings(self) -> Settings:
        return self._settings

    def system_prompt(self, **values: object) -> str:
        """Render the agent's prompt file, always exposing the voice guide."""
        values.setdefault("voice_style", self._prompts.voice_style(self.name))
        values.setdefault(
            "output_instructions",
            (
                "Return only the finished Russian text, ready to use. No JSON, code fences, "
                "field names or editorial explanation. If a selected atom cannot support a draft, "
                "return SKIP: followed by a brief editorial explanation instead of inventing content."
                if self.settings.text_provider == "openrouter"
                else "Return JSON matching the supplied schema. Include the draft/script, source atom_id, "
                "editorial rationale, scores and rejected items in their schema fields."
            ),
        )
        return self._prompts.render(self.prompt_file, **values)

    async def request_atom_batches(
        self,
        response_model,
        atoms,
        *,
        batch_size,
        limit,
        max_tokens,
        temperature,
        placeholder,
        source_atoms=None,
    ):
        from .rendering import render_atoms

        async def generate(batch):
            system = self.system_prompt(**{placeholder: min(limit, len(batch))})
            user = render_atoms(batch, related_atoms=source_atoms)
            estimate = request_tokens(system, user, json_schema_for(response_model))
            if (
                self.settings.text_provider == "groq"
                and estimate + max_tokens > self.settings.groq_tpm_limit
            ):
                if len(batch) == 1:
                    raise LLMError(
                        f"{self.name}: single atom exceeds request budget; shorten its context or prompt"
                    )
                mid = len(batch) // 2
                return await generate(batch[:mid]) + await generate(batch[mid:])
            result = await self.request(
                response_model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # Keep source links trustworthy even if a model invents an atom ID.
            ids = {atom.id for atom in batch}
            result.candidates = [c for c in result.candidates if c.atom_id in ids]
            return [result]

        results = []
        for start in range(0, len(atoms), batch_size):
            results.extend(await generate(atoms[start : start + batch_size]))
        candidates = [c for result in results for c in result.candidates]
        rejected = [note for result in results for note in result.rejected]
        return response_model(candidates=candidates, rejected=rejected)

    async def request_text(self, *, system, user, temperature=None, request_label=None):
        from app.services.checkpoints import active_checkpoint, fingerprint

        label = request_label or self.name
        context = active_checkpoint.get()
        key = "llm-text:" + fingerprint(
            [
                getattr(self._llm, "cache_identity", type(self._llm).__name__),
                label,
                system,
                user,
                temperature,
            ]
        )
        if context:
            cached = await context[0].get(context[1], key)
            if (
                cached
                and cached.get("status") == "complete"
                and isinstance(cached.get("text"), str)
                and cached["text"].strip()
            ):
                logger.info("Checkpoint hit stage=%s", label)
                return cached["text"]
            await context[0].put(context[1], key, {"status": "running", "stage": label})
        text = await self._llm.chat_text(
            system=system, user=user, temperature=temperature, label=label
        )
        if not isinstance(text, str) or not text.strip():
            raise LLMGenerationError(f"{label}: empty editorial response")
        text = text.strip()
        if context:
            await context[0].put(
                context[1], key, {"status": "complete", "stage": label, "text": text}
            )
        return text

    async def request(
        self,
        response_model: type[ModelT],
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_label: str | None = None,
        repair_attempts: int = 2,
    ) -> ModelT:
        from app.services.checkpoints import active_checkpoint, fingerprint

        if self.settings.text_provider == "openrouter":
            max_tokens = None  # old stage env vars must not affect requests/cache keys
        context = active_checkpoint.get()
        cache_key = "llm:" + fingerprint(
            [
                getattr(self._llm, "cache_identity", type(self._llm).__name__),
                request_label or self.name,
                system,
                user,
                response_model.model_json_schema(),
                max_tokens,
                temperature,
            ]
        )
        if context:
            cached = await context[0].get(context[1], cache_key)
            if cached and cached.get("status") == "complete":
                logger.info("Checkpoint hit stage=%s", request_label or self.name)
                return response_model.model_validate(cached["result"])
            await context[0].put(
                context[1], cache_key, {"status": "running", "stage": request_label or self.name}
            )
        schema = json_schema_for(response_model)
        attempt_user = user
        last_error: Exception | None = None

        for attempt in range(1, repair_attempts + 1):
            try:
                payload = await self._llm.chat_json(
                    system=system,
                    user=attempt_user,
                    schema=schema,
                    schema_name=response_model.__name__,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    label=f"{request_label or self.name}#{attempt}",
                )
            except LLMTruncationError:
                raise  # identical re-prompting cannot lift an upstream completion limit
            except LLMGenerationError:
                if attempt >= repair_attempts:
                    raise
                logger.warning(
                    "%s: structured generation failed on attempt %s; bounded repair via configured provider",
                    request_label or self.name,
                    attempt,
                )
                if self.settings.text_provider == "openrouter":
                    attempt_user = (
                        f"{user}\n\nThe previous response was invalid. "
                        "Return one complete JSON object matching the required schema. "
                        "Preserve all distinct ideas and relationships."
                    )
                    if self.settings.semantic_pipeline_enabled:
                        attempt_user += " Preserve omissions using overflow=true and specific overflow_hints where the schema supports them; never silently discard distinct ideas."
                continue
            except LLMError:
                raise  # quota/network problems are not repairable by re-prompting
            try:
                validated = response_model.model_validate(payload)
                if context:
                    await context[0].put(
                        context[1],
                        cache_key,
                        {
                            "status": "complete",
                            "stage": request_label or self.name,
                            "result": validated.model_dump(mode="json"),
                        },
                    )
                return validated
            except ValidationError as error:
                last_error = error
                logger.warning(
                    "%s returned an invalid payload (attempt %s): %s",
                    self.name,
                    attempt,
                    error.errors()[:3],
                )
                attempt_user = (
                    f"{user}\n\n---\n"
                    "Your previous answer did not validate against the required "
                    "JSON schema. Fix these problems and answer again with JSON "
                    "only:\n"
                    f"{json.dumps(error.errors()[:8], ensure_ascii=False, default=str)}"
                )

        raise AgentError(f"{self.name}: could not produce a valid result: {last_error}")
