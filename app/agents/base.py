"""Shared plumbing for the LLM agents.

Each agent is a thin object that:
  1. renders an editable markdown prompt from ``prompts/``,
  2. asks Groq for a JSON object constrained by the target Pydantic schema,
  3. validates the result, retrying once with the validation error attached.

No agent parses free text with regex; the contract between stages is always a
Pydantic model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.services.groq_client import GroqError
from app.services.prompts import PromptLibrary
from app.services.token_budget import request_tokens

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMClient(Protocol):
    """The single Groq capability the agents need (fakeable in tests)."""

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = ...,
        schema: dict[str, Any] | None = ...,
        schema_name: str = ...,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        label: str = ...,
    ) -> dict[str, Any]: ...


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
        return self._prompts.render(self.prompt_file, **values)

    async def request_atom_batches(
        self, response_model, atoms, *, batch_size, limit, max_tokens, temperature, placeholder
    ):
        from .rendering import render_atoms

        async def generate(batch):
            system = self.system_prompt(**{placeholder: min(limit, len(batch))})
            user = render_atoms(batch)
            estimate = request_tokens(system, user, json_schema_for(response_model))
            if estimate + max_tokens > self.settings.groq_tpm_limit:
                if len(batch) == 1:
                    raise GroqError(
                        f"{self.name}: single atom exceeds TPM budget; shorten its context or prompt"
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
            except GroqError:
                raise  # quota/network problems are not repairable by re-prompting
            try:
                return response_model.model_validate(payload)
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
