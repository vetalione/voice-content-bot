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
    """A JSON schema Groq accepts (no ``$defs`` indirection at the root)."""
    schema = model.model_json_schema(mode="serialization")
    schema.pop("$schema", None)
    return schema


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
        values.setdefault("voice_style", self._prompts.voice_style())
        return self._prompts.render(self.prompt_file, **values)

    async def request(
        self,
        response_model: type[ModelT],
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelT:
        schema = json_schema_for(response_model)
        attempt_user = user
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                payload = await self._llm.chat_json(
                    system=system,
                    user=attempt_user,
                    schema=schema,
                    schema_name=response_model.__name__,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    label=f"{self.name}#{attempt}",
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
