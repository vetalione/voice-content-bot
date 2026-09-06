"""Provider-neutral text contract and errors."""

from typing import Any, Protocol


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMGenerationError(LLMError):
    pass


class LLMClient(Protocol):
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
    ) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...
