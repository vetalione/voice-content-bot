"""Prompt loading from the editable ``prompts/`` directory.

Prompts live in markdown files on purpose: they are the part of the system the
owner edits most often, and editing them must not require reading Python.
Placeholders use ``{{name}}`` so markdown braces and JSON examples inside the
prompt files stay untouched.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

VOICE_STYLE_FILE = "VOICE_STYLE.md"


class PromptNotFoundError(FileNotFoundError):
    pass


class PromptLibrary:
    """Reads, caches and renders prompt templates."""

    def __init__(self, directory: Path, cache: bool = True) -> None:
        self._directory = Path(directory)
        self._cache_enabled = cache
        self._cache: dict[str, str] = {}

    @property
    def directory(self) -> Path:
        return self._directory

    def load(self, name: str) -> str:
        """Load ``prompts/<name>.md`` (extension optional)."""
        key = name if name.endswith(".md") else f"{name}.md"
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        path = self._directory / key
        if not path.exists():
            raise PromptNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if self._cache_enabled:
            self._cache[key] = text
        return text

    def voice_style(self) -> str:
        """The owner's personal voice guide; optional but expected."""
        try:
            return self.load(VOICE_STYLE_FILE)
        except PromptNotFoundError:
            logger.warning("%s is missing; continuing without a voice guide", VOICE_STYLE_FILE)
            return ""

    def render(self, name: str, **values: object) -> str:
        """Load a prompt and substitute ``{{placeholders}}``."""
        template = self.load(name)
        rendered = self.fill(template, **values)
        return rendered

    @staticmethod
    def fill(template: str, **values: object) -> str:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                logger.debug("Prompt placeholder {{%s}} left untouched", key)
                return match.group(0)
            return str(values[key])

        return _PLACEHOLDER_RE.sub(replace, template)

    def clear_cache(self) -> None:
        self._cache.clear()
