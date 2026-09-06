"""Telegram-safe text formatting and message splitting.

Telegram caps a text message at 4096 characters. Owner reports easily exceed
that, so :func:`split_html_message` cuts long HTML payloads on the coarsest
available boundary (blank line > line > sentence > word) and carries any
still-open inline tags across the seam, so no part ends with broken markup.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_CAPTION_LIMIT = 1024

# Tags Telegram accepts in HTML parse mode.
_SUPPORTED_TAGS = frozenset(
    {
        "b",
        "strong",
        "i",
        "em",
        "u",
        "ins",
        "s",
        "strike",
        "del",
        "span",
        "tg-spoiler",
        "a",
        "code",
        "pre",
        "blockquote",
    }
)
_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)((?:\s[^>]*?)?)/?>")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

# An open tag as (tag_name, full_opening_tag_text).
OpenTag = tuple[str, str]


def escape(text: str | None) -> str:
    """Escape user/model text for Telegram HTML parse mode."""
    return html.escape(text or "", quote=False)


def open_tag_stack(text: str) -> list[OpenTag]:
    """Return still-unclosed supported tags in ``text``, outermost first."""
    stack: list[OpenTag] = []
    for match in _TAG_RE.finditer(text):
        closing, name, attrs = match.group(1), match.group(2).lower(), match.group(3)
        if name not in _SUPPORTED_TAGS:
            continue
        if closing:
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == name:
                    del stack[index:]
                    break
        else:
            stack.append((name, f"<{name}{attrs or ''}>"))
    return stack


def _openers(tags: Iterable[OpenTag]) -> str:
    return "".join(tag for _, tag in tags)


def _closers(tags: Iterable[OpenTag]) -> str:
    return "".join(f"</{name}>" for name, _ in reversed(list(tags)))


def _safe_cut_index(text: str, limit: int) -> int:
    """Largest index <= ``limit`` that does not fall inside an HTML tag."""
    index = min(limit, len(text))
    while index > 0:
        prefix = text[:index]
        last_open = prefix.rfind("<")
        last_close = prefix.rfind(">")
        if last_open <= last_close:
            return index
        index = last_open
    return min(limit, len(text))


def _hard_split(text: str, limit: int) -> list[str]:
    """Last-resort splitter: cut on whitespace, never inside a tag."""
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = _safe_cut_index(rest, limit)
        space = rest[:cut].rfind(" ")
        if space > limit // 2:
            cut = space
        if cut <= 0:
            cut = min(limit, len(rest))
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def _atomize(text: str, limit: int) -> list[str]:
    """Break ``text`` into pieces that each fit inside ``limit``."""
    atoms: list[str] = []
    for block in text.split("\n\n"):
        if len(block) <= limit:
            atoms.append(block)
            continue
        for line in block.split("\n"):
            if len(line) <= limit:
                atoms.append(line)
                continue
            buffer = ""
            for sentence in _SENTENCE_RE.split(line):
                candidate = f"{buffer} {sentence}".strip() if buffer else sentence
                if len(candidate) <= limit:
                    buffer = candidate
                    continue
                if buffer:
                    atoms.append(buffer)
                    buffer = ""
                if len(sentence) <= limit:
                    buffer = sentence
                else:
                    atoms.extend(_hard_split(sentence, limit))
            if buffer:
                atoms.append(buffer)
    return atoms


def split_html_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into Telegram-sized HTML messages.

    Guarantees: every part is <= ``limit`` characters, no part ends with an
    unclosed supported tag, and no part begins in the middle of a tag.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    reserve = 128  # room for carried open/close tags
    budget = max(512, limit - reserve)
    atoms = _atomize(text, budget)

    parts: list[str] = []
    prefix: list[OpenTag] = []
    body = ""

    def flush() -> None:
        nonlocal prefix, body
        if not body.strip():
            body = ""
            return
        chunk = _openers(prefix) + body
        pending = open_tag_stack(chunk)
        parts.append((chunk + _closers(pending)).strip())
        prefix = pending
        body = ""

    for atom in atoms:
        candidate = f"{body}\n\n{atom}" if body else atom
        if len(_openers(prefix)) + len(candidate) <= budget:
            body = candidate
            continue
        flush()
        body = atom
    flush()
    return [part for part in parts if part.strip()]


def bullet_list(items: Iterable[str], marker: str = "•") -> str:
    return "\n".join(f"{marker} {item}" for item in items if str(item).strip())
