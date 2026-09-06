"""Text normalisation helpers used by transcript merging and dedupe."""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    folded = _PUNCT_RE.sub(" ", folded)
    return _SPACE_RE.sub(" ", folded).strip()


def tokens(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []


def token_overlap_ratio(left: str, right: str) -> float:
    """Fraction of ``left``'s tokens that also appear in ``right`` (multiset)."""
    left_tokens = tokens(left)
    if not left_tokens:
        return 0.0
    from collections import Counter

    right_counts = Counter(tokens(right))
    matched = 0
    for token in left_tokens:
        if right_counts.get(token, 0) > 0:
            right_counts[token] -= 1
            matched += 1
    return matched / len(left_tokens)


def longest_boundary_repeat(tail: str, head: str, max_tokens: int = 60) -> int:
    """Length of the longest token sequence that ends ``tail`` and starts ``head``.

    Used to remove words duplicated by the audio overlap between two chunks.
    """
    tail_tokens = tokens(tail)[-max_tokens:]
    head_tokens = tokens(head)[:max_tokens]
    limit = min(len(tail_tokens), len(head_tokens))
    for size in range(limit, 0, -1):
        if tail_tokens[-size:] == head_tokens[:size]:
            return size
    return 0


def drop_leading_tokens(text: str, count: int) -> str:
    """Remove the first ``count`` whitespace-separated words from ``text``."""
    if count <= 0:
        return text
    parts = text.split()
    if count >= len(parts):
        return ""
    return " ".join(parts[count:])


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
