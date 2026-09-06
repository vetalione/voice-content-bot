"""Timecode formatting/parsing helpers."""

from __future__ import annotations

import re

_TIMECODE_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.\d+)?\s*$")


def format_timecode(seconds: float | int | None) -> str:
    """Render seconds as ``MM:SS`` or ``H:MM:SS`` for hour-long recordings."""
    if seconds is None:
        return "00:00"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timecode(value: str | float | int | None) -> float:
    """Parse ``MM:SS`` / ``H:MM:SS`` / plain seconds into float seconds."""
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return 0.0
    match = _TIMECODE_RE.match(text)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        secs = int(match.group(3))
        return float(hours * 3600 + minutes * 60 + secs)
    try:
        return max(0.0, float(text))
    except ValueError:
        return 0.0


def format_range(start: float, end: float) -> str:
    return f"{format_timecode(start)}–{format_timecode(end)}"
