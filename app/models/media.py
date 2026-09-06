"""Models describing the incoming Telegram media and the job around it."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceMode(StrEnum):
    """Where the recording came from."""

    CHANNEL = "channel"
    PRIVATE = "private"


class MediaKind(StrEnum):
    VOICE = "voice"
    AUDIO = "audio"


class MediaRef(BaseModel):
    """Everything we need to fetch and describe a Telegram audio payload."""

    model_config = ConfigDict(frozen=True)

    kind: MediaKind
    file_id: str
    file_unique_id: str
    file_size: int | None = None
    duration_seconds: int | None = None
    mime_type: str | None = None
    file_name: str | None = None
    title: str | None = None
    performer: str | None = None

    @property
    def duration_label(self) -> str:
        if not self.duration_seconds:
            return "неизвестно"
        minutes, seconds = divmod(int(self.duration_seconds), 60)
        return f"{minutes} мин {seconds:02d} сек"

    @property
    def size_mb(self) -> float | None:
        if self.file_size is None:
            return None
        return round(self.file_size / (1024 * 1024), 2)


class JobRequest(BaseModel):
    """A unit of work handed from a Telegram handler to the background runner."""

    model_config = ConfigDict(frozen=True)

    mode: SourceMode
    chat_id: int
    message_id: int
    media: MediaRef
    requester_id: int | None = None
    caption: str | None = None
    posted_at: datetime | None = None
    forwarded: bool = False
    chat_title: str | None = None

    @property
    def dedupe_key(self) -> str:
        return f"{self.mode.value}:{self.chat_id}:{self.message_id}"

    @property
    def may_publish(self) -> bool:
        """Only channel mode is ever allowed to write to the public channel."""
        return self.mode is SourceMode.CHANNEL


class RecordingMetadata(BaseModel):
    """Human facing metadata for the owner report."""

    mode: SourceMode
    source_chat_id: int
    source_message_id: int
    chat_title: str | None = None
    posted_at: datetime | None = None
    duration_seconds: float = 0.0
    detected_language: str | None = None
    transcript_chars: int = 0
    transcript_words: int = 0
    chunks: int = 1
    atoms_found: int = 0
    atoms_used: int = 0
    forwarded: bool = False
    published: bool = False
    published_message_id: int | None = None
    kind: Literal["voice", "audio"] = "voice"
    processing_seconds: float = Field(default=0.0, ge=0.0)
