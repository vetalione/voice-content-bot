"""Transcript primitives shared by the transcription and merge services."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TranscriptSegment(BaseModel):
    """A timestamped fragment of speech, in seconds from recording start."""

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str = ""

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def shifted(self, offset: float) -> TranscriptSegment:
        """Return a copy translated along the timeline by ``offset`` seconds."""
        return TranscriptSegment(
            start=max(0.0, self.start + offset),
            end=max(0.0, self.end + offset),
            text=self.text,
        )

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    """A full, chronologically ordered transcript of one recording."""

    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str | None = None
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def window(self, start: float, end: float) -> list[TranscriptSegment]:
        """Segments whose midpoint falls inside ``[start, end)``."""
        result = []
        for seg in self.segments:
            midpoint = (seg.start + seg.end) / 2
            if start <= midpoint < end:
                result.append(seg)
        return result


class ChunkTranscript(BaseModel):
    """Raw transcription of a single audio chunk, timestamps chunk-relative."""

    index: int = Field(ge=0)
    offset_seconds: float = Field(default=0.0, ge=0.0)
    overlap_seconds: float = Field(default=0.0, ge=0.0)
    duration: float = Field(default=0.0, ge=0.0)
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    text: str = ""

    @property
    def resolved_text(self) -> str:
        if self.text.strip():
            return self.text.strip()
        return " ".join(seg.text for seg in self.segments).strip()
