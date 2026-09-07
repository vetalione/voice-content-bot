"""Output models for the teaser, Threads and Reels editors."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models.atoms import ContentAtom
from app.models.media import RecordingMetadata
from app.models.semantic import SemanticResult
from app.utils.timecode import format_timecode


class TeaserTimestamp(BaseModel):
    """One line of the optional mini table of contents."""

    seconds: float = Field(ge=0.0)
    label: str = Field(min_length=1, max_length=120)

    @property
    def timecode(self) -> str:
        return format_timecode(self.seconds)


class ChannelTeaser(BaseModel):
    """The editorial teaser published under the original channel post."""

    teaser: str = Field(min_length=1)
    timestamps: list[TeaserTimestamp] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=1000)

    @field_validator("teaser")
    @classmethod
    def _clean(cls, value: str) -> str:
        return value.strip()

    def render(self, include_timestamps: bool = True) -> str:
        """Plain-text channel message body."""
        body = self.teaser
        if include_timestamps and self.timestamps:
            lines = "\n".join(f"{item.timecode} — {item.label}" for item in self.timestamps)
            body = f"{body}\n\n{lines}"
        return body


class ThreadsCandidate(BaseModel):
    """A standalone Threads post draft plus the editorial rationale."""

    atom_id: str = Field(default="", max_length=64)
    start_seconds: float = Field(default=0.0, ge=0.0)
    end_seconds: float = Field(default=0.0, ge=0.0)
    angle: str = Field(default="", max_length=200, description="Internal label.")
    why_it_works: str = Field(default="", max_length=1200)
    readiness_score: float | None = Field(default=None, ge=0.0, le=10.0)
    draft: str = Field(min_length=1)

    @property
    def timecode(self) -> str:
        return f"{format_timecode(self.start_seconds)}–{format_timecode(self.end_seconds)}"


class ThreadsBatch(BaseModel):
    candidates: list[ThreadsCandidate] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list, description="Short notes on discarded atoms.")

    def best(self, limit: int) -> list[ThreadsCandidate]:
        ordered = sorted(self.candidates, key=lambda item: item.readiness_score or 0, reverse=True)
        return ordered[:limit]


class ReelsCandidate(BaseModel):
    """A talking-head Reel, broken into the beats plus a ready script."""

    atom_id: str = Field(default="", max_length=64)
    start_seconds: float = Field(default=0.0, ge=0.0)
    end_seconds: float = Field(default=0.0, ge=0.0)
    concept: str = Field(min_length=1, max_length=200)
    why_it_works: str = Field(default="", max_length=1200)
    score: float | None = Field(default=None, ge=0.0, le=10.0)
    target_duration_seconds: int = Field(default=60, ge=10, le=300)
    hook: str = Field(default="", max_length=600)
    setup: str = Field(default="", max_length=1200)
    development: str = Field(default="", max_length=2000)
    payoff: str = Field(default="", max_length=1500)
    ending: str = Field(default="", max_length=800)
    on_screen_text: list[str] = Field(default_factory=list)
    script: str = Field(min_length=1)

    @field_validator("on_screen_text", mode="before")
    @classmethod
    def _coerce_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list | tuple):
            return [str(item) for item in value if str(item).strip()]
        return []

    @property
    def timecode(self) -> str:
        return f"{format_timecode(self.start_seconds)}–{format_timecode(self.end_seconds)}"


class ReelsBatch(BaseModel):
    candidates: list[ReelsCandidate] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)

    def best(self, limit: int) -> list[ReelsCandidate]:
        ordered = sorted(self.candidates, key=lambda item: item.score or 0, reverse=True)
        return ordered[:limit]


class PipelineResult(BaseModel):
    """Everything one processed recording produced."""

    metadata: RecordingMetadata
    atoms: list[ContentAtom] = Field(default_factory=list)
    teaser: ChannelTeaser | None = None
    threads: list[ThreadsCandidate] = Field(default_factory=list)
    reels: list[ReelsCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    transcript_text: str = Field(default="", repr=False)
    semantic: SemanticResult | None = None
    usage_diagnostics: dict = Field(default_factory=dict)
    recording_id: str = ""
