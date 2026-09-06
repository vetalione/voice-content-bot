"""Content atoms: the independent, reusable units mined from a transcript."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.timecode import format_timecode


class AtomCategory(StrEnum):
    """Closed vocabulary so downstream editors can filter reliably."""

    ORIGINAL_IDEA = "original_idea"
    BUSINESS_INSIGHT = "business_insight"
    MARKETING_OBSERVATION = "marketing_observation"
    PREDICTION = "prediction"
    FRAMEWORK = "framework"
    STRONG_OPINION = "strong_opinion"
    PARADOX = "paradox"
    PERSONAL_STORY = "personal_story"
    FUNNY_INCIDENT = "funny_incident"
    FAILURE = "failure"
    RELATIONSHIP_OBSERVATION = "relationship_observation"
    TRAVEL_INCIDENT = "travel_incident"
    LESSON = "lesson"
    CULTURE_OBSERVATION = "culture_observation"
    PROVOCATIVE_QUESTION = "provocative_question"
    OTHER = "other"


# Categories that push an atom toward the expert/professional Threads feed.
PROFESSIONAL_CATEGORIES: frozenset[AtomCategory] = frozenset(
    {
        AtomCategory.ORIGINAL_IDEA,
        AtomCategory.BUSINESS_INSIGHT,
        AtomCategory.MARKETING_OBSERVATION,
        AtomCategory.PREDICTION,
        AtomCategory.FRAMEWORK,
        AtomCategory.STRONG_OPINION,
        AtomCategory.PARADOX,
        AtomCategory.PROVOCATIVE_QUESTION,
    }
)


class ContentAtom(BaseModel):
    """One self-contained idea or story located in the original recording.

    Every field must be grounded in the transcript. ``confidence`` expresses how
    sure the miner is that the atom is genuinely present, and ``should_ignore``
    lets the miner mark filler it extracted but does not endorse.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=160, description="Internal title.")
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    description: str = Field(default="", max_length=1200)
    key_claim: str = Field(default="", max_length=800)
    supporting_context: str = Field(
        default="",
        max_length=4000,
        description="Quote or close paraphrase from the transcript.",
    )
    categories: list[AtomCategory] = Field(default_factory=list)
    business_score: float = Field(default=0.0, ge=0.0, le=10.0)
    personal_score: float = Field(default=0.0, ge=0.0, le=10.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=10.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    should_ignore: bool = False
    ignore_reason: str = Field(default="", max_length=400)

    @field_validator("categories", mode="before")
    @classmethod
    def _coerce_categories(cls, value: object) -> list[str]:
        """Accept unknown category strings by mapping them to ``other``."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list | tuple | set):
            return []
        known = {item.value for item in AtomCategory}
        out: list[str] = []
        for item in value:
            name = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            out.append(name if name in known else AtomCategory.OTHER.value)
        # de-duplicate, keep order
        return list(dict.fromkeys(out))

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> object:
        """Models sometimes answer 0-100 or 0-10 instead of 0-1."""
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        if number > 10.0:
            return min(1.0, number / 100.0)
        if number > 1.0:
            return min(1.0, number / 10.0)
        return number

    def model_post_init(self, _context: object) -> None:
        if self.end_seconds < self.start_seconds:
            object.__setattr__(self, "end_seconds", self.start_seconds)

    @property
    def timecode(self) -> str:
        return f"{format_timecode(self.start_seconds)}–{format_timecode(self.end_seconds)}"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def is_professional(self) -> bool:
        return bool(PROFESSIONAL_CATEGORIES.intersection(self.categories))

    def usable(self, min_confidence: float) -> bool:
        return not self.should_ignore and self.confidence >= min_confidence


class ContentAtomSet(BaseModel):
    """Structured envelope returned by the content miner agent."""

    atoms: list[ContentAtom] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)

    def usable(self, min_confidence: float) -> list[ContentAtom]:
        return [atom for atom in self.atoms if atom.usable(min_confidence)]


class ExtractedAtom(BaseModel):
    """Minimal first-pass extraction; no editorial metadata."""

    title: str = Field(min_length=1, max_length=100)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    idea: str = Field(min_length=1, max_length=240)
    type: str = Field(min_length=1, max_length=40)


class AtomExtraction(BaseModel):
    atoms: list[ExtractedAtom] = Field(max_length=6)


class AtomEnrichment(BaseModel):
    categories: list[AtomCategory]
    business_score: float = Field(ge=0, le=10)
    personal_score: float = Field(ge=0, le=10)
    novelty_score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    should_ignore: bool
    ignore_reason: str = Field(max_length=200)
