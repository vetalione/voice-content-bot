"""Compact semantic contracts, separate from legacy editorial score fields."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AtomKind(StrEnum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    IMPLICATION = "implication"
    SYNTHESIS = "synthesis"
    COUNTERPOINT = "counterpoint"
    STORY = "story"


class ContentRoute(StrEnum):
    THREADS = "THREADS"
    REELS = "REELS"
    BOTH = "BOTH"
    ARCHIVE_ONLY = "ARCHIVE_ONLY"


class SourceRange(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("Source range end precedes start")
        return self


class SemanticAtom(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=160)
    claim: str = Field(min_length=1, max_length=1800)
    kind: AtomKind
    sources: list[SourceRange] = Field(min_length=1)
    parent_atom_ids: list[str] = Field(default_factory=list)
    supports_atom_ids: list[str] = Field(default_factory=list)
    derived_from_atom_ids: list[str] = Field(default_factory=list)
    contradicts_atom_ids: list[str] = Field(default_factory=list)
    related_atom_ids: list[str] = Field(default_factory=list)


class CoverageDiagnostics(BaseModel):
    coverage_confidence: float = Field(default=0, ge=0, le=1)
    semantic_density: Literal["sparse", "mixed", "dense", "unknown"] = "unknown"
    topic_shift_count: int = Field(default=0, ge=0)
    synthesis_detected: bool = False
    overflow: bool = False
    unresolved_omissions: list[str] = Field(default_factory=list)
    broad_atoms: bool = False
    distant_story_conclusion: bool = False


class SemanticExtraction(BaseModel):
    atoms: list[SemanticAtom] = Field(default_factory=list)
    diagnostics: CoverageDiagnostics = Field(default_factory=CoverageDiagnostics)
    overflow_hints: list[str] = Field(default_factory=list)


class DuplicateLink(BaseModel):
    duplicate_id: str
    keep_id: str
    reason: str


class RelationshipLink(BaseModel):
    atom_id: str
    target_id: str
    relation: Literal[
        "parent_atom_ids",
        "supports_atom_ids",
        "derived_from_atom_ids",
        "contradicts_atom_ids",
        "related_atom_ids",
    ]


class MergePatch(BaseModel):
    duplicates: list[DuplicateLink] = Field(default_factory=list)
    relationships: list[RelationshipLink] = Field(default_factory=list)
    synthesis_atoms: list[SemanticAtom] = Field(default_factory=list)


class AtomCorrection(BaseModel):
    atom_id: str
    claim: str
    reason: str


class CoveragePatch(BaseModel):
    omissions: list[SemanticAtom] = Field(default_factory=list)
    corrections: list[AtomCorrection] = Field(default_factory=list)
    relationships: list[RelationshipLink] = Field(default_factory=list)
    diagnostics: CoverageDiagnostics = Field(default_factory=CoverageDiagnostics)


class RouteDecision(BaseModel):
    atom_id: str
    route: ContentRoute
    reason: str


class RoutingResult(BaseModel):
    routes: list[RouteDecision] = Field(default_factory=list)


class SemanticResult(BaseModel):
    atoms: list[SemanticAtom] = Field(default_factory=list)
    diagnostics: CoverageDiagnostics = Field(default_factory=CoverageDiagnostics)
    routes: list[RouteDecision] = Field(default_factory=list)
    auditor: str = "primary"
    auditor_added_atom_count: int = 0
    escalation_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
