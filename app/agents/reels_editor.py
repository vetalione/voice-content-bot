"""Reels editor: selects atoms that work as a talking-head video and scripts them.

Reels tolerate a much wider range than Threads — breakups, airports, money,
status, embarrassment — so ranking rewards story energy and novelty rather than
professional relevance. Only genuinely usable atoms are turned into Reels; the
model is explicitly allowed to return an empty list.
"""

from __future__ import annotations

import logging

from app.models.atoms import AtomCategory, ContentAtom
from app.models.content import ReelsBatch, ReelsCandidate

from .base import StructuredAgent

logger = logging.getLogger(__name__)

# Categories with natural narrative tension.
_STORY_CATEGORIES = frozenset(
    {
        AtomCategory.PERSONAL_STORY,
        AtomCategory.FUNNY_INCIDENT,
        AtomCategory.FAILURE,
        AtomCategory.RELATIONSHIP_OBSERVATION,
        AtomCategory.TRAVEL_INCIDENT,
        AtomCategory.PARADOX,
        AtomCategory.STRONG_OPINION,
    }
)


def rank_for_reels(atoms: list[ContentAtom]) -> list[ContentAtom]:
    def score(atom: ContentAtom) -> float:
        value = atom.novelty_score * 1.5 + atom.personal_score * 1.2
        value += atom.business_score * 0.8
        if _STORY_CATEGORIES.intersection(atom.categories):
            value += 2.0
        # A Reel needs enough substance to fill 30-90 seconds.
        if atom.duration < 15:
            value -= 1.5
        return value * (0.5 + atom.confidence / 2)

    return sorted(atoms, key=score, reverse=True)


class ReelsEditorAgent(StructuredAgent):
    prompt_file = "reels_editor.md"
    name = "reels_editor"

    async def edit(self, atoms: list[ContentAtom]) -> ReelsBatch:
        if not atoms:
            return ReelsBatch()

        ranked = rank_for_reels(atoms)[: self.settings.max_reels_candidates]
        limit = self.settings.max_reels_candidates
        batch = await self.request_atom_batches(
            ReelsBatch,
            ranked,
            batch_size=self.settings.reels_batch_size,
            limit=limit,
            max_tokens=(
                self.settings.groq_reels_max_tokens
                if self.settings.text_provider == "groq"
                else self.settings.text_reels_max_tokens
            ),
            temperature=0.85,
            placeholder="max_reels",
        )
        batch.candidates = self._align(batch.candidates, atoms)[:limit]
        logger.info(
            "Reels editor produced %s candidate(s), rejected %s",
            len(batch.candidates),
            len(batch.rejected),
        )
        return batch

    @staticmethod
    def _align(candidates: list[ReelsCandidate], atoms: list[ContentAtom]) -> list[ReelsCandidate]:
        by_id = {atom.id: atom for atom in atoms}
        aligned: list[ReelsCandidate] = []
        for candidate in candidates:
            atom = by_id.get(candidate.atom_id)
            if atom is not None:
                candidate.start_seconds = atom.start_seconds
                candidate.end_seconds = atom.end_seconds
            aligned.append(candidate)
        return sorted(aligned, key=lambda item: item.score, reverse=True)
