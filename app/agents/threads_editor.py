"""Threads editor: turns professional atoms into standalone post drafts.

Threads is the expert/personal-professional channel, so atoms are pre-ranked
here in Python (cheap, deterministic) before the model sees them; the model then
does the editorial selection and writing. Personal atoms are not removed — they
are offered as supporting material only.
"""

from __future__ import annotations

import logging

from app.models.atoms import ContentAtom
from app.models.content import ThreadsBatch, ThreadsCandidate

from .base import StructuredAgent

logger = logging.getLogger(__name__)


def rank_for_threads(atoms: list[ContentAtom]) -> list[ContentAtom]:
    """Professional relevance first, novelty second, personal colour last."""

    def score(atom: ContentAtom) -> float:
        value = atom.business_score * 1.6 + atom.novelty_score * 1.2
        value += atom.personal_score * 0.35
        if atom.is_professional:
            value += 2.0
        return value * (0.5 + atom.confidence / 2)

    return sorted(atoms, key=score, reverse=True)


class ThreadsEditorAgent(StructuredAgent):
    prompt_file = "threads_editor.md"
    name = "threads_editor"

    async def edit(self, atoms: list[ContentAtom]) -> ThreadsBatch:
        if not atoms:
            return ThreadsBatch()

        ranked = rank_for_threads(atoms)
        limit = self.settings.max_threads_candidates
        batch = await self.request_atom_batches(
            ThreadsBatch,
            ranked,
            batch_size=self.settings.threads_batch_size,
            limit=limit,
            max_tokens=self.settings.groq_threads_max_tokens,
            temperature=0.8,
            placeholder="max_posts",
        )
        batch.candidates = self._align(batch.candidates, atoms)[:limit]
        logger.info(
            "Threads editor produced %s candidate(s), rejected %s",
            len(batch.candidates),
            len(batch.rejected),
        )
        return batch

    @staticmethod
    def _align(
        candidates: list[ThreadsCandidate], atoms: list[ContentAtom]
    ) -> list[ThreadsCandidate]:
        """Repair timecodes from the referenced atom so links stay trustworthy."""
        by_id = {atom.id: atom for atom in atoms}
        aligned: list[ThreadsCandidate] = []
        for candidate in candidates:
            atom = by_id.get(candidate.atom_id)
            if atom is not None:
                candidate.start_seconds = atom.start_seconds
                candidate.end_seconds = atom.end_seconds
            aligned.append(candidate)
        return sorted(aligned, key=lambda item: item.readiness_score, reverse=True)
