"""Rendering atoms into prompt input for the downstream editors."""

from __future__ import annotations

from collections.abc import Iterable

from app.models.atoms import ContentAtom


def render_atom(atom: ContentAtom) -> str:
    categories = ", ".join(atom.categories) or "other"
    return "\n".join(
        [
            f"### ATOM {atom.id} — {atom.label}",
            f"timecode: {atom.timecode} (start_seconds={atom.start_seconds:.0f}, "
            f"end_seconds={atom.end_seconds:.0f})",
            f"categories: {categories}",
            (
                f"scores: business={atom.business_score:.1f} "
                f"personal={atom.personal_score:.1f} "
                f"novelty={atom.novelty_score:.1f} "
                f"confidence={atom.confidence:.2f}"
            ),
            f"key_claim: {atom.key_claim}",
            f"description: {atom.description}",
            f"from the recording: {atom.supporting_context}",
        ]
    )


def render_atoms(atoms: Iterable[ContentAtom]) -> str:
    return "\n\n".join(render_atom(atom) for atom in atoms)
