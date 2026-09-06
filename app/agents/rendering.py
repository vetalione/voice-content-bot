"""Rendering atoms into prompt input for the downstream editors."""

from __future__ import annotations

from collections.abc import Iterable

from app.models.atoms import ContentAtom


def render_atom(atom: ContentAtom) -> str:
    if atom.semantic_kind is not None:
        return "\n".join(
            [
                f"### ATOM {atom.id} — {atom.label}",
                f"kind: {atom.semantic_kind}; source_ranges: {[r.model_dump() for r in atom.source_ranges]}",
                f"claim: {atom.semantic_claim}",
                f"from recording: {atom.supporting_context}",
                f"supports: {atom.supports_atom_ids}; derived_from: {atom.derived_from_atom_ids}; contradicts: {atom.contradicts_atom_ids}",
            ]
        )
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


def render_atoms(atoms: Iterable[ContentAtom], related_atoms=None) -> str:
    atoms = list(atoms)
    body = "\n\n".join(render_atom(atom) for atom in atoms)
    if not related_atoms:
        return body
    selected = {a.id for a in atoms}
    relations = (
        "parent_atom_ids",
        "supports_atom_ids",
        "derived_from_atom_ids",
        "contradicts_atom_ids",
        "related_atom_ids",
    )
    linked = {x for a in atoms for name in relations for x in getattr(a, name)}
    supporting = [
        a
        for a in related_atoms
        if a.id not in selected
        and (a.id in linked or any(selected.intersection(getattr(a, n)) for n in relations))
    ]
    if supporting:
        body += (
            "\n\nSUPPORT / IMPLICATIONS (context only, do not write additional candidates):\n"
            + "\n\n".join(render_atom(a) for a in supporting)
        )
    return body
