"""Deep extraction, conservative merge, independent audit, then routing."""

from __future__ import annotations

import json
import logging

from app.agents.base import StructuredAgent
from app.agents.content_miner import build_windows
from app.models.atoms import AtomCategory, ContentAtom
from app.models.semantic import (
    ContentRoute,
    CoverageDiagnostics,
    CoveragePatch,
    MergePatch,
    RouteDecision,
    RoutingResult,
    SemanticExtraction,
    SemanticResult,
)
from app.services.llm import LLMError
from app.services.token_budget import approximate_tokens
from app.utils.text import token_overlap_ratio

logger = logging.getLogger(__name__)
RELATIONS = (
    "parent_atom_ids",
    "supports_atom_ids",
    "derived_from_atom_ids",
    "contradicts_atom_ids",
    "related_atom_ids",
)


def atom_payload(atoms):
    return json.dumps(
        [a.model_dump(mode="json") for a in atoms], ensure_ascii=False, separators=(",", ":")
    )


def diagnostics_union(items):
    return CoverageDiagnostics(
        coverage_confidence=min((d.coverage_confidence for d in items), default=0),
        semantic_density="dense" if any(d.semantic_density == "dense" for d in items) else "mixed",
        topic_shift_count=max((d.topic_shift_count for d in items), default=0),
        synthesis_detected=any(d.synthesis_detected for d in items),
        overflow=any(d.overflow for d in items),
        unresolved_omissions=list(dict.fromkeys(x for d in items for x in d.unresolved_omissions)),
        broad_atoms=any(d.broad_atoms for d in items),
        distant_story_conclusion=any(d.distant_story_conclusion for d in items),
    )


def escalation_reasons(settings, transcript, atoms, diag):
    reasons = []
    if transcript.duration >= settings.escalate_if_duration_minutes * 60:
        reasons.append("long_recording")
    if diag.coverage_confidence < settings.coverage_confidence_threshold:
        reasons.append("low_coverage_confidence")
    if diag.unresolved_omissions:
        reasons.append("unresolved_omissions")
    if diag.overflow or len(atoms) > settings.target_atom_limit:
        reasons.append("overflow")
    if diag.topic_shift_count >= 6:
        reasons.append("many_topic_transitions")
    if diag.distant_story_conclusion and not diag.synthesis_detected:
        reasons.append("missing_distant_synthesis")
    if diag.broad_atoms and (diag.semantic_density == "dense" or transcript.word_count >= 1500):
        reasons.append("broad_atoms_dense_source")
    if (
        len(atoms) <= 2
        and transcript.word_count >= 1500
        and (diag.topic_shift_count >= 3 or diag.semantic_density == "dense")
    ):
        reasons.append("suspiciously_low_coverage")
    if settings.force_quality_audit:
        reasons.append("manual_override")
    return reasons


def choose_auditor(settings, reasons):
    mode = settings.quality_auditor
    if mode == "none":
        return None
    if mode == "claude":
        if not settings.claude_enabled:
            raise LLMError("QUALITY_AUDITOR=claude requires CLAUDE_ENABLED=true")
        return "claude"
    if mode == "kimi_k3" or (mode == "auto" and reasons):
        if not settings.openrouter_allow_escalation:
            if mode == "kimi_k3":
                raise LLMError("Explicit Kimi audit requires OPENROUTER_ALLOW_ESCALATION=true")
            return None
        return "escalation"
    return None


def add_atoms(current, additions, prefix, duration):
    known = {a.id for a in current}
    renames = {}
    if len({a.id for a in additions}) != len(additions):
        raise LLMError("Duplicate IDs within semantic response")
    for atom in additions:
        if any(r.end_seconds > duration + 1 for r in atom.sources):
            raise LLMError(f"Semantic atom source outside transcript: {atom.id}")
        renames[atom.id] = (
            f"{prefix}.{atom.id}" if atom.id in known or prefix.startswith("w") else atom.id
        )
    for source in additions:
        atom = source.model_copy(deep=True)
        atom.id = renames[atom.id]
        for name in RELATIONS:
            setattr(atom, name, list(dict.fromkeys(renames.get(x, x) for x in getattr(atom, name))))
        current.append(atom)
    return renames


def apply_merge(atoms, patch):
    by_id = {a.id: a for a in atoms}
    aliases = {}
    for item in patch.duplicates:
        a, b = by_id.get(item.duplicate_id), by_id.get(item.keep_id)
        # Conservative recall-first deletion: same semantic role AND near-identical claim.
        if (
            a
            and b
            and a.id != b.id
            and a.kind == b.kind
            and token_overlap_ratio(a.claim, b.claim) >= 0.85
        ):
            b.sources = list(
                {(r.start_seconds, r.end_seconds): r for r in b.sources + a.sources}.values()
            )
            for field in RELATIONS:
                setattr(b, field, list(dict.fromkeys(getattr(b, field) + getattr(a, field))))
            aliases[a.id] = b.id
            del by_id[a.id]

    def resolve(value):
        seen = set()
        while value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    for link in patch.relationships:
        a, b = by_id.get(resolve(link.atom_id)), by_id.get(resolve(link.target_id))
        if a and b and a.id != b.id and link.relation in RELATIONS:
            setattr(a, link.relation, list(dict.fromkeys([*getattr(a, link.relation), b.id])))
    for atom in by_id.values():
        for name in RELATIONS:
            setattr(
                atom,
                name,
                list(
                    dict.fromkeys(
                        resolve(x)
                        for x in getattr(atom, name)
                        if resolve(x) in by_id and resolve(x) != atom.id
                    )
                ),
            )
    return list(by_id.values())


class SemanticMinerAgent(StructuredAgent):
    name = "semantic_extraction"
    prompt_file = "semantic_extraction.md"

    def _input(self, transcript, atoms):
        text = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript.segments)
        if not text:
            text = transcript.text
        value = f"TRANSCRIPT (absolute seconds):\n{text}\n\nCURRENT ATOMS:\n{atom_payload(atoms)}"
        if approximate_tokens(value) > self.settings.semantic_max_input_tokens - 2500:
            raise LLMError(
                "Global semantic input exceeds configured context budget; saved extraction is retained. Increase SEMANTIC_MAX_INPUT_TOKENS within model context or split this recording."
            )
        return value

    async def _stage(self, model, prompt, user, label, llm=None):
        agent = StructuredAgent(llm or self._llm, self.prompts, self.settings)
        return await agent.request(
            model,
            system=self.prompts.render(
                prompt,
                target_atom_limit=self.settings.target_atom_limit,
                voice_style=self.prompts.voice_style("content_router"),
            ),
            user=user,
            max_tokens=self.settings.semantic_max_output_tokens,
            request_label=label,
        )

    async def mine(self, transcript):
        settings = self.settings
        windows = build_windows(
            transcript,
            settings.extraction_window_minutes * 60,
            settings.extraction_overlap_minutes * 60,
        )
        atoms = []
        diagnostics = []
        warnings = []
        for window in windows:
            previous = []
            hints = []
            for page in range(settings.overflow_max_passes):
                user = f"WINDOW {window.label}; absolute source timestamps:\n{window.text}\nAlready extracted: {atom_payload(previous)}\nOverflow hints: {json.dumps(hints, ensure_ascii=False)}"
                result = await self._stage(
                    SemanticExtraction,
                    "semantic_extraction.md",
                    user,
                    f"semantic_extraction/window={window.index + 1}/page={page + 1}",
                )
                diagnostics.append(result.diagnostics)
                add_atoms(
                    atoms, result.atoms, f"w{window.index + 1}p{page + 1}", transcript.duration
                )
                previous.extend(result.atoms)
                hints = result.overflow_hints
                if not result.diagnostics.overflow:
                    break
            else:
                warnings.append(
                    f"Window {window.index + 1}: overflow continuation exhausted; hints preserved for audit: {hints}"
                )
        if not windows:
            return SemanticResult(diagnostics=diagnostics_union(diagnostics), warnings=warnings)
        patch = await self._stage(
            MergePatch,
            "relationship_merge.md",
            self._input(transcript, atoms),
            "relationship_merge",
        )
        add_atoms(atoms, patch.synthesis_atoms, "merge", transcript.duration)
        atoms = apply_merge(atoms, patch)
        audit = await self._stage(
            CoveragePatch,
            "coverage_audit.md",
            self._input(transcript, atoms),
            "coverage_audit/primary",
        )
        added = len(audit.omissions)
        atoms = self._apply_audit(atoms, audit, transcript, "audit")
        diag = audit.diagnostics
        # Unresolved overflow remains visible even when an auditor returns optimistic defaults.
        if warnings:
            diag.overflow = True
        reasons = escalation_reasons(
            settings, transcript, atoms, diagnostics_union([*diagnostics, diag])
        )
        role = choose_auditor(settings, reasons)
        auditor = "primary"
        if role:
            if not hasattr(self._llm, "audit_view"):
                raise LLMError("Configured provider does not expose quality auditing")
            try:
                premium = await self._stage(
                    CoveragePatch,
                    "coverage_audit.md",
                    self._input(transcript, atoms),
                    f"coverage_audit/{role}",
                    llm=self._llm.audit_view(role),
                )
                added += len(premium.omissions)
                atoms = self._apply_audit(atoms, premium, transcript, "quality")
                diag = premium.diagnostics
                auditor = role
            except LLMError as error:
                warnings.append(f"Quality audit unavailable: {error}")
                auditor = f"{role}:failed"
        if len(atoms) > settings.target_atom_limit or any(
            "overflow continuation exhausted" in warning for warning in warnings
        ):
            diag.overflow = True
        routing = await self._stage(
            RoutingResult, "content_router.md", atom_payload(atoms), "content_router"
        )
        known = {a.id for a in atoms}
        decisions = {r.atom_id: r for r in routing.routes if r.atom_id in known}
        missing = known - decisions.keys()
        if missing:
            warnings.append(f"No routing decision for {sorted(missing)}; preserved in archive")
        routes = [
            decisions.get(
                a.id,
                RouteDecision(
                    atom_id=a.id, route=ContentRoute.ARCHIVE_ONLY, reason="Missing routing decision"
                ),
            )
            for a in atoms
        ]
        return SemanticResult(
            atoms=atoms,
            diagnostics=diag,
            routes=routes,
            auditor=auditor,
            auditor_added_atom_count=added,
            escalation_reasons=reasons,
            warnings=warnings,
        )

    def _apply_audit(self, atoms, audit, transcript, prefix):
        add_atoms(atoms, audit.omissions, prefix, transcript.duration)
        by_id = {a.id: a for a in atoms}
        for correction in audit.corrections:
            if correction.atom_id in by_id and correction.claim.strip():
                by_id[correction.atom_id].claim = correction.claim
        return apply_merge(atoms, MergePatch(relationships=audit.relationships))


def editorial_atoms(result, transcript):
    routes = {r.atom_id: r.route for r in result.routes}
    output = []
    for atom in result.atoms:
        excerpts = [
            " ".join(
                s.text
                for s in transcript.segments
                if s.end > r.start_seconds and s.start <= r.end_seconds
            )
            for r in atom.sources
        ]
        category = (
            AtomCategory.PERSONAL_STORY if atom.kind == "story" else AtomCategory.ORIGINAL_IDEA
        )
        output.append(
            ContentAtom(
                id=atom.id,
                label=atom.title,
                key_claim=atom.claim[:800],
                description=atom.claim[:1200],
                start_seconds=min(r.start_seconds for r in atom.sources),
                end_seconds=max(r.end_seconds for r in atom.sources),
                supporting_context="\n".join(excerpts)[:4000],
                categories=[category],
                confidence=1,
                semantic_kind=atom.kind,
                content_route=routes.get(atom.id, ContentRoute.ARCHIVE_ONLY),
                source_ranges=atom.sources,
                semantic_claim=atom.claim,
                **{name: getattr(atom, name) for name in RELATIONS},
            )
        )
    return output
