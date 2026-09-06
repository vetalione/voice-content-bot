"""Two-stage mining: minimal extraction, then single-atom enrichment.

Text windows are capped at two minutes with at most 15 seconds of overlap.
Generation recovery retries once, then halves text once; it never redoes STT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.models.atoms import AtomEnrichment, AtomExtraction, ContentAtom, ContentAtomSet
from app.models.transcript import Transcript, TranscriptSegment
from app.services.groq_client import GroqGenerationError
from app.utils.text import token_overlap_ratio
from app.utils.timecode import format_timecode

from .base import StructuredAgent

logger = logging.getLogger(__name__)

_DUPLICATE_ATOM_RATIO = 0.72


@dataclass(slots=True)
class TranscriptWindow:
    index: int
    start: float
    end: float
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(f"[{format_timecode(seg.start)}] {seg.text}" for seg in self.segments)

    @property
    def label(self) -> str:
        return f"{format_timecode(self.start)}–{format_timecode(self.end)}"


def build_windows(
    transcript: Transcript, window_seconds: float, overlap_seconds: float
) -> list[TranscriptWindow]:
    """Split a transcript into overlapping time windows for mining."""
    duration = transcript.duration or (transcript.segments[-1].end if transcript.segments else 0.0)
    if not transcript.segments:
        return []
    if duration <= window_seconds:
        return [
            TranscriptWindow(index=0, start=0.0, end=duration, segments=list(transcript.segments))
        ]

    step = max(60.0, window_seconds - overlap_seconds)
    windows: list[TranscriptWindow] = []
    start = 0.0
    index = 0
    while start < duration - 1.0:
        end = min(duration, start + window_seconds)
        segments = transcript.window(start, end)
        if segments:
            windows.append(TranscriptWindow(index=index, start=start, end=end, segments=segments))
            index += 1
        if end >= duration:
            break
        start += step
    return windows


def deduplicate_atoms(atoms: list[ContentAtom]) -> list[ContentAtom]:
    """Drop atoms that restate an atom already kept from an overlapping window."""

    def weight(atom: ContentAtom) -> float:
        return (
            atom.novelty_score + max(atom.business_score, atom.personal_score) + atom.confidence * 5
        )

    ordered = sorted(atoms, key=weight, reverse=True)
    kept: list[ContentAtom] = []
    for atom in ordered:
        fingerprint = f"{atom.label} {atom.key_claim} {atom.description}"
        duplicate = False
        for existing in kept:
            time_overlap = min(atom.end_seconds, existing.end_seconds) - max(
                atom.start_seconds, existing.start_seconds
            )
            if time_overlap <= 0:
                continue
            other = f"{existing.label} {existing.key_claim} {existing.description}"
            if token_overlap_ratio(fingerprint, other) >= _DUPLICATE_ATOM_RATIO:
                duplicate = True
                break
        if not duplicate:
            kept.append(atom)
    return sorted(kept, key=lambda item: item.start_seconds)


class ContentMinerAgent(StructuredAgent):
    prompt_file = "content_miner.md"
    name = "content_miner"

    async def mine(self, transcript: Transcript) -> ContentAtomSet:
        settings = self.settings
        windows = build_windows(
            transcript,
            window_seconds=min(settings.miner_window_minutes * 60, 120),
            overlap_seconds=min(settings.miner_window_overlap_minutes * 60, 15),
        )
        if not windows:
            logger.warning("Nothing to mine: empty transcript")
            return ContentAtomSet()

        system = self.system_prompt()
        collected: list[ContentAtom] = []
        notes: list[str] = []

        for window in windows:
            logger.info("Mining window %s/%s (%s)", window.index + 1, len(windows), window.label)
            user = (
                f"Recording total duration: {format_timecode(transcript.duration)}.\n"
                f"You are analysing window {window.index + 1} of {len(windows)}, "
                f"covering {window.label} of the original recording.\n"
                "Timestamps in square brackets are absolute positions in the full "
                "recording — reuse them, never invent new ones.\n\n"
                "TRANSCRIPT WINDOW:\n"
                f"{window.text}"
            )
            extraction = await self._mine_window(window, system, user)
            enriched = []
            for position, seed in enumerate(extraction):
                start = min(max(seed.start_seconds, window.start), window.end)
                end = min(max(seed.end_seconds, start), window.end)
                excerpt = " ".join(
                    seg.text for seg in window.segments if seg.end > start and seg.start <= end
                )[:1200]
                metadata = await self.request(
                    AtomEnrichment,
                    system=self.prompts.render(
                        "content_enrichment.md",
                        voice_style=self.prompts.voice_style("content_miner"),
                    ),
                    user=f"Title: {seed.title}\nIdea: {seed.idea}\nType: {seed.type}\nSource excerpt: {excerpt}",
                    max_tokens=self.settings.groq_mining_max_tokens,
                    request_label=f"content_enrichment/window={window.index + 1}/atom={position + 1}",
                )
                enriched.append(
                    ContentAtom(
                        id=f"a{position + 1}",
                        label=seed.title,
                        key_claim=seed.idea,
                        description=seed.idea,
                        supporting_context=excerpt,
                        start_seconds=start,
                        end_seconds=end,
                        **metadata.model_dump(),
                    )
                )
            result = ContentAtomSet(atoms=enriched)
            for position, atom in enumerate(result.atoms):
                atom.id = f"w{window.index + 1}a{position + 1}"
                # Clamp hallucinated timestamps back into this window.
                atom.start_seconds = min(max(atom.start_seconds, window.start), window.end)
                atom.end_seconds = min(max(atom.end_seconds, atom.start_seconds), window.end)
                collected.append(atom)
            if result.notes:
                notes.append(f"[{window.label}] {result.notes}")

        deduped = deduplicate_atoms(collected)
        for position, atom in enumerate(deduped, start=1):
            atom.id = f"a{position}"
        logger.info(
            "Mined %s atom(s) from %s window(s) (%s before dedupe)",
            len(deduped),
            len(windows),
            len(collected),
        )
        return ContentAtomSet(atoms=deduped, notes="\n".join(notes)[:2000])

    async def _extract(self, window, system, user, attempt, output_budget=None):
        result = await self.request(
            AtomExtraction,
            system=system,
            user=user,
            max_tokens=output_budget or self.settings.groq_extraction_max_tokens,
            request_label=f"content_extraction/window={window.index + 1}:{window.label}/attempt={attempt}",
            repair_attempts=1,
        )
        for atom in result.atoms:
            atom.start_seconds = min(max(atom.start_seconds, window.start), window.end)
            atom.end_seconds = min(max(atom.end_seconds, atom.start_seconds), window.end)
        return result

    async def _mine_window(self, window, system, user):
        # The shared client's scheduler retains failed reservations and applies a
        # cooldown before each subsequent attempt. No audio/STT work occurs here.
        for attempt in (1, 2):
            try:
                budget = self.settings.groq_extraction_max_tokens if attempt == 1 else 1600
                return (await self._extract(window, system, user, attempt, budget)).atoms
            except GroqGenerationError:
                if attempt == 1:
                    logger.warning(
                        "Extraction window %s failed; retry same text once with output_budget=1600 through TPM scheduler",
                        window.label,
                    )
        if len(window.segments) < 2:
            raise GroqGenerationError("Extraction failed twice; text window cannot be reduced")
        mid = len(window.segments) // 2
        atoms = []
        logger.warning("Extraction window %s failed twice; reduce TEXT window once", window.label)
        for index, segments in enumerate((window.segments[:mid], window.segments[mid:]), 1):
            part = TranscriptWindow(window.index, segments[0].start, segments[-1].end, segments)
            result = await self._extract(
                part, system, f"TRANSCRIPT WINDOW:\n{part.text}", f"reduced-{index}"
            )
            atoms.extend(result.atoms)
        # Each reduced window separately observes the six-atom limit.
        return atoms
