"""Content miner: finds independent content atoms across the whole transcript.

Deliberately *not* a summariser. A single summary of a 20-minute recording
collapses six usable ideas into one paragraph, and everything generated from it
inherits that loss. Instead the transcript is walked window by window (default
20 minutes with a 1-minute overlap) and each window is mined for standalone
atoms. Atoms are then de-duplicated across window seams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.models.atoms import ContentAtom, ContentAtomSet
from app.models.transcript import Transcript, TranscriptSegment
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
            window_seconds=settings.miner_window_minutes * 60,
            overlap_seconds=settings.miner_window_overlap_minutes * 60,
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
            result = await self.request(
                ContentAtomSet, system=system, user=user, max_tokens=settings.groq_mining_max_tokens
            )
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
