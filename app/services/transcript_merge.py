"""Merge per-chunk transcripts into one continuous, correctly timed transcript.

Two problems are solved here:

* **Offsets.** Groq returns timestamps relative to the uploaded chunk. Every
  segment is shifted by the chunk's ``offset_seconds`` so timecodes refer to the
  original recording.
* **Overlap.** Chunks intentionally overlap by a few seconds so no word is cut
  in half. That means the seam is transcribed twice. Segments that fall inside
  the already-covered region and repeat text we have are dropped, and whatever
  duplicated words survive at the exact seam are trimmed token by token.
"""

from __future__ import annotations

import logging

from app.models.transcript import ChunkTranscript, Transcript, TranscriptSegment
from app.utils.text import (
    drop_leading_tokens,
    longest_boundary_repeat,
    normalize,
    token_overlap_ratio,
)

logger = logging.getLogger(__name__)

# How far past the previous chunk's end a segment may reach and still be
# considered part of the duplicated overlap region.
_END_TOLERANCE_SECONDS = 0.75
# Token overlap above which two texts count as "the same thing said twice".
_DUPLICATE_RATIO = 0.7
# Number of trailing characters of the merged text used as the comparison tail.
_TAIL_CHARS = 600


def _is_duplicate(candidate: str, tail: str) -> bool:
    normalized = normalize(candidate)
    if not normalized:
        return True
    if normalized in normalize(tail):
        return True
    return token_overlap_ratio(candidate, tail) >= _DUPLICATE_RATIO


def merge_chunk_transcripts(chunks: list[ChunkTranscript]) -> Transcript:
    """Combine chunk transcripts in chronological order.

    Input timestamps are chunk-relative; output timestamps are absolute.
    """
    ordered = sorted(chunks, key=lambda chunk: (chunk.offset_seconds, chunk.index))
    segments: list[TranscriptSegment] = []
    language: str | None = None
    duration = 0.0
    dropped = 0

    for position, chunk in enumerate(ordered):
        language = language or chunk.language
        duration = max(duration, chunk.offset_seconds + chunk.duration)

        shifted = [seg.shifted(chunk.offset_seconds) for seg in chunk.segments if seg.text]
        shifted.sort(key=lambda seg: (seg.start, seg.end))

        if position == 0 or not segments:
            segments.extend(shifted)
            continue

        covered_until = segments[-1].end
        tail = " ".join(seg.text for seg in segments[-8:])[-_TAIL_CHARS:]

        kept: list[TranscriptSegment] = []
        for seg in shifted:
            inside_overlap = seg.end <= covered_until + _END_TOLERANCE_SECONDS
            if inside_overlap and _is_duplicate(seg.text, tail):
                dropped += 1
                continue
            kept.append(seg)

        if kept:
            repeat = longest_boundary_repeat(tail, kept[0].text)
            if repeat:
                trimmed = drop_leading_tokens(kept[0].text, repeat)
                logger.debug("Trimmed %s duplicated word(s) at chunk seam", repeat)
                if trimmed:
                    kept[0] = TranscriptSegment(start=kept[0].start, end=kept[0].end, text=trimmed)
                else:
                    kept = kept[1:]

        # Never let a seam produce a timestamp that goes backwards.
        for seg in kept:
            if seg.start < covered_until:
                seg.start = min(covered_until, seg.end)
            segments.append(seg)
            covered_until = max(covered_until, seg.end)

    if segments:
        duration = max(duration, segments[-1].end)

    if dropped:
        logger.info("Dropped %s duplicated overlap segment(s)", dropped)

    return Transcript(segments=segments, language=language, duration=duration)


def transcript_preview(transcript: Transcript, max_chars: int = 800) -> str:
    """Short timestamped excerpt, useful for logs and the owner report."""
    lines: list[str] = []
    total = 0
    for seg in transcript.segments:
        line = f"[{seg.start:.0f}s] {seg.text}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def render_timestamped_transcript(transcript: Transcript) -> str:
    """Full transcript with ``MM:SS`` prefixes, for the optional .txt export."""
    from app.utils.timecode import format_timecode

    return "\n".join(f"[{format_timecode(seg.start)}] {seg.text}" for seg in transcript.segments)
