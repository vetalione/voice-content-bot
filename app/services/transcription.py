"""Transcription service: audio chunks in, timestamped chunk transcripts out."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.config import Settings
from app.models.transcript import ChunkTranscript, TranscriptSegment
from app.services.audio import AudioChunk
from app.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    """Injected into the pipeline so tests can substitute a fake."""

    async def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript: ...


def _parse_segments(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments") or []
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        segments.append(TranscriptSegment(start=start, end=max(start, end), text=text))
    return segments


class GroqTranscriber:
    """Groq Speech-to-Text with segment-level timestamps."""

    def __init__(self, client: GroqClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript:
        settings = self._settings
        payload = await self._client.transcribe_file(
            chunk.path,
            model=settings.groq_whisper_model,
            language=settings.transcript_language or None,
        )
        segments = _parse_segments(payload)
        text = str(payload.get("text", "") or "").strip()

        if not segments and text:
            # Timestamps are unavailable for this model/format: approximate one
            # segment spanning the whole chunk so downstream stages still work.
            logger.warning(
                "Chunk %s returned no segments; using a single approximate segment",
                chunk.index,
            )
            segments = [TranscriptSegment(start=0.0, end=chunk.duration or 0.0, text=text)]

        duration = chunk.duration or (segments[-1].end if segments else 0.0)
        return ChunkTranscript(
            index=chunk.index,
            offset_seconds=chunk.offset_seconds,
            overlap_seconds=chunk.overlap_seconds,
            duration=duration,
            language=payload.get("language"),
            segments=segments,
            text=text,
        )
