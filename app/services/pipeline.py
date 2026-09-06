"""The shared processing pipeline.

Channel mode and private mode run the *same* code path — download, prepare,
transcribe, merge, mine, then the three editors. The only difference is a single
guard: :attr:`JobRequest.may_publish`. That is what makes the private mode a
real test of production behaviour rather than a parallel implementation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.agents.channel_teaser import ChannelTeaserAgent
from app.agents.content_miner import ContentMinerAgent
from app.agents.reels_editor import ReelsEditorAgent
from app.agents.semantic_miner import editorial_atoms
from app.agents.threads_editor import ThreadsEditorAgent
from app.config import Settings
from app.models.content import PipelineResult
from app.models.media import JobRequest, RecordingMetadata
from app.models.semantic import SemanticResult
from app.models.transcript import ChunkTranscript, Transcript
from app.services.audio import AudioProcessor
from app.services.checkpoints import (
    active_checkpoint,
    build_store,
    checkpoint_scope,
    fingerprint,
    recording_id,
)
from app.services.transcript_merge import merge_chunk_transcripts
from app.services.transcription import Transcriber
from app.services.usage import recording_usage, restore_usage
from app.telegram.delivery import DeliveryGateway
from app.telegram.downloader import FileDownloader
from app.utils.tempfiles import workspace

logger = logging.getLogger(__name__)


class EmptyTranscriptError(RuntimeError):
    """Transcription produced no usable text."""


class ContentPipeline:
    """Orchestrates one recording end to end."""

    def __init__(
        self,
        settings: Settings,
        downloader: FileDownloader,
        audio: AudioProcessor,
        transcriber: Transcriber,
        miner: ContentMinerAgent,
        teaser_agent: ChannelTeaserAgent,
        threads_agent: ThreadsEditorAgent,
        reels_agent: ReelsEditorAgent,
        delivery: DeliveryGateway,
        checkpoint_store=None,
    ) -> None:
        self._settings = settings
        self._downloader = downloader
        self._audio = audio
        self._transcriber = transcriber
        self._miner = miner
        self._teaser_agent = teaser_agent
        self._threads_agent = threads_agent
        self._reels_agent = reels_agent
        self._delivery = delivery
        self.checkpoints = checkpoint_store or build_store(settings)

    # ------------------------------------------------------------------- stages
    async def transcribe(
        self, source: Path, workdir: Path, reported_duration: float | None
    ) -> tuple[Transcript, int]:
        """Prepare the audio, transcribe every chunk, merge into one transcript."""
        prepared = await self._audio.prepare(source, workdir, reported_duration=reported_duration)
        chunk_transcripts: list[ChunkTranscript] = []
        for chunk in prepared.chunks:
            logger.info(
                "Transcribing chunk %s/%s at +%.0fs",
                chunk.index + 1,
                prepared.chunk_count,
                chunk.offset_seconds,
            )
            scope = active_checkpoint.get()
            key = "stt_chunk:" + fingerprint(
                [
                    chunk.index,
                    chunk.offset_seconds,
                    self._settings.groq_whisper_model,
                    self._settings.transcript_language,
                    self._settings.audio_chunk_minutes,
                    self._settings.audio_chunk_overlap_seconds,
                    self._settings.audio_target_codec,
                ]
            )
            cached = await scope[0].get(scope[1], key) if scope else None
            if cached:
                part = ChunkTranscript.model_validate(cached)
            else:
                part = await self._transcriber.transcribe_chunk(chunk)
                if scope:
                    await scope[0].put(scope[1], key, part.model_dump(mode="json"))
            chunk_transcripts.append(part)

        transcript = merge_chunk_transcripts(chunk_transcripts)
        if not transcript.duration and prepared.duration:
            transcript.duration = prepared.duration
        if not transcript.text.strip():
            raise EmptyTranscriptError(
                "Groq returned an empty transcript — the recording may be silent "
                "or in an unsupported format."
            )
        logger.info(
            "Transcript ready: %s words, %s segments, %.0fs",
            transcript.word_count,
            len(transcript.segments),
            transcript.duration,
        )
        return transcript, prepared.chunk_count

    async def analyse(
        self, request: JobRequest, transcript: Transcript, chunks: int, started: float
    ) -> PipelineResult:
        """Mine atoms and run the three editors."""
        settings = self._settings
        atom_set = await self._miner.mine(transcript)
        semantic = atom_set if isinstance(atom_set, SemanticResult) else None
        usable = (
            editorial_atoms(semantic, transcript)
            if semantic is not None
            else atom_set.usable(settings.min_atom_confidence)
        )
        logger.info(
            "Atoms: %s total, %s usable (min confidence %.2f)",
            len(atom_set.atoms),
            len(usable),
            settings.min_atom_confidence,
        )

        warnings: list[str] = list(semantic.warnings) if semantic else []
        if not usable and semantic is None:
            warnings.append(
                "Ни один контент-атом не прошёл порог уверенности — "
                "тизер и черновики построены на всём, что нашлось."
            )
            usable = [atom for atom in atom_set.atoms if not atom.should_ignore]

        teaser = None
        if usable:
            teaser = await self._teaser_agent.write(transcript, usable)

        threads_batch = await self._threads_agent.edit(usable)
        reels_batch = await self._reels_agent.edit(usable)

        metadata = RecordingMetadata(
            mode=request.mode,
            source_chat_id=request.chat_id,
            source_message_id=request.message_id,
            chat_title=request.chat_title,
            posted_at=request.posted_at,
            duration_seconds=transcript.duration,
            detected_language=transcript.language,
            transcript_chars=len(transcript.text),
            transcript_words=transcript.word_count,
            chunks=chunks,
            atoms_found=len(atom_set.atoms),
            atoms_used=len(usable),
            forwarded=request.forwarded,
            kind=request.media.kind.value,
            processing_seconds=time.monotonic() - started,
        )
        return PipelineResult(
            metadata=metadata,
            atoms=usable,
            semantic=semantic,
            teaser=teaser,
            threads=threads_batch.best(settings.max_threads_candidates),
            reels=reels_batch.best(settings.max_reels_candidates),
            warnings=warnings,
            transcript_text=transcript.text,
        )

    async def publish(self, request: JobRequest, result: PipelineResult) -> None:
        """Publish the teaser — channel mode only, enforced here."""
        if not request.may_publish:
            logger.info("Private mode: skipping channel publication by design")
            return
        if result.teaser is None:
            result.warnings.append("Тизер не сгенерирован, публикация пропущена.")
            return

        body = result.teaser.render(include_timestamps=self._settings.teaser_include_timestamps)
        message_id = await self._delivery.publish_to_channel(
            body, reply_to_message_id=request.message_id
        )
        result.metadata.published = message_id is not None
        result.metadata.published_message_id = message_id
        if message_id is None and not self._settings.dry_run_publish:
            result.warnings.append("Telegram не вернул id опубликованного тизера.")

    # -------------------------------------------------------------------- entry
    async def run(self, request: JobRequest) -> PipelineResult:
        job = recording_id(request)
        with (
            checkpoint_scope(self.checkpoints, job),
            recording_usage(request.media.duration_seconds or 0, job) as usage,
        ):
            await restore_usage(self.checkpoints, job, usage)
            await self.checkpoints.put(job, "request", request.model_dump(mode="json"))
            await self.checkpoints.put(job, "status", {"status": "running"})
            try:
                result = await self._run(request)
            except BaseException as error:
                await self.checkpoints.put(
                    job, "status", {"status": "interrupted", "error_type": type(error).__name__}
                )
                raise
            await self.checkpoints.put(job, "status", {"status": "complete"})
            result.recording_id = job
            result.usage_diagnostics = await self.checkpoints.get(job, "usage") or {
                "events": usage.events
            }
            await self.checkpoints.put(job, "latest_result", result.model_dump(mode="json"))
            return result

    async def _run(self, request: JobRequest) -> PipelineResult:
        """Full flow for one recording. Temp files are always cleaned up."""
        started = time.monotonic()
        settings = self._settings
        with workspace(
            settings.work_dir,
            prefix=f"{request.mode.value}-{request.message_id}",
            keep=settings.keep_temp_files,
        ) as workdir:
            job = recording_id(request)
            signature = fingerprint(
                [
                    settings.groq_whisper_model,
                    settings.transcript_language,
                    settings.audio_chunk_minutes,
                    settings.audio_chunk_overlap_seconds,
                    settings.audio_target_codec,
                ]
            )
            saved = await self.checkpoints.get(job, "transcript")
            if saved and saved.get("signature") == signature:
                transcript = Transcript.model_validate(saved["transcript"])
                chunks = saved["chunks"]
                logger.info("Checkpoint hit: transcription job=%s", job)
            else:
                source = await self._downloader.download(request.media, workdir)
                transcript, chunks = await self.transcribe(
                    source, workdir, reported_duration=request.media.duration_seconds
                )
                await self.checkpoints.put(
                    job,
                    "transcript",
                    {
                        "signature": signature,
                        "transcript": transcript.model_dump(mode="json"),
                        "chunks": chunks,
                    },
                )
            result = await self.analyse(request, transcript, chunks, started)
            await self.checkpoints.put(job, "analysis", result.model_dump(mode="json"))
            published = await self.checkpoints.get(job, "published:" + request.dedupe_key)
            if published:
                result.metadata.published = True
                result.metadata.published_message_id = published["message_id"]
            else:
                await self.publish(request, result)
                if result.metadata.published:
                    await self.checkpoints.put(
                        job,
                        "published:" + request.dedupe_key,
                        {"message_id": result.metadata.published_message_id},
                    )
        result.metadata.processing_seconds = time.monotonic() - started
        return result
