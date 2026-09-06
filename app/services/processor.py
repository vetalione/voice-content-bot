"""Job-level wrapper around the pipeline: reporting, failures, transcript export.

The Telegram handlers know only about this class and the :class:`JobRunner`.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.models.content import PipelineResult
from app.models.media import JobRequest, SourceMode
from app.services.dedupe import DedupeStore
from app.services.pipeline import ContentPipeline
from app.telegram.delivery import DeliveryGateway
from app.telegram.report import render_failure, render_report
from app.utils.timecode import format_timecode

logger = logging.getLogger(__name__)


class RecordingProcessor:
    def __init__(
        self,
        settings: Settings,
        pipeline: ContentPipeline,
        delivery: DeliveryGateway,
        dedupe: DedupeStore,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._delivery = delivery
        self._dedupe = dedupe

    async def process(self, request: JobRequest) -> PipelineResult:
        logger.info(
            "Processing %s job: chat=%s message=%s duration=%ss",
            request.mode.value,
            request.chat_id,
            request.message_id,
            request.media.duration_seconds,
        )
        result = await self._pipeline.run(request)
        await self._deliver(request, result)
        return result

    async def _deliver(self, request: JobRequest, result: PipelineResult) -> None:
        await self._delivery.send_owner_html(render_report(result))

        if self._settings.enable_full_transcript and result.transcript_text:
            filename = f"transcript-{request.mode.value}-{request.message_id}.txt"
            await self._delivery.send_owner_document(
                filename,
                result.transcript_text.encode("utf-8"),
                caption=(
                    f"Полный транскрипт ({format_timecode(result.metadata.duration_seconds)})"
                ),
            )

    async def notify_failure(self, request: JobRequest, error: BaseException) -> None:
        """Tell the owner a job died, and let the message be retried later."""
        # Releasing the dedupe key lets the owner re-forward the same recording.
        self._dedupe.release(request.dedupe_key)
        try:
            await self._delivery.send_owner_html(
                render_failure(request.mode, request.chat_id, request.message_id, error)
            )
        except Exception:
            logger.exception("Could not notify the owner about the failure")

        if request.mode is SourceMode.PRIVATE and request.requester_id:
            owner = self._settings.owner_telegram_id
            if request.requester_id != owner:
                try:
                    await self._delivery.send_plain(
                        request.requester_id,
                        "Не получилось обработать запись. Подробности отправлены владельцу.",
                    )
                except Exception:
                    logger.exception("Could not notify the requester")
