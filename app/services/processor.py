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
        text = getattr(self._pipeline._miner, "_llm", None)
        if hasattr(text, "ensure_primary"):
            await text.ensure_primary()
        await self._check_budget(enforce=True)
        result = await self._pipeline.run(request)
        await self._check_budget(enforce=False)
        await self._deliver(request, result)
        return result

    async def _check_budget(self, *, enforce):
        from datetime import UTC, datetime

        # Exhausted paid credits must not block an explicitly free-only profile.
        if not self._settings.openrouter_allow_paid:
            return
        text = getattr(self._pipeline._miner, "_llm", None)
        if not hasattr(text, "monthly_usage"):
            return
        used = await text.monthly_usage()
        if used is None:
            return
        s = self._settings
        threshold = (
            s.monthly_llm_budget_usd
            if used >= s.monthly_llm_budget_usd
            else s.soft_budget_warning_usd
        )
        if used >= threshold:
            key = f"{datetime.now(UTC):%Y-%m}:{threshold}"
            notified = await self._pipeline.checkpoints.get("budget-notifications", key)
            if not notified:
                try:
                    await self._delivery.send_owner_html(
                        f"<b>Бюджет OpenRouter</b>: ${used:.2f} за месяц по данным API; ориентир ${s.monthly_llm_budget_usd:.2f}."
                    )
                    await self._pipeline.checkpoints.put(
                        "budget-notifications", key, {"notified": True, "reported_usage": used}
                    )
                except Exception:
                    logger.exception("Could not deliver budget warning")
        if enforce and used >= s.monthly_llm_budget_usd and s.stop_on_budget_exceeded:
            raise RuntimeError(
                "Configured monthly budget reached; processing paused by STOP_ON_BUDGET_EXCEEDED"
            )

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
