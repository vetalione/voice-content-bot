"""aiogram routers.

Handlers do almost nothing: they validate, build a :class:`JobRequest`, claim a
dedupe key and hand the work to the :class:`JobRunner`. All of that finishes in
milliseconds, which is what keeps the webhook response fast.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from app.config import Settings
from app.models.media import JobRequest, MediaRef, SourceMode
from app.services.dedupe import DedupeStore
from app.services.jobs import JobRunner, QueueFullError
from app.services.processor import RecordingProcessor
from app.telegram.delivery import DeliveryGateway
from app.telegram.filters import (
    AllowedChannelFilter,
    AllowedPrivateUserFilter,
    HasSupportedMediaFilter,
)
from app.utils.timecode import format_timecode

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Пришли или перешли сюда голосовое / аудио — я расшифрую его, найду "
    "контент-атомы и верну черновики для Threads и Reels.\n\n"
    "В личке я ничего не публикую в канал: это тестовый и архивный режим.\n"
    "Новые голосовые в канале обрабатываются автоматически.\n\n"
    "/status — очередь и настройки"
)


def _is_forwarded(message: Message) -> bool:
    if getattr(message, "forward_origin", None) is not None:
        return True
    return bool(
        getattr(message, "forward_from", None)
        or getattr(message, "forward_from_chat", None)
        or getattr(message, "forward_date", None)
    )


class JobSubmitter:
    """Claims the dedupe key and enqueues the pipeline job."""

    def __init__(
        self,
        settings: Settings,
        runner: JobRunner,
        processor: RecordingProcessor,
        dedupe: DedupeStore,
        delivery: DeliveryGateway,
    ) -> None:
        self._settings = settings
        self._runner = runner
        self._processor = processor
        self._dedupe = dedupe
        self._delivery = delivery

    async def submit(self, request: JobRequest) -> bool:
        """Returns True when the job was accepted."""
        if not self._dedupe.claim(request.dedupe_key):
            logger.info("Skipping duplicate delivery of %s", request.dedupe_key)
            return False

        async def run() -> None:
            await self._processor.process(request)

        async def on_error(error: BaseException) -> None:
            await self._processor.notify_failure(request, error)

        try:
            self._runner.submit(f"pipeline:{request.dedupe_key}", run, on_error)
        except QueueFullError as error:
            self._dedupe.release(request.dedupe_key)
            logger.error("%s", error)
            await self._delivery.send_owner_html(
                "<b>⚠️ Очередь переполнена</b>\nЗапись "
                f"<code>{request.message_id}</code> не принята в обработку. "
                "Попробуй ещё раз, когда текущая запись закончится."
            )
            return False
        return True


def build_router(settings: Settings, submitter: JobSubmitter, runner: JobRunner) -> Router:
    """Assemble the router. Handler order defines precedence."""
    router = Router(name="voice-content-bot")

    allowed_channel = AllowedChannelFilter(settings.allowed_channel_id)
    allowed_private = AllowedPrivateUserFilter(settings.allowed_user_ids)
    has_media = HasSupportedMediaFilter()

    # ------------------------------------------------------- MODE A: the channel
    @router.channel_post(allowed_channel, has_media)
    async def on_channel_audio(message: Message, media: MediaRef) -> None:
        request = JobRequest(
            mode=SourceMode.CHANNEL,
            chat_id=message.chat.id,
            message_id=message.message_id,
            media=media,
            caption=message.caption or message.text,
            posted_at=message.date,
            chat_title=message.chat.title,
        )
        accepted = await submitter.submit(request)
        logger.info(
            "Channel %s post %s: accepted=%s (%s)",
            media.kind.value,
            message.message_id,
            accepted,
            format_timecode(media.duration_seconds or 0),
        )

    @router.channel_post(allowed_channel)
    async def on_channel_other(message: Message) -> None:
        # Text posts, photos, documents, video notes: nothing to transcribe.
        logger.info(
            "Ignoring unsupported channel post %s in %s",
            message.message_id,
            message.chat.id,
        )

    @router.channel_post()
    async def on_foreign_channel(message: Message) -> None:
        logger.info("Ignoring post from non-allowed channel %s", message.chat.id)

    # Edited channel posts must never trigger a second run.
    @router.edited_channel_post()
    async def on_edited_channel_post(message: Message) -> None:
        logger.info("Ignoring edited channel post %s", message.message_id)

    # ------------------------------------------ MODE B: private test / archive
    @router.message(F.chat.type == "private", allowed_private, Command("start", "help"))
    async def on_private_help(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @router.message(F.chat.type == "private", allowed_private, Command("status"))
    async def on_private_status(message: Message) -> None:
        stats = runner.stats
        await message.answer(
            "\n".join(
                [
                    f"В очереди: {stats.queued}",
                    f"Обрабатывается: {stats.running}",
                    f"Готово: {stats.completed} · Ошибок: {stats.failed}",
                    f"Канал: {settings.allowed_channel_id}",
                    f"STT: {settings.groq_whisper_model}",
                    f"LLM: {settings.groq_llm_model}",
                    f"Лимит загрузки: {settings.max_stt_upload_mb:.0f} МБ · "
                    f"чанк {settings.audio_chunk_minutes:.0f} мин",
                    f"Полный транскрипт: {'вкл' if settings.enable_full_transcript else 'выкл'}",
                ]
            )
        )

    @router.message(F.chat.type == "private", allowed_private, has_media)
    async def on_private_audio(message: Message, media: MediaRef) -> None:
        request = JobRequest(
            mode=SourceMode.PRIVATE,
            chat_id=message.chat.id,
            message_id=message.message_id,
            media=media,
            requester_id=message.from_user.id if message.from_user else None,
            caption=message.caption or message.text,
            posted_at=message.date,
            forwarded=_is_forwarded(message),
            chat_title="личка",
        )
        accepted = await submitter.submit(request)
        if accepted:
            await message.answer(
                "Принял"
                + (" (форвард)" if request.forwarded else "")
                + f", длительность {format_timecode(media.duration_seconds or 0)}.\n"
                "Расшифровываю и разбираю на атомы — пришлю отчёт, когда закончу. "
                "В канал ничего не уйдёт."
            )
        else:
            await message.answer("Эта запись уже в обработке или уже обработана.")

    @router.message(F.chat.type == "private", allowed_private)
    async def on_private_unsupported(message: Message) -> None:
        await message.answer(
            "Поддерживаются только голосовые и аудио. " + HELP_TEXT.split("\n\n")[0]
        )

    @router.message(F.chat.type == "private")
    async def on_unauthorized_private(message: Message) -> None:
        # Silently ignored on purpose: no reply, no download, no Groq call.
        user = message.from_user
        logger.warning(
            "Dropped private message from unauthorized user %s (%s)",
            getattr(user, "id", None),
            getattr(user, "username", None),
        )

    @router.message()
    async def on_other_chat(message: Message) -> None:
        logger.info("Ignoring message from chat %s (%s)", message.chat.id, message.chat.type)

    return router
