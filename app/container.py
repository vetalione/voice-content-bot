"""Composition root.

Everything is wired here and nowhere else, so swapping an implementation (for
example a Postgres-backed :class:`DedupeStore`, or a different transcriber) is a
one-line change and does not touch the pipeline or the handlers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aiogram import Bot, Dispatcher

from app.agents.channel_teaser import ChannelTeaserAgent
from app.agents.content_miner import ContentMinerAgent
from app.agents.reels_editor import ReelsEditorAgent
from app.agents.threads_editor import ThreadsEditorAgent
from app.config import Settings
from app.services.audio import AudioProcessor
from app.services.dedupe import DedupeStore, TTLDedupeStore
from app.services.groq_client import GroqClient
from app.services.jobs import JobRunner
from app.services.pipeline import ContentPipeline
from app.services.processor import RecordingProcessor
from app.services.prompts import PromptLibrary
from app.services.transcription import GroqTranscriber
from app.telegram.delivery import TelegramDelivery, build_bot
from app.telegram.downloader import TelegramFileDownloader
from app.telegram.handlers import JobSubmitter, build_router

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    bot: Bot
    dispatcher: Dispatcher
    runner: JobRunner
    dedupe: DedupeStore
    groq: GroqClient
    audio: AudioProcessor
    pipeline: ContentPipeline
    processor: RecordingProcessor
    delivery: TelegramDelivery
    prompts: PromptLibrary

    async def startup(self) -> None:
        await self.runner.start()

    async def shutdown(self) -> None:
        await self.runner.stop()
        await self.groq.aclose()
        await self.bot.session.close()


def build_container(settings: Settings) -> Container:
    missing = settings.missing_required()
    if missing:
        # Not fatal: /health must still answer so a misconfigured Render deploy
        # is debuggable instead of crash-looping.
        logger.error("Missing required environment variables: %s", ", ".join(missing))

    bot = build_bot(settings)
    delivery = TelegramDelivery(bot, settings)
    groq = GroqClient(settings)
    prompts = PromptLibrary(settings.prompts_dir)
    audio = AudioProcessor(settings)

    pipeline = ContentPipeline(
        settings=settings,
        downloader=TelegramFileDownloader(bot, settings),
        audio=audio,
        transcriber=GroqTranscriber(groq, settings),
        miner=ContentMinerAgent(groq, prompts, settings),
        teaser_agent=ChannelTeaserAgent(groq, prompts, settings),
        threads_agent=ThreadsEditorAgent(groq, prompts, settings),
        reels_agent=ReelsEditorAgent(groq, prompts, settings),
        delivery=delivery,
    )

    dedupe = TTLDedupeStore(
        ttl_seconds=settings.dedupe_ttl_seconds,
        max_entries=settings.dedupe_max_entries,
    )
    runner = JobRunner(concurrency=settings.job_concurrency, queue_size=settings.job_queue_size)
    processor = RecordingProcessor(settings, pipeline, delivery, dedupe)
    submitter = JobSubmitter(settings, runner, processor, dedupe, delivery)

    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(settings, submitter, runner))

    logger.info(
        "Container ready (channel=%s owner=%s allowed_users=%s ffmpeg=%s)",
        settings.allowed_channel_id,
        settings.owner_telegram_id,
        sorted(settings.allowed_user_ids),
        audio.ffmpeg_available,
    )
    return Container(
        settings=settings,
        bot=bot,
        dispatcher=dispatcher,
        runner=runner,
        dedupe=dedupe,
        groq=groq,
        audio=audio,
        pipeline=pipeline,
        processor=processor,
        delivery=delivery,
        prompts=prompts,
    )
