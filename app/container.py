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
from app.agents.semantic_miner import SemanticMinerAgent
from app.agents.threads_editor import ThreadsEditorAgent
from app.config import Settings
from app.services.audio import AudioProcessor
from app.services.claude_client import ClaudeAgentClient
from app.services.dedupe import DedupeStore, TTLDedupeStore
from app.services.groq_client import GroqClient
from app.services.jobs import JobRunner
from app.services.llm import LLMClient
from app.services.model_preferences import ModelPreferences
from app.services.openrouter_client import OpenRouterClient
from app.services.pipeline import ContentPipeline
from app.services.processor import RecordingProcessor
from app.services.prompts import PromptLibrary
from app.services.provider_routing import ModelCatalog, TextRouter
from app.services.transcription import GroqTranscriber
from app.telegram.delivery import TelegramDelivery, build_bot
from app.telegram.downloader import TelegramFileDownloader
from app.telegram.handlers import JobSubmitter, build_router
from app.telegram.webhook import sync_webhook_updates

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    bot: Bot
    dispatcher: Dispatcher
    runner: JobRunner
    dedupe: DedupeStore
    groq: GroqClient
    text: LLMClient
    audio: AudioProcessor
    pipeline: ContentPipeline
    processor: RecordingProcessor
    delivery: TelegramDelivery
    prompts: PromptLibrary
    preferences: ModelPreferences | None = None

    async def startup(self) -> None:
        if hasattr(self.text, "preflight"):
            await self.text.preflight()
        await self.runner.start()
        await sync_webhook_updates(self.bot, self.settings)

    async def shutdown(self) -> None:
        await self.runner.stop()
        if self.text is not self.groq:
            await self.text.aclose()
        if self.preferences and self.preferences.catalog is not getattr(self.text, "catalog", None):
            await self.preferences.catalog.aclose()
        await self.groq.aclose()
        await self.bot.session.close()
        await self.pipeline.checkpoints.aclose()


def build_container(settings: Settings) -> Container:
    missing = settings.missing_required()
    if missing:
        # Not fatal: /health must still answer so a misconfigured Render deploy
        # is debuggable instead of crash-looping.
        logger.error("Missing required environment variables: %s", ", ".join(missing))

    bot = build_bot(settings)
    delivery = TelegramDelivery(bot, settings)
    groq = GroqClient(settings)
    if settings.text_provider == "openrouter" and settings.semantic_pipeline_enabled:
        text = TextRouter(
            settings, claude=ClaudeAgentClient(settings) if settings.claude_enabled else None
        )
    else:
        text = OpenRouterClient(settings) if settings.text_provider == "openrouter" else groq
    prompts = PromptLibrary(settings.prompts_dir)
    audio = AudioProcessor(settings)

    pipeline = ContentPipeline(
        settings=settings,
        downloader=TelegramFileDownloader(bot, settings),
        audio=audio,
        transcriber=GroqTranscriber(groq, settings),
        miner=(SemanticMinerAgent if settings.semantic_pipeline_enabled else ContentMinerAgent)(
            text, prompts, settings
        ),
        teaser_agent=ChannelTeaserAgent(text, prompts, settings),
        threads_agent=ThreadsEditorAgent(text, prompts, settings),
        reels_agent=ReelsEditorAgent(text, prompts, settings),
        delivery=delivery,
    )

    dedupe = TTLDedupeStore(
        ttl_seconds=settings.dedupe_ttl_seconds,
        max_entries=settings.dedupe_max_entries,
    )
    runner = JobRunner(concurrency=settings.job_concurrency, queue_size=settings.job_queue_size)
    processor = RecordingProcessor(settings, pipeline, delivery, dedupe)
    preferences = (
        ModelPreferences(
            settings, pipeline.checkpoints, getattr(text, "catalog", None) or ModelCatalog()
        )
        if settings.text_provider == "openrouter"
        else None
    )

    async def process_selected(request, selected):
        # Independent agents/settings per accepted job: menu changes never mutate
        # in-flight work, including concurrent workers. Storage and STT are shared.
        selected_text = (
            TextRouter(
                selected, claude=ClaudeAgentClient(selected) if selected.claude_enabled else None
            )
            if selected.semantic_pipeline_enabled
            else OpenRouterClient(selected)
        )
        try:
            selected_pipeline = ContentPipeline(
                settings=selected,
                downloader=pipeline._downloader,
                audio=audio,
                transcriber=pipeline._transcriber,
                miner=(
                    SemanticMinerAgent if selected.semantic_pipeline_enabled else ContentMinerAgent
                )(selected_text, prompts, selected),
                teaser_agent=ChannelTeaserAgent(selected_text, prompts, selected),
                threads_agent=ThreadsEditorAgent(selected_text, prompts, selected),
                reels_agent=ReelsEditorAgent(selected_text, prompts, selected),
                delivery=delivery,
                checkpoint_store=pipeline.checkpoints,
            )
            return await RecordingProcessor(selected, selected_pipeline, delivery, dedupe).process(
                request
            )
        finally:
            await selected_text.aclose()

    submitter = JobSubmitter(
        settings, runner, processor, dedupe, delivery, preferences, process_selected
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(settings, submitter, runner, preferences))

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
        text=text,
        audio=audio,
        pipeline=pipeline,
        processor=processor,
        delivery=delivery,
        prompts=prompts,
        preferences=preferences,
    )
