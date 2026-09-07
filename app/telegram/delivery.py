"""Outbound Telegram traffic: owner reports and public channel publishing.

Every outbound call in the project goes through :class:`DeliveryGateway`, which
gives tests a single seam to assert on — in particular that private mode never
publishes to the channel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import BufferedInputFile, ReplyParameters

from app.config import Settings
from app.telegram.formatting import TELEGRAM_MESSAGE_LIMIT, split_html_message

logger = logging.getLogger(__name__)

# Telegram tolerates ~1 message/second to the same chat.
_SEND_PAUSE_SECONDS = 0.4


class DeliveryGateway(Protocol):
    async def send_owner_html(self, text: str) -> None: ...
    async def send_owner_document(
        self, filename: str, content: bytes, caption: str = ""
    ) -> None: ...
    async def send_plain(self, chat_id: int, text: str) -> None: ...
    async def publish_to_channel(
        self, text: str, reply_to_message_id: int | None = None
    ) -> int | None: ...


def build_bot(settings: Settings) -> Bot:
    """Bot instance with HTML as the default parse mode."""
    return Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


class TelegramDelivery:
    """Concrete gateway backed by aiogram."""

    def __init__(self, bot: Bot, settings: Settings) -> None:
        self._bot = bot
        self._settings = settings

    async def _send_with_retry(self, chat_id: int, text: str, **kwargs: object) -> int | None:
        for attempt in (1, 2, 3):
            try:
                message = await self._bot.send_message(chat_id=chat_id, text=text, **kwargs)  # type: ignore[arg-type]
                return message.message_id
            except TelegramRetryAfter as error:
                delay = float(error.retry_after) + 1
                logger.warning("Flood limit, sleeping %.0fs (attempt %s)", delay, attempt)
                await asyncio.sleep(delay)
            except TelegramAPIError:
                raise
        logger.error("Gave up sending message to %s after flood retries", chat_id)
        return None

    async def send_owner_html(self, text: str) -> None:
        owner = self._settings.owner_telegram_id
        if not owner:
            logger.error("OWNER_TELEGRAM_ID is not configured; dropping report")
            return
        parts = split_html_message(text, TELEGRAM_MESSAGE_LIMIT)
        for index, part in enumerate(parts):
            await self._send_with_retry(
                owner,
                part,
                parse_mode=ParseMode.HTML,
                link_preview_options=None,
                disable_notification=index > 0,
            )
            if index + 1 < len(parts):
                await asyncio.sleep(_SEND_PAUSE_SECONDS)
        logger.info("Sent owner report in %s message(s)", len(parts))

    async def send_owner_document(self, filename: str, content: bytes, caption: str = "") -> None:
        owner = self._settings.owner_telegram_id
        if not owner:
            return
        await self._bot.send_document(
            chat_id=owner,
            document=BufferedInputFile(content, filename=filename),
            caption=caption[:1024] or None,
        )
        logger.info("Sent document %s (%.1f KB) to owner", filename, len(content) / 1024)

    async def send_plain(self, chat_id: int, text: str) -> None:
        for part in split_html_message(text, TELEGRAM_MESSAGE_LIMIT) or [text]:
            await self._send_with_retry(chat_id, part, parse_mode=None)

    async def publish_to_channel(
        self, text: str, reply_to_message_id: int | None = None
    ) -> int | None:
        """Publish as a reply when configured; never silently create a standalone post."""
        channel_id = self._settings.allowed_channel_id
        if self._settings.dry_run_publish:
            logger.warning("DRY_RUN_PUBLISH is on — teaser not published")
            return None

        reply_parameters = None
        if reply_to_message_id and self._settings.teaser_reply_to_source:
            reply_parameters = ReplyParameters(
                message_id=reply_to_message_id, allow_sending_without_reply=False
            )

        message_id = await self._send_with_retry(
            channel_id,
            text,
            parse_mode=None,
            reply_parameters=reply_parameters,
        )
        logger.info("Published teaser to channel %s as %s", channel_id, message_id)
        return message_id
