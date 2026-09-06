"""aiogram filters that form the security boundary of the bot.

These are pure and side-effect free so they can be unit tested without a Bot
instance or a network connection.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from aiogram.filters import BaseFilter
from aiogram.types import Message

from app.models.media import MediaKind, MediaRef

logger = logging.getLogger(__name__)


def extract_media(message: Message) -> MediaRef | None:
    """Return a :class:`MediaRef` for a supported post, else ``None``.

    Only ``voice`` and ``audio`` are supported. Documents, video notes, photos,
    text posts and polls are explicitly out of scope for v1.
    """
    if message.voice is not None:
        voice = message.voice
        return MediaRef(
            kind=MediaKind.VOICE,
            file_id=voice.file_id,
            file_unique_id=voice.file_unique_id,
            file_size=voice.file_size,
            duration_seconds=voice.duration,
            mime_type=voice.mime_type,
        )
    if message.audio is not None:
        audio = message.audio
        return MediaRef(
            kind=MediaKind.AUDIO,
            file_id=audio.file_id,
            file_unique_id=audio.file_unique_id,
            file_size=audio.file_size,
            duration_seconds=audio.duration,
            mime_type=audio.mime_type,
            file_name=audio.file_name,
            title=audio.title,
            performer=audio.performer,
        )
    return None


class AllowedChannelFilter(BaseFilter):
    """Passes only posts from the single configured broadcast channel."""

    def __init__(self, channel_id: int) -> None:
        self.channel_id = channel_id

    async def __call__(self, message: Message) -> bool:
        allowed = message.chat.id == self.channel_id
        if not allowed:
            logger.info(
                "Ignoring post from unrelated chat %s (expected %s)",
                message.chat.id,
                self.channel_id,
            )
        return allowed


class AllowedPrivateUserFilter(BaseFilter):
    """Passes only private messages from an explicitly allow-listed user.

    Unauthorized users are dropped here, *before* any download or Groq call, so
    they can never consume quota.
    """

    def __init__(self, allowed_ids: Iterable[int]) -> None:
        self.allowed_ids = frozenset(int(item) for item in allowed_ids)

    async def __call__(self, message: Message) -> bool:
        if message.chat.type != "private":
            return False
        user = message.from_user
        if user is None or user.id not in self.allowed_ids:
            # The filter is evaluated once per candidate handler, so the warning
            # is emitted by the catch-all handler instead of here.
            logger.debug(
                "Private message from unauthorized user %s did not pass the filter",
                getattr(user, "id", None),
            )
            return False
        return True


class HasSupportedMediaFilter(BaseFilter):
    """Passes voice/audio posts and injects ``media`` into the handler kwargs."""

    async def __call__(self, message: Message) -> bool | dict[str, MediaRef]:
        media = extract_media(message)
        if media is None:
            return False
        return {"media": media}
