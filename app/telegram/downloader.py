"""Telegram file download.

The Bot API refuses ``getFile`` for anything above 20 MB, which is a real limit
for long ``audio`` documents (a 60-minute 128 kbps mp3 is ~57 MB). Voice notes
are opus and stay small, so this mostly affects forwarded music-style files. The
limit is detected up front and reported to the owner instead of failing deep in
the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from aiogram import Bot

from app.config import Settings
from app.models.media import MediaRef

logger = logging.getLogger(__name__)

_MIME_EXTENSIONS = {
    "audio/ogg": ".oga",
    "audio/opus": ".opus",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
}


class FileTooLargeError(RuntimeError):
    """Above the Bot API download limit."""


class FileDownloader(Protocol):
    async def download(self, media: MediaRef, directory: Path) -> Path: ...


def guess_extension(media: MediaRef, telegram_path: str | None = None) -> str:
    if telegram_path:
        suffix = Path(telegram_path).suffix.lower()
        if suffix:
            return suffix
    if media.file_name:
        suffix = Path(media.file_name).suffix.lower()
        if suffix:
            return suffix
    if media.mime_type:
        mapped = _MIME_EXTENSIONS.get(media.mime_type.lower().split(";")[0].strip())
        if mapped:
            return mapped
    return ".oga" if media.kind == "voice" else ".mp3"


class TelegramFileDownloader:
    """Downloads voice/audio payloads to a scratch directory."""

    def __init__(self, bot: Bot, settings: Settings) -> None:
        self._bot = bot
        self._settings = settings

    async def download(self, media: MediaRef, directory: Path) -> Path:
        limit = self._settings.telegram_max_download_bytes
        if media.file_size and media.file_size > limit:
            raise FileTooLargeError(
                f"Telegram file is {media.file_size / 1024 / 1024:.1f} MB; the Bot API "
                f"cannot download more than {limit / 1024 / 1024:.0f} MB."
            )

        file = await self._bot.get_file(media.file_id)
        if file.file_size and file.file_size > limit:
            raise FileTooLargeError(
                f"Telegram file is {file.file_size / 1024 / 1024:.1f} MB; the Bot API "
                f"cannot download more than {limit / 1024 / 1024:.0f} MB."
            )

        extension = guess_extension(media, file.file_path)
        destination = directory / f"source{extension}"
        directory.mkdir(parents=True, exist_ok=True)
        await self._bot.download_file(file.file_path, destination=destination)
        size = destination.stat().st_size
        logger.info(
            "Downloaded %s (%s, %.2f MB) to %s",
            media.kind,
            media.file_unique_id,
            size / 1024 / 1024,
            destination.name,
        )
        return destination
