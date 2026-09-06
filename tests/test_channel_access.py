"""Channel mode: only the allowed channel is processed, only voice/audio."""

from __future__ import annotations

import pytest

from app.models.media import MediaKind, SourceMode
from app.telegram.filters import AllowedChannelFilter, extract_media
from tests.conftest import (
    CHANNEL_ID,
    audio_payload,
    channel_post_update,
    voice_payload,
)


async def test_allowed_channel_filter_accepts_configured_channel(bot_free_message):
    filter_ = AllowedChannelFilter(CHANNEL_ID)
    assert await filter_(bot_free_message(CHANNEL_ID)) is True


async def test_allowed_channel_filter_rejects_other_channels(bot_free_message):
    filter_ = AllowedChannelFilter(CHANNEL_ID)
    assert await filter_(bot_free_message(-1009999999999)) is False


async def test_voice_post_in_allowed_channel_is_queued(wiring, offline_bot):
    update = channel_post_update(chat_id=CHANNEL_ID, media=voice_payload())
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert len(wiring.processor.processed) == 1
    request = wiring.processor.processed[0]
    assert request.mode is SourceMode.CHANNEL
    assert request.chat_id == CHANNEL_ID
    assert request.media.kind is MediaKind.VOICE
    assert request.may_publish is True


async def test_audio_post_in_allowed_channel_is_queued(wiring, offline_bot):
    update = channel_post_update(chat_id=CHANNEL_ID, media=audio_payload(), media_key="audio")
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert len(wiring.processor.processed) == 1
    assert wiring.processor.processed[0].media.kind is MediaKind.AUDIO


async def test_voice_post_in_foreign_channel_is_ignored(wiring, offline_bot):
    update = channel_post_update(chat_id=-1005555555555, media=voice_payload())
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    assert wiring.processor.processed == []
    assert wiring.delivery.channel_posts == []
    assert offline_bot.sent == []


@pytest.mark.parametrize(
    ("media_key", "payload"),
    [
        ("text", None),
        ("photo", None),
        ("document", None),
        ("video_note", None),
    ],
)
async def test_unsupported_channel_post_is_ignored(wiring, offline_bot, media_key, payload):
    extra: dict[str, object]
    if media_key == "text":
        extra = {"text": "Обычный текстовый пост"}
    elif media_key == "photo":
        extra = {
            "photo": [
                {
                    "file_id": "p",
                    "file_unique_id": "pu",
                    "width": 10,
                    "height": 10,
                    "file_size": 100,
                }
            ]
        }
    elif media_key == "document":
        extra = {
            "document": {
                "file_id": "d",
                "file_unique_id": "du",
                "file_name": "notes.pdf",
                "mime_type": "application/pdf",
            }
        }
    else:
        extra = {
            "video_note": {
                "file_id": "v",
                "file_unique_id": "vu",
                "length": 240,
                "duration": 30,
            }
        }

    update = channel_post_update(chat_id=CHANNEL_ID, extra=extra)
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    assert wiring.processor.processed == []
    assert wiring.delivery.channel_posts == []
    assert extract_media(update.channel_post) is None


async def test_edited_channel_post_does_not_reprocess(wiring, offline_bot):
    post = channel_post_update(chat_id=CHANNEL_ID, media=voice_payload()).channel_post
    assert post is not None
    edited = channel_post_update(chat_id=CHANNEL_ID, media=voice_payload())
    payload = edited.model_dump(mode="json", exclude_none=True)
    update = type(edited).model_validate(
        {"update_id": 42, "edited_channel_post": payload["channel_post"]}
    )
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    assert wiring.processor.processed == []
