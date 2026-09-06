"""Private mode: allow-list enforcement and the no-publishing guarantee."""

from __future__ import annotations

from app.models.media import SourceMode
from app.telegram.filters import AllowedPrivateUserFilter
from tests.conftest import (
    ALLOWED_USER_ID,
    OWNER_ID,
    STRANGER_ID,
    audio_payload,
    private_message_update,
    voice_payload,
)


async def test_owner_is_always_allowed(settings):
    assert OWNER_ID in settings.allowed_user_ids
    assert ALLOWED_USER_ID in settings.allowed_user_ids
    assert STRANGER_ID not in settings.allowed_user_ids


async def test_private_filter_accepts_allowed_user(settings, bot_free_private_message):
    filter_ = AllowedPrivateUserFilter(settings.allowed_user_ids)
    assert await filter_(bot_free_private_message(ALLOWED_USER_ID)) is True
    assert await filter_(bot_free_private_message(OWNER_ID)) is True


async def test_private_filter_rejects_stranger(settings, bot_free_private_message):
    filter_ = AllowedPrivateUserFilter(settings.allowed_user_ids)
    assert await filter_(bot_free_private_message(STRANGER_ID)) is False


async def test_private_filter_rejects_group_chats(settings, bot_free_private_message):
    filter_ = AllowedPrivateUserFilter(settings.allowed_user_ids)
    message = bot_free_private_message(ALLOWED_USER_ID, chat_type="group")
    assert await filter_(message) is False


async def test_allowed_user_voice_is_queued_without_publishing(wiring, offline_bot):
    update = private_message_update(user_id=ALLOWED_USER_ID, media=voice_payload())
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert len(wiring.processor.processed) == 1
    request = wiring.processor.processed[0]
    assert request.mode is SourceMode.PRIVATE
    assert request.may_publish is False
    assert wiring.delivery.channel_posts == []


async def test_forwarded_audio_from_archive_is_processed_privately(wiring, offline_bot):
    update = private_message_update(
        user_id=OWNER_ID,
        media=audio_payload(),
        media_key="audio",
        forwarded=True,
    )
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert len(wiring.processor.processed) == 1
    request = wiring.processor.processed[0]
    assert request.forwarded is True
    assert request.mode is SourceMode.PRIVATE
    assert request.may_publish is False
    assert wiring.delivery.channel_posts == []


async def test_forwarded_voice_is_marked_as_forward(wiring, offline_bot):
    update = private_message_update(user_id=OWNER_ID, media=voice_payload(), forwarded=True)
    await wiring.dispatcher.feed_update(offline_bot.bot, update)
    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert wiring.processor.processed[0].forwarded is True


async def test_unauthorized_private_user_triggers_nothing(wiring, offline_bot):
    """No job, no reply, no download, no Groq call, no channel post."""
    update = private_message_update(user_id=STRANGER_ID, media=voice_payload())
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    assert wiring.processor.processed == []
    assert wiring.delivery.channel_posts == []
    assert wiring.delivery.owner_messages == []
    assert wiring.delivery.plain_messages == []
    assert offline_bot.sent == [], "unauthorized users must get no reply at all"
    assert wiring.runner.stats.queued == 0


async def test_unauthorized_private_text_is_silently_dropped(wiring, offline_bot):
    update = private_message_update(user_id=STRANGER_ID, text="привет, что ты умеешь?")
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    assert wiring.processor.processed == []
    assert offline_bot.sent == []


async def test_authorized_private_unsupported_message_gets_a_hint(wiring, offline_bot):
    update = private_message_update(user_id=OWNER_ID, text="а видео можно?")
    await wiring.dispatcher.feed_update(offline_bot.bot, update)

    assert wiring.processor.processed == []
    assert len(offline_bot.sent) == 1, "authorized users get a reply, but no job"
