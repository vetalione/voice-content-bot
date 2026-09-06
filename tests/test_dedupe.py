"""Duplicate update protection within the lifetime of the process."""

from __future__ import annotations

from app.services.dedupe import TTLDedupeStore
from tests.conftest import CHANNEL_ID, channel_post_update, voice_payload


def test_claim_is_idempotent():
    store = TTLDedupeStore(ttl_seconds=60, max_entries=10)
    assert store.claim("channel:-100:5") is True
    assert store.claim("channel:-100:5") is False
    assert store.seen("channel:-100:5") is True


def test_release_allows_reprocessing():
    store = TTLDedupeStore(ttl_seconds=60, max_entries=10)
    store.claim("k")
    store.release("k")
    assert store.claim("k") is True


def test_entries_expire_after_ttl():
    now = {"value": 1000.0}
    store = TTLDedupeStore(ttl_seconds=10, max_entries=10, clock=lambda: now["value"])
    assert store.claim("k") is True
    now["value"] += 5
    assert store.claim("k") is False
    now["value"] += 20
    assert store.claim("k") is True


def test_store_is_bounded():
    store = TTLDedupeStore(ttl_seconds=600, max_entries=5)
    for index in range(20):
        store.claim(f"k{index}")
    assert len(store) <= 5
    # The oldest entries were evicted, the newest survive.
    assert store.seen("k19") is True


async def test_telegram_retry_does_not_process_twice(wiring, offline_bot):
    """Telegram redelivering the same channel post must not run the pipeline twice."""
    first = channel_post_update(
        update_id=100, chat_id=CHANNEL_ID, message_id=777, media=voice_payload()
    )
    retry = channel_post_update(
        update_id=101, chat_id=CHANNEL_ID, message_id=777, media=voice_payload()
    )

    await wiring.dispatcher.feed_update(offline_bot.bot, first)
    await wiring.dispatcher.feed_update(offline_bot.bot, retry)

    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert len(wiring.processor.processed) == 1


async def test_same_message_after_failure_can_be_retried(wiring, offline_bot):
    """notify_failure releases the key so the owner can re-forward the recording."""
    update = channel_post_update(
        update_id=200, chat_id=CHANNEL_ID, message_id=888, media=voice_payload()
    )
    await wiring.dispatcher.feed_update(offline_bot.bot, update)
    key = f"channel:{CHANNEL_ID}:888"
    assert wiring.dedupe.seen(key) is True

    wiring.dedupe.release(key)
    await wiring.dispatcher.feed_update(
        offline_bot.bot,
        channel_post_update(
            update_id=201, chat_id=CHANNEL_ID, message_id=888, media=voice_payload()
        ),
    )

    await wiring.runner.start()
    assert await wiring.runner.drain(timeout=5)
    await wiring.runner.stop()

    assert len(wiring.processor.processed) == 2
