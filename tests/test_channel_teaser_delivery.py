from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

from app.models.atoms import ContentAtom, ContentAtomSet
from app.models.content import ReelsBatch, ThreadsBatch
from app.services.dedupe import TTLDedupeStore
from app.services.jobs import JobRunner
from app.services.processor import RecordingProcessor
from app.telegram.delivery import TelegramDelivery
from app.telegram.handlers import JobSubmitter, build_router
from tests.conftest import (
    CHANNEL_ID,
    OWNER_ID,
    channel_post_update,
    private_message_update,
    voice_payload,
)
from tests.test_pipeline import build_pipeline


@pytest.mark.parametrize("mode,atoms_count", [("channel", 1), ("channel", 3), ("private", 3)])
async def test_voice_event_sends_only_teaser_and_source_timestamps_as_reply(
    settings, offline_bot, monkeypatch, mode, atoms_count
):
    s = settings.model_copy(
        update={
            "text_provider": "openrouter",
            "dry_run_publish": False,
            "teaser_include_timestamps": True,
            "teaser_reply_to_source": True,
        }
    )
    sent = []

    async def fake_call(self, method, request_timeout=None):
        sent.append(method)
        return SimpleNamespace(message_id=800)

    monkeypatch.setattr(Bot, "__call__", fake_call)
    delivery = TelegramDelivery(offline_bot.bot, s)
    llm = SimpleNamespace(chat_text=AsyncMock(return_value="Тизер о важных идеях записи."))
    pipeline, _, _ = build_pipeline(s, llm=llm, delivery=delivery)
    dedupe = TTLDedupeStore()
    processor = RecordingProcessor(s, pipeline, delivery, dedupe)
    pipeline._miner = SimpleNamespace(
        mine=AsyncMock(
            return_value=ContentAtomSet(
                atoms=[
                    ContentAtom(
                        id=f"a{i}",
                        label=f"Тема {i + 1}",
                        key_claim="Мысль",
                        start_seconds=i * 30,
                        end_seconds=(i + 1) * 30,
                        confidence=0.9,
                    )
                    for i in range(atoms_count)
                ]
            )
        )
    )
    pipeline._threads_agent = SimpleNamespace(edit=AsyncMock(return_value=ThreadsBatch()))
    pipeline._reels_agent = SimpleNamespace(edit=AsyncMock(return_value=ReelsBatch()))
    runner = JobRunner(concurrency=1, queue_size=2)
    submitter = JobSubmitter(s, runner, processor, dedupe, delivery)
    dp = Dispatcher()
    dp.include_router(build_router(s, submitter, runner))
    update = (
        channel_post_update(message_id=501, media=voice_payload())
        if mode == "channel"
        else private_message_update(user_id=OWNER_ID, message_id=501, media=voice_payload())
    )
    await dp.feed_update(offline_bot.bot, update)
    await runner.start()
    try:
        assert await runner.drain(timeout=5)
    finally:
        await runner.stop()
    assert runner.stats.failed == 0
    public = [m for m in sent if isinstance(m, SendMessage) and m.chat_id == CHANNEL_ID]
    if mode == "private":
        assert public == []
    else:
        assert len(public) == 1
        message = public[0]
        assert message.reply_parameters.message_id == 501
        assert message.reply_parameters.allow_sending_without_reply is False
        assert "Тизер о важных идеях записи." in message.text
        assert "00:00 — Тема 1" in message.text
        if atoms_count == 3:
            assert "00:30 — Тема 2" in message.text and "01:00 — Тема 3" in message.text
        assert "Threads" not in message.text and "Reels" not in message.text


async def test_missing_source_never_falls_back_to_standalone_channel_post(settings):
    method = SendMessage(chat_id=CHANNEL_ID, text="Тизер")
    bot = SimpleNamespace(
        send_message=AsyncMock(
            side_effect=TelegramBadRequest(method=method, message="message to be replied not found")
        )
    )
    with pytest.raises(TelegramBadRequest):
        await TelegramDelivery(bot, settings).publish_to_channel("Тизер", reply_to_message_id=501)
    assert bot.send_message.await_count == 1
    assert (
        bot.send_message.call_args.kwargs["reply_parameters"].allow_sending_without_reply is False
    )
