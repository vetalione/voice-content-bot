import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from aiogram import Dispatcher
from aiogram.types import Update

from app.services.checkpoints import CheckpointError, NullStore, SQLiteStore
from app.services.llm import LLMError
from app.services.model_preferences import ModelPreferences, model_key
from app.services.processor import RecordingProcessor
from app.telegram.handlers import JobSubmitter, build_router
from app.telegram.settings_menu import menu
from tests.conftest import OWNER_ID, STRANGER_ID, private_message_update, voice_payload
from tests.test_provider_routing import info, response, router


@pytest.fixture
def preferences(settings, tmp_path):
    s = settings.model_copy(
        update={
            "text_provider": "openrouter",
            "semantic_pipeline_enabled": True,
            "openrouter_primary_model": "moonshotai/kimi-k2.5",
            "openrouter_escalation_model": "moonshotai/kimi-k3",
            "openrouter_allow_paid": True,
            "openrouter_allow_escalation": True,
            "quality_auditor": "auto",
            "writing_provider": "claude",
            "claude_enabled": True,
            "openrouter_api_key": "fake",
            "openrouter_max_retries": 0,
        }
    )
    models = [
        info("openrouter/free", "0", []),
        info("sample/text:free", "0"),
        info("sample/not-free:free", "0.01"),
        info("moonshotai/kimi-k2.5"),
        info("moonshotai/kimi-k3"),
    ]
    catalog = SimpleNamespace(models={m["id"]: m for m in models}, refresh=AsyncMock())
    return ModelPreferences(s, SQLiteStore(tmp_path / "settings.sqlite3"), catalog)


async def test_free_choice_survives_restart_and_disables_all_paid_paths(preferences):
    await preferences.select("free", "sample/text:free")
    restarted = ModelPreferences(
        preferences.settings, SQLiteStore(preferences.store.path), preferences.catalog
    )
    s = await restarted.snapshot()
    assert s.openrouter_primary_model == "sample/text:free"
    assert s.semantic_pipeline_enabled
    assert not s.openrouter_allow_paid
    assert not s.openrouter_primary_fallback_model
    assert not s.openrouter_escalation_model
    assert not s.openrouter_allow_escalation
    assert not s.claude_enabled
    assert s.quality_auditor == "none"
    assert s.writing_provider == "primary"
    assert s.semantic_reasoning_effort == s.writing_reasoning_effort == "low"
    assert preferences.settings.openrouter_allow_paid  # no shared settings mutation


async def test_switch_back_to_kimi_restores_explicit_server_policy(preferences):
    await preferences.select("free", "openrouter/free")
    await preferences.select("paid", "moonshotai/kimi-k2.5")
    s = await preferences.snapshot()
    assert s.openrouter_allow_paid
    assert s.openrouter_primary_model == "moonshotai/kimi-k2.5"
    assert s.quality_auditor == "auto"


async def test_free_list_checks_prices_and_stale_choices(preferences):
    assert await preferences.choices() == ["sample/text:free"]
    for model in ("sample/not-free:free", "missing/model:free", "moonshotai/kimi-k2.5"):
        with pytest.raises(LLMError):
            await preferences.select("free", model)
    assert await preferences.current() == {"mode": "configured"}


async def test_server_paid_gate_cannot_be_overridden_from_telegram(preferences):
    await preferences.select("paid", "moonshotai/kimi-k2.5")
    preferences.settings = preferences.settings.model_copy(update={"openrouter_allow_paid": False})
    with pytest.raises(LLMError):
        await preferences.snapshot()
    with pytest.raises(LLMError):
        await preferences.select("paid", "moonshotai/kimi-k2.5")
    # Owner can still recover through the free button.
    await preferences.select("free", "openrouter/free")
    assert not (await preferences.snapshot()).openrouter_allow_paid


async def test_no_false_persistence_confirmation(preferences):
    preferences.store = NullStore()
    with pytest.raises(CheckpointError):
        await preferences.select("free", "openrouter/free")


def callback(user, data, chat_type="private"):
    return Update.model_validate(
        {
            "update_id": 200,
            "callback_query": {
                "id": "button",
                "from": {"id": user, "is_bot": False, "first_name": "User"},
                "chat_instance": "x",
                "data": data,
                "message": {
                    "message_id": 3,
                    "date": int(time.time()),
                    "chat": {"id": OWNER_ID, "type": chat_type},
                },
            },
        }
    )


@pytest.mark.parametrize("user,chat_type", [(STRANGER_ID, "private"), (OWNER_ID, "group")])
async def test_callback_owner_and_private_guards(preferences, wiring, offline_bot, user, chat_type):
    dp = Dispatcher()
    dp.include_router(
        build_router(preferences.settings, wiring.submitter, wiring.runner, preferences)
    )
    await dp.feed_update(offline_bot.bot, callback(user, "model:auto", chat_type))
    assert await preferences.current() == {"mode": "configured"}
    assert offline_bot.sent[-1].show_alert


async def test_owner_can_open_menu_and_select_both_modes(preferences, wiring, offline_bot):
    dp = Dispatcher()
    dp.include_router(
        build_router(preferences.settings, wiring.submitter, wiring.runner, preferences)
    )
    update = private_message_update(user_id=OWNER_ID, text="/settings")
    await dp.feed_update(offline_bot.bot, update)
    assert offline_bot.sent[-1].reply_markup
    await dp.feed_update(offline_bot.bot, callback(OWNER_ID, "model:auto"))
    assert (await preferences.current())["mode"] == "free"
    await dp.feed_update(
        offline_bot.bot, callback(OWNER_ID, f"model:paid:{model_key('moonshotai/kimi-k2.5')}")
    )
    assert (await preferences.current())["mode"] == "paid"
    await dp.feed_update(offline_bot.bot, private_message_update(user_id=OWNER_ID, text="/status"))
    assert "moonshotai/kimi-k2.5" in offline_bot.sent[-1].text


async def test_menu_paginates_long_model_ids_with_short_callbacks(preferences):
    for n in range(20):
        name = "vendor/" + "long-model-name" * 6 + str(n) + ":free"
        preferences.catalog.models[name] = info(name, "0")
    _, keyboard = await menu(preferences, 0)
    buttons = [b for row in keyboard.inline_keyboard for b in row]
    assert len([b for b in buttons if b.callback_data.startswith("model:free:")]) == 6
    assert all(len(b.callback_data.encode()) <= 64 for b in buttons)
    assert any(b.callback_data == "model:page:1" for b in buttons)


async def test_queued_and_running_jobs_keep_their_snapshot(preferences, wiring, offline_bot):
    seen = []
    entered, finish = asyncio.Event(), asyncio.Event()

    async def process(request, selected):
        seen.append(selected)
        if len(seen) == 1:
            entered.set()
            await finish.wait()

    submitter = JobSubmitter(
        preferences.settings,
        wiring.runner,
        wiring.processor,
        wiring.dedupe,
        wiring.delivery,
        preferences,
        process,
    )
    dp = Dispatcher()
    dp.include_router(build_router(preferences.settings, submitter, wiring.runner, preferences))
    await preferences.select("free", "openrouter/free")
    await dp.feed_update(
        offline_bot.bot,
        private_message_update(user_id=OWNER_ID, media=voice_payload(), message_id=101),
    )
    await preferences.select("paid", "moonshotai/kimi-k2.5")
    await dp.feed_update(
        offline_bot.bot,
        private_message_update(user_id=OWNER_ID, media=voice_payload(), message_id=102),
    )
    await wiring.runner.start()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await preferences.select("free", "sample/text:free")
        finish.set()
        assert await wiring.runner.drain(timeout=5)
    finally:
        finish.set()
        await wiring.runner.stop()
    assert [s.openrouter_primary_model for s in seen] == ["openrouter/free", "moonshotai/kimi-k2.5"]
    assert not seen[0].openrouter_allow_paid
    assert seen[1].openrouter_allow_paid


@pytest.mark.parametrize("status", [200, 429, 404])
async def test_free_profile_http_is_zero_price_and_never_paid_fallback(preferences, status):
    await preferences.select("free", "openrouter/free")
    s = await preferences.snapshot()
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        assert body["model"] == "openrouter/free"
        assert body["provider"]["max_price"] == {
            "prompt": 0,
            "completion": 0,
            "request": 0,
            "image": 0,
        }
        assert body["reasoning"]["effort"] == "low"
        return (
            response()
            if status == 200
            else httpx.Response(status, json={"error": {"message": "Unavailable"}})
        )

    client = router(s, list(preferences.catalog.models.values()), handler)
    try:
        if status == 200:
            await client.chat_json(system="s", user="u", label="channel_teaser#1")
        else:
            with pytest.raises(LLMError):
                await client.chat_json(system="s", user="u", label="channel_teaser#1")
        assert len(calls) == 1
    finally:
        await client.aclose()


async def test_mandatory_reasoning_never_disabled_even_without_effort_list(preferences):
    s = preferences.settings.model_copy(
        update={"writing_reasoning_effort": "none", "writing_provider": "primary"}
    )
    model = info(s.openrouter_primary_model)
    model["reasoning"] = {"mandatory": True}

    def handler(req):
        assert json.loads(req.content)["reasoning"] == {"effort": "low"}
        return response()

    client = router(s, [model], handler)
    try:
        await client.chat_json(system="s", user="u", label="channel_teaser#1")
    finally:
        await client.aclose()


async def test_free_profile_ignores_exhausted_paid_monthly_budget(preferences):
    await preferences.select("free", "openrouter/free")
    s = (await preferences.snapshot()).model_copy(update={"stop_on_budget_exceeded": True})
    text = SimpleNamespace(monthly_usage=AsyncMock(return_value=100))
    pipeline = SimpleNamespace(_miner=SimpleNamespace(_llm=text))
    processor = RecordingProcessor(s, pipeline, None, None)
    await processor._check_budget(enforce=True)
    text.monthly_usage.assert_not_awaited()
