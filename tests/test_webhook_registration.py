import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.telegram.webhook import sync_webhook_updates
from scripts import set_webhook


async def test_registration_subscribes_to_buttons_without_dropping_pending(settings, monkeypatch):
    configured = settings.model_copy(
        update={
            "public_base_url": "https://bot.example.test",
            "webhook_secret": "test-webhook-secret",
        }
    )
    call = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(set_webhook, "call", call)
    monkeypatch.setattr(set_webhook, "get_settings", lambda: configured)
    monkeypatch.setattr(sys, "argv", ["set_webhook.py", "set"])
    await set_webhook.main()
    args, params = call.call_args_list[0]
    assert args[1] == "setWebhook"
    assert set(params["allowed_updates"]) == {
        "message",
        "channel_post",
        "edited_channel_post",
        "callback_query",
    }
    assert params["drop_pending_updates"] is False
    assert params["url"] == "https://bot.example.test/telegram/webhook"
    assert params["secret_token"] == "test-webhook-secret"


def bot_info(**overrides):
    return SimpleNamespace(
        **{
            "url": "https://bot.example.test/telegram/webhook",
            "has_custom_certificate": False,
            "allowed_updates": ["message", "channel_post"],
            "max_connections": 7,
            **overrides,
        }
    )


async def test_startup_repairs_old_subscription_using_deployed_secret(settings):
    s = settings.model_copy(
        update={"public_base_url": "https://bot.example.test", "webhook_secret": "render-secret"}
    )
    bot = SimpleNamespace(
        get_webhook_info=AsyncMock(return_value=bot_info()), set_webhook=AsyncMock()
    )
    await sync_webhook_updates(bot, s)
    params = bot.set_webhook.call_args.kwargs
    assert "callback_query" in params["allowed_updates"]
    assert params["secret_token"] == "render-secret"
    assert params["max_connections"] == 7
    assert params["url"] == "https://bot.example.test/telegram/webhook"
    assert params["drop_pending_updates"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": "https://different.example.test/telegram/webhook"},
        {"has_custom_certificate": True},
        {"allowed_updates": ["message", "callback_query"]},
        {"allowed_updates": []},
    ],
)
async def test_startup_preserves_other_destinations_and_current_subscriptions(settings, overrides):
    s = settings.model_copy(update={"public_base_url": "https://bot.example.test"})
    bot = SimpleNamespace(
        get_webhook_info=AsyncMock(return_value=bot_info(**overrides)), set_webhook=AsyncMock()
    )
    await sync_webhook_updates(bot, s)
    bot.set_webhook.assert_not_awaited()


async def test_subscription_api_failure_does_not_break_startup_or_log_secrets(settings, caplog):
    s = settings.model_copy(update={"public_base_url": "https://bot.example.test"})
    bot = SimpleNamespace(get_webhook_info=AsyncMock(side_effect=RuntimeError("private-token")))
    await sync_webhook_updates(bot, s)
    assert "Webhook subscription sync failed" in caplog.text
    assert "private-token" not in caplog.text
