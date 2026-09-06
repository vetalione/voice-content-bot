"""Webhook endpoint: fast response, secret token, duplicate updates."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import (
    ALLOWED_USER_ID,
    CHANNEL_ID,
    channel_post_update,
    private_message_update,
    voice_payload,
)


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    """A TestClient whose pipeline is replaced by an instant no-op job."""
    from aiogram import Bot

    processed: list[str] = []

    async def fake_call(self, method, request_timeout=None):
        return None

    monkeypatch.setattr(Bot, "__call__", fake_call, raising=True)

    async def fake_process(self, request):
        processed.append(request.dedupe_key)
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "app.services.processor.RecordingProcessor.process", fake_process, raising=True
    )

    settings = settings.model_copy(update={"webhook_secret": "s3cret"})
    app = create_app(settings)
    with TestClient(app) as test_client:
        test_client.processed = processed  # type: ignore[attr-defined]
        test_client.settings = settings  # type: ignore[attr-defined]
        yield test_client


def post_update(client, update, secret: str | None = "s3cret"):
    headers = {"X-Telegram-Bot-Api-Secret-Token": secret} if secret else {}
    return client.post(
        client.settings.webhook_path,
        json=update.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def test_health_reports_configuration_and_queue(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["configured"] is True
    assert body["missing_env"] == []
    assert "queue" in body


def test_root_points_at_health(client):
    assert client.get("/").json()["health"] == "/health"


def test_misconfigured_service_starts_degraded_instead_of_crash_looping(
    settings: Settings,
):
    """A bad BOT_TOKEN must still leave /health answering, with the reason."""
    broken = settings.model_copy(update={"bot_token": "not-a-valid-token"})
    with TestClient(create_app(broken)) as test_client:
        body = test_client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["startup_error"]

        response = test_client.post(broken.webhook_path, json={"update_id": 1, "message": {}})
        assert response.status_code == 503


def test_webhook_returns_immediately_for_a_channel_voice_post(client):
    update = channel_post_update(
        update_id=1, chat_id=CHANNEL_ID, message_id=11, media=voice_payload()
    )
    response = post_update(client, update)
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_webhook_rejects_a_bad_secret_token(client):
    update = channel_post_update(update_id=2, media=voice_payload())
    response = post_update(client, update, secret="wrong")
    assert response.status_code == 403


def test_webhook_rejects_a_missing_secret_token(client):
    update = channel_post_update(update_id=3, media=voice_payload())
    response = post_update(client, update, secret=None)
    assert response.status_code == 403


def test_duplicate_update_id_is_acknowledged_but_not_reprocessed(client):
    update = channel_post_update(
        update_id=99, chat_id=CHANNEL_ID, message_id=55, media=voice_payload()
    )
    first = post_update(client, update)
    second = post_update(client, update)

    assert first.json() == {"ok": True}
    assert second.json()["duplicate"] is True


def test_unparseable_update_is_acknowledged_so_telegram_stops_retrying(client):
    response = client.post(
        client.settings.webhook_path,
        json={"update_id": "not-an-int", "channel_post": {"broken": True}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_malformed_body_is_rejected(client):
    response = client.post(
        client.settings.webhook_path,
        content=b"not json at all",
        headers={
            "Content-Type": "application/json",
            "X-Telegram-Bot-Api-Secret-Token": "s3cret",
        },
    )
    assert response.status_code == 400


def test_private_voice_from_allowed_user_is_accepted(client):
    update = private_message_update(update_id=7, user_id=ALLOWED_USER_ID, media=voice_payload())
    assert post_update(client, update).status_code == 200


def test_foreign_channel_post_is_acknowledged_and_ignored(client):
    update = channel_post_update(update_id=8, chat_id=-1009999999999, media=voice_payload())
    assert post_update(client, update).json() == {"ok": True}
