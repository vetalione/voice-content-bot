"""Configuration parsing and readiness reporting."""

from __future__ import annotations

from app.config import Settings
from tests.conftest import FAKE_TOKEN


def make(**overrides) -> Settings:
    base = {
        "text_provider": "groq",
        "bot_token": FAKE_TOKEN,
        "groq_api_key": "gsk_real_key",
        "owner_telegram_id": 1,
        "allowed_channel_id": -100123,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_fully_configured_reports_nothing_missing():
    assert make().missing_required() == []


def test_empty_values_are_reported():
    settings = make(bot_token="", groq_api_key="", owner_telegram_id=0, allowed_channel_id=0)
    assert set(settings.missing_required()) == {
        "BOT_TOKEN",
        "GROQ_API_KEY",
        "OWNER_TELEGRAM_ID",
        "ALLOWED_CHANNEL_ID",
    }


def test_unedited_env_example_placeholders_count_as_missing():
    """Copying .env.example without editing must not look healthy."""
    settings = make(bot_token="123456789:AA...replace-me", groq_api_key="gsk_...replace-me")
    assert set(settings.missing_required()) == {"BOT_TOKEN", "GROQ_API_KEY"}


def test_allowed_user_ids_parses_separators_and_includes_the_owner():
    settings = make(owner_telegram_id=111, allowed_user_ids="222, 333 444")
    assert settings.allowed_user_ids == frozenset({111, 222, 333, 444})


def test_allowed_user_ids_defaults_to_just_the_owner():
    assert make(owner_telegram_id=777, allowed_user_ids="").allowed_user_ids == frozenset({777})


def test_webhook_path_always_has_a_leading_slash():
    assert make(webhook_path="hook").webhook_path == "/hook"


def test_webhook_url_is_built_from_the_public_base_url():
    settings = make(public_base_url="https://x.onrender.com/", webhook_path="/tg")
    assert settings.webhook_url == "https://x.onrender.com/tg"
    assert make(public_base_url="").webhook_url == ""


def test_derived_byte_and_second_limits():
    settings = make(max_stt_upload_mb=24.0, audio_chunk_minutes=10.0)
    assert settings.max_stt_upload_bytes == 24 * 1024 * 1024
    assert settings.audio_chunk_seconds == 600.0
