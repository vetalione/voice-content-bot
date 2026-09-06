"""Groq failure handling: 429s, permanent errors, and owner notification.

The rule under test: a Groq problem must surface loudly and must never cause a
silent switch to a paid provider.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.models.media import SourceMode
from app.services.dedupe import TTLDedupeStore
from app.services.groq_client import GroqClient, GroqError, GroqQuotaError
from app.services.jobs import JobRunner
from app.services.processor import RecordingProcessor
from app.utils.retry import RetryableError, retry_async
from tests.conftest import FakeDelivery, FakeTranscriber, make_job_request
from tests.test_pipeline import build_pipeline


def make_client(settings: Settings, handler) -> GroqClient:
    """A GroqClient backed by httpx.MockTransport — no sockets involved."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport,
        base_url=settings.groq_base_url,
        headers={"Authorization": "Bearer test"},
    )
    return GroqClient(settings, client=http)


# ------------------------------------------------------------------- retries ---
async def test_retry_async_stops_after_the_bound():
    attempts = {"count": 0}

    async def always_fails():
        attempts["count"] += 1
        raise RetryableError("429")

    with pytest.raises(RetryableError):
        await retry_async(always_fails, attempts=3, base_delay=0, label="t", sleep=_no_sleep)
    assert attempts["count"] == 3


async def test_retry_async_returns_on_recovery():
    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RetryableError("429", retry_after=0)
        return "ok"

    assert await retry_async(flaky, attempts=5, base_delay=0, label="t", sleep=_no_sleep) == "ok"
    assert attempts["count"] == 3


async def _no_sleep(_delay: float) -> None:
    return None


# ---------------------------------------------------------------- groq client ---
@pytest.mark.parametrize("suffix", [".oga", ".OGA", ".ogg", ".mp3"])
async def test_transcription_upload_filename(settings, tmp_path, suffix):
    path = tmp_path / f"source{suffix}"
    audio = b"OggS-test-audio-payload"
    path.write_bytes(audio)
    expected_name = "source.ogg" if suffix.lower() == ".oga" else path.name

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        assert request.url.path.endswith("/audio/transcriptions")
        assert f'filename="{expected_name}"'.encode() in body
        assert audio in body
        return httpx.Response(200, json={"text": "transcribed"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.groq_base_url
    ) as http:
        client = GroqClient(settings, client=http)
        assert await client.transcribe_file(path, model=settings.groq_whisper_model) == {
            "text": "transcribed"
        }
    assert path.read_bytes() == audio


async def test_rate_limit_is_retried_then_raised_as_quota_error(settings, monkeypatch):
    monkeypatch.setattr("app.utils.retry.asyncio.sleep", _no_sleep)
    settings = settings.model_copy(update={"groq_max_retries": 2, "groq_retry_base_delay": 0.0})
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"message": "Rate limit reached"}},
        )

    client = make_client(settings, handler)
    with pytest.raises(GroqQuotaError) as excinfo:
        await client.chat_json(system="s", user="u", label="miner")
    await client.aclose()

    assert calls["count"] == 3, "initial attempt plus GROQ_MAX_RETRIES"
    assert "429" in str(excinfo.value)


async def test_server_error_is_retryable(settings, monkeypatch):
    monkeypatch.setattr("app.utils.retry.asyncio.sleep", _no_sleep)
    settings = settings.model_copy(update={"groq_max_retries": 1, "groq_retry_base_delay": 0.0})
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="upstream unavailable")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"teaser":"ок","timestamps":[]}'}}]},
        )

    client = make_client(settings, handler)
    payload = await client.chat_json(system="s", user="u", label="teaser")
    await client.aclose()

    assert payload["teaser"] == "ок"
    assert calls["count"] == 2


async def test_invalid_api_key_is_not_retried(settings):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    client = make_client(settings, handler)
    with pytest.raises(GroqError) as excinfo:
        await client.chat_json(system="s", user="u", label="miner")
    await client.aclose()

    assert calls["count"] == 1, "4xx other than 429 is permanent"
    assert excinfo.value.status_code == 401
    assert not isinstance(excinfo.value, GroqQuotaError)


async def test_json_schema_rejection_falls_back_to_json_object(settings):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as jsonlib

        body = jsonlib.loads(request.content)
        seen.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "json_schema is not supported by this model"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    client = make_client(settings, handler)
    payload = await client.chat_json(system="s", user="u", schema={"type": "object"}, label="miner")
    await client.aclose()

    assert seen == ["json_schema", "json_object"]
    assert payload == {"ok": True}


async def test_json_wrapped_in_prose_is_recovered(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": 'Вот результат:\n```json\n{"teaser": "ок"}\n```'}}
                ]
            },
        )

    client = make_client(settings, handler)
    assert (await client.chat_json(system="s", user="u"))["teaser"] == "ок"
    await client.aclose()


async def test_non_json_response_raises(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "просто текст"}}]})

    client = make_client(settings, handler)
    with pytest.raises(GroqError):
        await client.chat_json(system="s", user="u")
    await client.aclose()


async def test_network_failure_is_wrapped(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    client = make_client(settings, handler)
    with pytest.raises(GroqError):
        await client.chat_json(system="s", user="u")
    await client.aclose()


# ------------------------------------------------------- failure notification ---
async def test_transcription_failure_notifies_the_owner_and_publishes_nothing(settings):
    transcriber = FakeTranscriber(error=GroqQuotaError("429 rate limit"))
    delivery = FakeDelivery()
    pipeline, _, _ = build_pipeline(settings, transcriber=transcriber, delivery=delivery)
    dedupe = TTLDedupeStore()
    processor = RecordingProcessor(settings, pipeline, delivery, dedupe)  # type: ignore[arg-type]
    request = make_job_request(SourceMode.CHANNEL)
    dedupe.claim(request.dedupe_key)

    with pytest.raises(GroqQuotaError):
        await processor.process(request)
    await processor.notify_failure(request, GroqQuotaError("429 rate limit"))

    assert delivery.channel_posts == []
    assert any("не удалась" in message for message in delivery.owner_messages)
    assert dedupe.seen(request.dedupe_key) is False, "failed job can be retried"


async def test_job_runner_routes_errors_to_the_error_handler():
    runner = JobRunner(concurrency=1, queue_size=4)
    await runner.start()
    seen: list[BaseException] = []

    async def failing() -> None:
        raise GroqQuotaError("boom")

    async def on_error(error: BaseException) -> None:
        seen.append(error)

    runner.submit("failing", failing, on_error)
    assert await runner.drain(timeout=5)
    await runner.stop()

    assert len(seen) == 1
    assert isinstance(seen[0], GroqQuotaError)
    assert runner.stats.failed == 1


async def test_job_runner_survives_a_failed_job_and_keeps_working():
    runner = JobRunner(concurrency=1, queue_size=4)
    await runner.start()
    done: list[str] = []

    async def failing() -> None:
        raise RuntimeError("nope")

    async def ok() -> None:
        done.append("ok")

    runner.submit("failing", failing)
    runner.submit("ok", ok)
    assert await runner.drain(timeout=5)
    await runner.stop()

    assert done == ["ok"]
    assert runner.stats.completed == 1
    assert runner.stats.failed == 1
