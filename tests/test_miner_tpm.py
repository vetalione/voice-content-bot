"""Two-stage mining and scheduling failed attempts without redoing STT."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.agents.base import json_schema_for
from app.agents.content_miner import ContentMinerAgent
from app.models.atoms import AtomExtraction
from app.services.groq_client import GroqClient, GroqGenerationError
from app.services.prompts import PromptLibrary
from app.services.tpm import RollingTPM, duration_seconds
from tests.conftest import FakeLLM, FakeTranscriber, make_job_request
from tests.test_pipeline import build_pipeline


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


async def reserve(scheduler, tokens):
    await scheduler.reserve(
        tokens, label="extract/window=1", attempt=1, input_tokens=tokens - 800, output=800
    )


async def test_reservations_include_failed_calls_and_expire():
    clock = Clock()
    scheduler = RollingTPM(8000, clock=clock.clock, sleep=clock.sleep)
    await reserve(scheduler, 5000)
    # No success/usage response: the reservation must still count.
    await reserve(scheduler, 4000)
    assert sum(clock.waits) == 60
    assert list(scheduler.events) == [(60, 4000)]


async def test_generation_cooldown_and_provider_headers():
    clock = Clock()
    scheduler = RollingTPM(8000, clock=clock.clock, sleep=clock.sleep)
    await reserve(scheduler, 1000)
    scheduler.observe(
        {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "1m10s"},
        generation_failed=True,
    )
    await reserve(scheduler, 1000)
    assert clock.now == 70
    assert sum(clock.waits) == 70
    assert max(clock.waits) <= 10


async def test_concurrent_admission_cannot_overbook():
    clock = Clock()
    scheduler = RollingTPM(8000, clock=clock.clock, sleep=clock.sleep)
    await asyncio.gather(reserve(scheduler, 5000), reserve(scheduler, 5000))
    assert clock.now == 60


async def test_actual_failed_http_attempts_wait_and_log_original(settings, caplog):
    clock = Clock()
    times = []

    def handler(request):
        times.append(clock.now)
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(
            400,
            headers={"x-ratelimit-remaining-tokens": "7000", "x-ratelimit-reset-tokens": "2s"},
            json={
                "error": {
                    "code": "json_validate_failed",
                    "message": "original provider detail",
                    "failed_generation": "partial " + settings.groq_api_key,
                },
                "finish_reason": "length",
            },
        )

    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        client = GroqClient(settings, client=http)
        client._tpm[settings.groq_llm_model] = RollingTPM(
            8000, clock=clock.clock, sleep=clock.sleep
        )
        with caplog.at_level("INFO"):
            for _ in range(2):
                with pytest.raises(GroqGenerationError):
                    await client.chat_json(
                        system="extract",
                        user="text",
                        schema=json_schema_for(AtomExtraction),
                        max_tokens=800,
                        label="extraction/window=1",
                    )
    assert times == [0, 60]
    assert "original provider detail" in caplog.text
    assert "failed_generation" in caplog.text and "partial [REDACTED]" in caplog.text
    assert settings.groq_api_key not in caplog.text
    for field in (
        "http_status",
        "error.code",
        "finish_reason",
        "x-ratelimit-remaining-tokens",
        "rolling_tpm_estimate",
        "wait_before_request",
    ):
        assert field in caplog.text


def test_minimal_schema_and_atom_cap():
    schema = json_schema_for(AtomExtraction)
    atom = schema["$defs"]["ExtractedAtom"]
    assert set(atom["properties"]) == {"title", "start_seconds", "end_seconds", "idea", "type"}
    assert set(atom["required"]) == set(atom["properties"])
    assert atom["additionalProperties"] is False
    seed = {"title": "t", "start_seconds": 0, "end_seconds": 1, "idea": "i", "type": "idea"}
    with pytest.raises(ValidationError):
        AtomExtraction.model_validate({"atoms": [seed] * 9})


async def test_no_per_atom_enrichment(settings, transcript):
    llm = FakeLLM()
    result = await ContentMinerAgent(llm, PromptLibrary(settings.prompts_dir), settings).mine(
        transcript
    )
    extraction = [c for c in llm.calls if c["label"].startswith("content_extraction")]
    enrichment = [c for c in llm.calls if c["label"].startswith("content_enrichment")]
    assert extraction and all(c["max_tokens"] == 1500 for c in extraction)
    assert len(enrichment) == 0
    assert len(result.atoms) == 2
    assert all("Source excerpt:" in c["user"] for c in enrichment)
    assert all("TRANSCRIPT WINDOW" not in c["user"] for c in enrichment)


async def test_generation_recovery_does_not_call_whisper_again(settings):
    class OnceBroken(FakeLLM):
        def __init__(self):
            super().__init__()
            self.failed = False

        async def chat_json(self, **kwargs):
            if kwargs["label"].startswith("content_extraction") and not self.failed:
                self.failed = True
                raise GroqGenerationError("invalid JSON")
            return await super().chat_json(**kwargs)

    transcriber = FakeTranscriber()
    pipeline, _, _ = build_pipeline(settings, llm=OnceBroken(), transcriber=transcriber)
    await pipeline.run(make_job_request())
    assert len(transcriber.calls) == 1


def test_header_duration_units():
    assert duration_seconds("1m2.5s") == 62.5
    assert duration_seconds("200ms") == 0.2


async def test_successful_usage_avoids_reported_unnecessary_wait():
    clock = Clock()
    scheduler = RollingTPM(8000, clock=clock.clock, sleep=clock.sleep)
    for reserved, actual in ((2318, 1502), (1821, 597), (1993, 633)):
        reservation = await scheduler.reserve(
            reserved, label="replay", attempt=1, input_tokens=reserved - 800, output=800
        )
        scheduler.settle(reservation, {"total_tokens": actual})
    await reserve(scheduler, 1875)
    assert clock.waits == []
    assert sum(tokens for _, tokens in scheduler.events) == 2732 + 1875


async def test_missing_usage_retains_full_reservation():
    clock = Clock()
    scheduler = RollingTPM(8000, clock=clock.clock, sleep=clock.sleep)
    reservation = await scheduler.reserve(
        5000, label="test", attempt=1, input_tokens=4200, output=800
    )
    scheduler.settle(reservation, {})
    await reserve(scheduler, 4000)
    assert clock.now == 60
