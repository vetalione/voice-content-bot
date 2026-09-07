import json

import httpx
import pytest

from app.models.media import SourceMode
from app.services.checkpoints import (
    CheckpointError,
    SQLiteStore,
    SupabaseStore,
    checkpoint_scope,
    recording_id,
)
from app.services.dedupe import TTLDedupeStore
from app.services.llm import LLMError
from app.services.processor import RecordingProcessor
from app.services.usage import capture_usage, recording_usage
from tests.conftest import FakeLLM, FakeTranscriber, make_job_request
from tests.test_pipeline import build_pipeline


async def test_crash_replay_skips_transcription_and_completed_editors(settings, tmp_path):
    s = settings.model_copy(
        update={"checkpoint_backend": "sqlite", "checkpoint_sqlite_path": tmp_path / "db.sqlite"}
    )

    class FailReels(FakeLLM):
        cache_identity = "FakeLLM"

        async def chat_json(self, **kwargs):
            if kwargs["label"].startswith("reels_editor"):
                raise LLMError("simulated crash")
            return await super().chat_json(**kwargs)

    transcriber = FakeTranscriber()
    first, _, _ = build_pipeline(s, llm=FailReels(), transcriber=transcriber)
    request = make_job_request(SourceMode.PRIVATE)
    with pytest.raises(LLMError):
        await first.run(request)
    assert (await first.checkpoints.get(recording_id(request), "status"))["status"] == "interrupted"

    class NoSTT:
        async def transcribe_chunk(self, *args):
            pytest.fail("Whisper repeated after saved transcript")

    second, llm, delivery = build_pipeline(s, transcriber=NoSTT())
    result = await second.run(request)
    assert result.reels and not delivery.channel_posts
    assert all(c["label"].startswith("reels_editor") for c in llm.calls)
    assert (await second.checkpoints.get(recording_id(request), "status"))["status"] == "complete"


async def test_reforward_same_private_audio_reuses_transcript(settings, tmp_path):
    s = settings.model_copy(
        update={"checkpoint_backend": "sqlite", "checkpoint_sqlite_path": tmp_path / "db.sqlite"}
    )
    first, _, _ = build_pipeline(s)
    request = make_job_request(SourceMode.PRIVATE)
    await first.run(request)
    second, llm, delivery = build_pipeline(s)
    result = await second.run(request.model_copy(update={"message_id": 999}))
    assert result.metadata.source_message_id == 999 and not llm.calls
    assert not delivery.channel_posts


def test_channel_private_checkpoint_id_isolation():
    assert recording_id(make_job_request(SourceMode.CHANNEL)) != recording_id(
        make_job_request(SourceMode.PRIVATE)
    )


async def test_publication_marker_prevents_duplicate_teaser(settings, tmp_path):
    s = settings.model_copy(
        update={"checkpoint_backend": "sqlite", "checkpoint_sqlite_path": tmp_path / "db.sqlite"}
    )
    first, _, _ = build_pipeline(s)
    await first.run(make_job_request())
    second, _, delivery = build_pipeline(s)
    result = await second.run(make_job_request())
    assert result.metadata.published and not delivery.channel_posts


async def test_supabase_roundtrip_and_auth_not_in_payload():
    rows = {}

    def handler(req):
        assert req.headers["apikey"] == "service-test"
        if req.method == "POST":
            value = json.loads(req.content)
            assert "service-test" not in req.content.decode()
            rows[(value["job"], value["stage"])] = value["data"]
            return httpx.Response(201)
        assert req.url.params["select"] == "data"
        return httpx.Response(200, json=[{"data": rows[("job", "transcript")]}])

    http = httpx.AsyncClient(
        base_url="https://example.supabase.co/rest/v1/", transport=httpx.MockTransport(handler)
    )
    store = SupabaseStore("https://example.supabase.co", "service-test", http)
    await store.put("job", "transcript", {"text": "saved"})
    assert await store.get("job", "transcript") == {"text": "saved"}
    await store.aclose()


async def test_supabase_outage_does_not_silently_fallback_to_ephemeral_storage():
    http = httpx.AsyncClient(
        base_url="https://example.supabase.co/rest/v1/",
        transport=httpx.MockTransport(lambda req: httpx.Response(503)),
    )
    store = SupabaseStore("https://example.supabase.co", "secret", http)
    with pytest.raises(CheckpointError):
        await store.get("job", "stage")
    await store.aclose()


async def test_reported_usage_persists_and_aggregates_across_resume(tmp_path):
    store = SQLiteStore(tmp_path / "usage.db")
    for cost in [0.01, 0.02]:
        with checkpoint_scope(store, "job"), recording_usage(60, "job"):
            await capture_usage(
                {"provider": "openrouter", "reported_cost": cost, "input_tokens": 100}
            )
    events = (await store.get("job", "usage"))["events"]
    assert len(events) == 2 and sum(e["reported_cost"] for e in events) == 0.03


@pytest.mark.parametrize("stop", [False, True])
async def test_budget_warning_does_not_halt_unless_configured(settings, stop):
    pipeline, _, delivery = build_pipeline(settings)

    class BudgetLLM(FakeLLM):
        async def monthly_usage(self):
            return 16

    pipeline._miner._llm = BudgetLLM()
    s = settings.model_copy(update={"stop_on_budget_exceeded": stop, "openrouter_allow_paid": True})
    processor = RecordingProcessor(s, pipeline, delivery, TTLDedupeStore())
    if stop:
        with pytest.raises(RuntimeError, match="monthly budget"):
            await processor.process(make_job_request(SourceMode.PRIVATE))
    else:
        assert (await processor.process(make_job_request(SourceMode.PRIVATE))).atoms
    assert "Бюджет OpenRouter" in delivery.owner_messages[0]
    assert not delivery.channel_posts


async def test_semantic_public_teaser_never_contains_private_drafts(settings):
    from app.agents.semantic_miner import SemanticMinerAgent
    from app.services.prompts import PromptLibrary
    from tests.test_semantic import SemanticFake, art_atoms

    class ContentFake(SemanticFake):
        async def chat_text(self, **kwargs):
            if kwargs["label"].startswith("channel_teaser"):
                assert "PRIVATE DRAFT SECRET" not in kwargs["user"]
                return "Listen to these ideas"
            if kwargs["label"].startswith("threads_editor"):
                return "PRIVATE DRAFT SECRET"
            return "SKIP: not suitable"

        async def chat_json(self, **kwargs):
            if kwargs["schema_name"] == "RoutingResult":
                return {
                    "routes": [
                        {"atom_id": a["id"], "route": "BOTH", "reason": "relevant"}
                        for a in json.loads(kwargs["user"])
                    ]
                }
            if kwargs["schema_name"] == "ChannelTeaser":
                assert "PRIVATE DRAFT SECRET" not in kwargs["user"]
                return {"teaser": "Listen to these ideas", "timestamps": [], "reasoning": ""}
            if kwargs["schema_name"] == "ThreadsBatch":
                return {
                    "candidates": [
                        {
                            "atom_id": "w1p1.taste",
                            "draft": "PRIVATE DRAFT SECRET",
                            "readiness_score": 8,
                        }
                    ],
                    "rejected": [],
                }
            if kwargs["schema_name"] == "ReelsBatch":
                return {"candidates": [], "rejected": []}
            return await super().chat_json(**kwargs)

    s = settings.model_copy(
        update={
            "semantic_pipeline_enabled": True,
            "quality_auditor": "none",
            "text_provider": "openrouter",
        }
    )
    fake = ContentFake(art_atoms())
    pipeline, _, delivery = build_pipeline(s, llm=fake)
    pipeline._miner = SemanticMinerAgent(fake, PromptLibrary(s.prompts_dir), s)
    result = await pipeline.run(make_job_request(SourceMode.CHANNEL))
    assert result.semantic and result.threads
    assert all("PRIVATE DRAFT SECRET" not in body for body, _ in delivery.channel_posts)
    assert len(result.atoms) == 3


async def test_attempt_cap_survives_malformed_response_without_usage(settings, tmp_path):
    from app.services.openrouter_client import OpenRouterClient
    from app.services.usage import restore_usage

    s = settings.model_copy(
        update={"openrouter_api_key": "test", "openrouter_max_requests_per_recording": 1}
    )
    store = SQLiteStore(tmp_path / "db.sqlite")
    requests = []

    def handler(req):
        requests.append(req)
        return httpx.Response(200, text="broken response without usage")

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        client = OpenRouterClient(s, http)
        with checkpoint_scope(store, "job"), recording_usage(60, "job"):
            with pytest.raises(LLMError):
                await client.chat_json(system="s", user="u")
        with checkpoint_scope(store, "job"), recording_usage(60, "job") as usage:
            await restore_usage(store, "job", usage)
            with pytest.raises(LLMError, match="request limit"):
                await client.chat_json(system="s", user="u")
    assert len(requests) == 1
