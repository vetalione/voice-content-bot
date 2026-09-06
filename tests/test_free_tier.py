"""Free-tier request sizing, editor batching and bounded TPM recovery."""

import json
import re

import httpx
import pytest

from app.agents.reels_editor import ReelsEditorAgent
from app.agents.threads_editor import ThreadsEditorAgent
from app.services.groq_client import GroqClient, GroqError, GroqQuotaError
from app.services.prompts import PromptLibrary
from app.services.token_budget import request_tokens
from app.utils.retry import RetryableError, retry_async
from tests.test_agents import atom


@pytest.mark.parametrize(
    "editor,budget,score",
    [
        (ThreadsEditorAgent, 1200, "readiness_score"),
        (ReelsEditorAgent, 1600, "score"),
    ],
)
async def test_oversized_atoms_split_and_results_ranked(settings, editor, budget, score):
    calls = []

    class LLM:
        async def chat_json(self, **kwargs):
            calls.append(kwargs)
            ids = re.findall(r"### ATOM (a\d+)", kwargs["user"])
            return {
                "candidates": [
                    dict(
                        atom_id=i,
                        draft="draft",
                        concept="concept",
                        script="script",
                        **{score: int(i[1:])},
                    )
                    for i in ids
                ]
            }

    # Two large Cyrillic contexts exceed the cap; each alone fits.
    atoms = [
        atom(
            id=f"a{i}",
            supporting_context="я" * 4000,
            description="д" * 1200,
            key_claim="к" * 800,
            start_seconds=i * 60,
            end_seconds=(i + 1) * 60,
        )
        for i in range(7)
    ]
    agent = editor(LLM(), PromptLibrary(settings.prompts_dir), settings)
    result = await agent.edit(atoms)
    seen = []
    for call in calls:
        ids = re.findall(r"### ATOM (a\d+)", call["user"])
        assert 1 <= len(ids) <= 2
        seen.extend(ids)
        assert call["max_tokens"] == budget
        assert request_tokens(call["system"], call["user"], call["schema"]) + budget <= 8000
    assert sorted(seen) == sorted(a.id for a in atoms)
    assert len(calls) > 4
    assert [c.atom_id for c in result.candidates] == ["a6", "a5", "a4", "a3", "a2"]
    assert result.candidates[0].start_seconds == 360


async def test_structurally_oversized_request_never_sent(settings):
    def handler(request):
        pytest.fail("Oversized request reached network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GroqClient(settings, client=http)
        with pytest.raises(GroqError, match="exceeds TPM limit"):
            await client.chat_json(system="я" * 15000, user="u", max_tokens=1200)


@pytest.mark.parametrize("status", [413, 429])
async def test_server_oversize_not_retried(settings, status):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(
            status, json={"error": {"message": "TPM: Limit 8000, Requested 8202"}}
        )

    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(GroqError):
            await GroqClient(settings, client=http).chat_json(system="s", user="u", max_tokens=400)
    assert count == 1


async def test_tpm_retry_after_then_success(settings, monkeypatch, caplog):
    waits = []
    calls = []

    async def sleep(delay):
        waits.append(delay)

    # Use the real retry algorithm, with only sleeping replaced.
    async def retry(operation, **kwargs):
        return await retry_async(operation, **kwargs, sleep=sleep)

    monkeypatch.setattr("app.services.groq_client.retry_async", retry)

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "61"},
                json={"error": {"message": "tokens per minute exhausted"}},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    with caplog.at_level("INFO"):
        async with httpx.AsyncClient(
            base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
        ) as http:
            assert (
                await GroqClient(settings, client=http).chat_json(
                    system="s", user="u", max_tokens=400
                )
                == {}
            )
    assert waits == [61]
    assert len(calls) == 2
    assert "approximate_input_tokens=" in caplog.text and "output_budget=400" in caplog.text


async def test_tpm_exhaustion_is_bounded(settings, monkeypatch):
    waits = []

    async def sleep(delay):
        waits.append(delay)

    async def retry(operation, **kwargs):
        return await retry_async(operation, **kwargs, sleep=sleep)

    monkeypatch.setattr("app.services.groq_client.retry_async", retry)

    def handler(request):
        return httpx.Response(429, json={"error": {"message": "tokens per minute exhausted"}})

    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(GroqQuotaError):
            await GroqClient(settings, client=http).chat_json(system="s", user="u", max_tokens=400)
    assert waits == [60] * settings.groq_max_retries


async def test_retry_after_not_shortened_by_backoff_cap():
    waits = []

    async def sleep(delay):
        waits.append(delay)

    async def operation():
        if not waits:
            raise RetryableError("TPM", retry_after=150)
        return True

    assert await retry_async(operation, attempts=2, max_delay=120, sleep=sleep)
    assert waits == [150]


def test_stage_style_excludes_other_editor_examples(tmp_path):
    (tmp_path / "VOICE_STYLE.md").write_text(
        "## Tone\nplain\n## Threads examples\nTHREADS_ONLY\n## Reels examples\nREELS_ONLY\n"
    )
    library = PromptLibrary(tmp_path)
    assert "REELS_ONLY" not in library.voice_style("threads_editor")
    assert "THREADS_ONLY" not in library.voice_style("reels_editor")
    assert "plain" in library.voice_style("threads_editor")


async def test_miner_and_teaser_use_stage_caps(settings, transcript):
    from app.agents.channel_teaser import ChannelTeaserAgent
    from app.agents.content_miner import ContentMinerAgent
    from tests.conftest import FakeLLM

    llm = FakeLLM()
    prompts = PromptLibrary(settings.prompts_dir)
    mined = await ContentMinerAgent(llm, prompts, settings).mine(transcript)
    await ChannelTeaserAgent(llm, prompts, settings).write(transcript, mined.atoms)
    assert llm.calls[0]["max_tokens"] == 1500
    assert llm.calls[-1]["max_tokens"] == 400
    assert "TRANSCRIPT OPENING" not in llm.calls[-1]["user"]
