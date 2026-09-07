import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.agents.base import StructuredAgent
from app.agents.channel_teaser import ChannelTeaserAgent
from app.agents.reels_editor import ReelsEditorAgent
from app.agents.semantic_miner import SemanticMinerAgent
from app.agents.threads_editor import ThreadsEditorAgent
from app.models.atoms import ContentAtom
from app.models.content import ChannelTeaser
from app.services.checkpoints import SQLiteStore, checkpoint_scope
from app.services.llm import LLMError
from app.services.openrouter_client import OpenRouterClient
from app.services.prompts import PromptLibrary
from app.services.usage import recording_usage
from tests.test_provider_routing import info, router


@pytest.fixture
def paid(settings):
    return settings.model_copy(
        update={
            "text_provider": "openrouter",
            "semantic_pipeline_enabled": True,
            "openrouter_allow_paid": True,
            "openrouter_model": "paid/model",
            "openrouter_primary_model": "paid/model",
            "openrouter_api_key": "fake",
            "groq_tpm_limit": 1,
            "text_max_input_tokens": 1,
            "semantic_max_input_tokens": 1,
            "semantic_max_output_tokens": 1,
            "text_extraction_max_tokens": 1,
            "text_teaser_max_tokens": 1,
            "text_threads_max_tokens": 1,
            "text_reels_max_tokens": 1,
            "openrouter_max_retries": 0,
        }
    )


def completion(text):
    return httpx.Response(
        200,
        json={
            "model": "paid/model",
            "usage": {
                "prompt_tokens": 4000,
                "completion_tokens": 16000,
                "completion_tokens_details": {"reasoning_tokens": 12000},
                "cost": 0.1,
            },
            "choices": [{"finish_reason": "stop", "message": {"content": text}}],
        },
    )


@pytest.mark.parametrize("method", ["chat_json", "chat_text"])
async def test_old_caps_cannot_limit_input_reasoning_or_output(paid, method):
    def handler(req):
        body = json.loads(req.content)
        assert "max_tokens" not in body and "max_completion_tokens" not in body
        assert body["reasoning"] == {"effort": "medium"}
        if method == "chat_json":
            assert body["response_format"]["json_schema"]["strict"]
            return completion('{"teaser":"valid"}')
        assert "response_format" not in body
        assert body["messages"][0]["content"] == "Write a draft"
        return completion("Текст без JSON. " * 1000)

    client = router(paid, [info("paid/model")], handler)
    try:
        with recording_usage(300, "limits") as usage:
            if method == "chat_json":
                agent = StructuredAgent(client, PromptLibrary(paid.prompts_dir), paid)
                await agent.request(
                    ChannelTeaser,
                    system="s",
                    user="word " * 15000,
                    request_label="semantic_extraction",
                    max_tokens=1,
                )
            else:
                # Label deliberately selects medium to prove quality was not lowered.
                assert await client.chat_text(
                    system="Write a draft",
                    user="word " * 15000,
                    label="semantic_explanation",
                    max_tokens=1,
                )
            assert usage.requests == 1
            assert usage.input_tokens == 4000 and usage.output_tokens == 16000
            assert usage.reasoning_tokens == 12000 and usage.reported_cost == 0.1
    finally:
        await client.aclose()


async def test_one_optional_global_ceiling_overrides_stage_limits(paid):
    s = paid.model_copy(update={"openrouter_max_output_tokens": 32768})

    def handler(req):
        assert json.loads(req.content)["max_tokens"] == 32768
        return completion("{}")

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        await OpenRouterClient(s, http).chat_json(system="s", user="u", max_tokens=400)


def atoms():
    return [
        ContentAtom(
            id=f"a{i}",
            label=f"Идея {i}",
            key_claim="Главная мысль",
            start_seconds=i * 20,
            end_seconds=(i + 1) * 20,
            supporting_context="Значимый контекст " * 100,
            confidence=1,
            content_route="BOTH",
            supports_atom_ids=["support"],
        )
        for i in range(7)
    ] + [
        ContentAtom(
            id="support",
            label="Доказательство",
            key_claim="Уточнение, которое нельзя потерять",
            start_seconds=160,
            end_seconds=180,
            content_route="ARCHIVE_ONLY",
        )
    ]


@pytest.mark.parametrize(
    "editor,field", [(ThreadsEditorAgent, "draft"), (ReelsEditorAgent, "script")]
)
async def test_writers_keep_plain_text_source_links_and_selection(paid, editor, field):
    # Enough text to exceed the old 4000/5000-character model validation cap.
    output = 'Слова автора, "кавычки", переносы.\n' * 200
    llm = type("TextOnly", (), {})()
    llm.chat_text = AsyncMock(return_value=output)
    llm.chat_json = AsyncMock(side_effect=AssertionError("writer must not request JSON"))
    agent = editor(llm, PromptLibrary(paid.prompts_dir), paid)
    result = await agent.edit(atoms())
    assert len(result.candidates) == 4
    assert llm.chat_text.await_count == 4
    for item, call in zip(result.candidates, llm.chat_text.call_args_list, strict=False):
        assert getattr(item, field) == output.strip()
        atom = next(a for a in atoms() if a.id == item.atom_id)
        assert item.start_seconds == atom.start_seconds
        assert item.end_seconds == atom.end_seconds
        assert "Уточнение, которое нельзя потерять" in call.kwargs["user"]
        assert "Return JSON" not in call.kwargs["system"]
        assert "max_tokens" not in call.kwargs and "schema" not in call.kwargs


async def test_text_resume_reuses_output_and_does_not_collide_with_json(paid, tmp_path):
    llm = type("TextOnly", (), {})()
    llm.chat_text = AsyncMock(return_value="Готовый черновик")
    agent = StructuredAgent(llm, PromptLibrary(paid.prompts_dir), paid)
    store = SQLiteStore(tmp_path / "checkpoints.sqlite3")
    with checkpoint_scope(store, "recording"):
        assert await agent.request_text(system="s", user="u") == "Готовый черновик"
        assert await agent.request_text(system="s", user="u") == "Готовый черновик"
    assert llm.chat_text.await_count == 1


async def test_teaser_is_text_and_timestamps_are_local(paid, transcript):
    llm = type("TextOnly", (), {})()
    llm.chat_text = AsyncMock(return_value="В этой истории есть поворот.")
    s = paid.model_copy(update={"teaser_include_timestamps": True})
    result = await ChannelTeaserAgent(llm, PromptLibrary(s.prompts_dir), s).write(
        transcript, atoms()
    )
    assert result.teaser == "В этой истории есть поворот."
    assert all(t.seconds <= transcript.duration for t in result.timestamps)
    assert not result.reasoning
    assert "schema" not in llm.chat_text.call_args.kwargs


def test_semantic_global_context_not_capped_by_old_env(paid, transcript):
    agent = SemanticMinerAgent(None, PromptLibrary(paid.prompts_dir), paid)
    assert transcript.segments[0].text in agent._input(transcript, [])


async def test_text_fallback_keeps_text_contract(paid):
    s = paid.model_copy(update={"openrouter_primary_fallback_model": "paid/fallback"})
    seen = []

    def handler(req):
        body = json.loads(req.content)
        assert "response_format" not in body
        seen.append(body["model"])
        if len(seen) == 1:
            return httpx.Response(404, json={"error": {"message": "No endpoint"}})
        return completion("Черновик")

    client = router(s, [info("paid/model"), info("paid/fallback")], handler)
    try:
        assert await client.chat_text(system="s", user="u", label="threads_editor/a1") == "Черновик"
        assert seen == ["paid/model", "paid/fallback"]
    finally:
        await client.aclose()


async def test_text_free_quota_never_switches_to_paid(paid):
    s = paid.model_copy(
        update={"openrouter_model": "openrouter/free", "openrouter_allow_paid": False}
    )
    seen = []

    def handler(req):
        body = json.loads(req.content)
        assert body["provider"]["max_price"]["completion"] == 0
        assert "response_format" not in body
        seen.append(body["model"])
        return httpx.Response(
            429, json={"error": {"message": "quota exhausted"}}, headers={"retry-after": "10"}
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(LLMError):
            await OpenRouterClient(s, http).chat_text(system="s", user="u")
    assert seen == ["openrouter/free"]


async def test_rejection_explanation_stays_text(paid):
    llm = type("TextOnly", (), {})()
    llm.chat_text = AsyncMock(return_value="SKIP: В источнике не хватает аргумента.")
    result = await ThreadsEditorAgent(llm, PromptLibrary(paid.prompts_dir), paid).edit(atoms()[:1])
    assert result.candidates == []
    assert result.rejected == ["a0 — В источнике не хватает аргумента."]
