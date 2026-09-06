"""Recover structured generation failures without raising output budgets."""

import json

import httpx
import pytest

from app.agents.content_miner import ContentMinerAgent
from app.models.transcript import Transcript, TranscriptSegment
from app.services.groq_client import GroqClient, GroqGenerationError
from app.services.prompts import PromptLibrary


async def test_json_fallback_preserves_schema_and_low_reasoning(settings):
    calls = []
    schema = {"type": "object", "properties": {"atoms": {"type": "array"}}}

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "unsupported_response_format",
                        "message": "json_schema is not supported by this model",
                    }
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": '{"atoms":[]}'}}]},
        )

    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        assert await GroqClient(settings, client=http).chat_json(
            system="s", user="u", schema=schema, max_tokens=1200
        ) == {"atoms": []}
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert '"atoms"' in calls[1]["messages"][0]["content"]
    assert all(c["reasoning_effort"] == "low" and c["max_tokens"] == 1200 for c in calls)


@pytest.mark.parametrize("content", ["", '{"atoms":[]}', '{"atoms":'])
def test_truncated_generation_is_never_accepted(content):
    with pytest.raises(GroqGenerationError, match="budget exhausted"):
        GroqClient._extract_json(
            {"choices": [{"finish_reason": "length", "message": {"content": content}}]}, "miner"
        )


async def test_miner_splits_failed_window_without_losing_segments(settings):
    from app.agents.content_miner import TranscriptWindow

    calls = []

    class LLM:
        async def chat_json(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= 2:
                raise GroqGenerationError("invalid JSON")
            return {
                "atoms": [
                    {
                        "title": "idea",
                        "idea": "thought",
                        "type": "idea",
                        "start_seconds": 0,
                        "end_seconds": 20,
                    }
                ]
            }

    segments = [
        TranscriptSegment(start=i * 10, end=(i + 1) * 10, text=f"unique{i}") for i in range(4)
    ]
    agent = ContentMinerAgent(LLM(), PromptLibrary(settings.prompts_dir), settings)
    result = await agent._mine_window(TranscriptWindow(0, 0, 40, segments), "s", "full")
    assert len(calls) == 4
    assert calls[0]["user"] == calls[1]["user"]
    for i in range(4):
        assert sum(f"unique{i}" in c["user"] for c in calls[2:]) == 1
    assert [c["max_tokens"] for c in calls] == [1500, 1600, 1500, 1500]
    assert len(result) == 2


async def test_miner_generation_failure_is_bounded(settings):
    calls = []

    class LLM:
        async def chat_json(self, **kwargs):
            calls.append(kwargs)
            raise GroqGenerationError("invalid JSON")

    transcript = Transcript(
        duration=100,
        segments=[TranscriptSegment(start=i, end=i + 1, text="word") for i in range(100)],
    )
    with pytest.raises(GroqGenerationError):
        await ContentMinerAgent(LLM(), PromptLibrary(settings.prompts_dir), settings).mine(
            transcript
        )
    assert len(calls) == 3  # original, one retry, first reduced window; no recursion
