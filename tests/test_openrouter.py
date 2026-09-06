"""OpenRouter routing, cost safety, local validation and bounded pipeline calls.

All HTTP requests use MockTransport; no real API calls.
"""

import json
import time

import httpx
import pytest

from app.agents.base import StructuredAgent, json_schema_for
from app.models.content import ChannelTeaser
from app.models.transcript import Transcript, TranscriptSegment
from app.services.llm import LLMError, LLMGenerationError
from app.services.openrouter_client import OpenRouterClient
from app.services.prompts import PromptLibrary
from app.services.usage import recording_usage
from app.utils.retry import retry_async
from tests.conftest import make_job_request
from tests.test_pipeline import build_pipeline


@pytest.fixture
def or_settings(settings):
    return settings.model_copy(
        update={
            "text_provider": "openrouter",
            "openrouter_api_key": "router-test",
            "openrouter_model": "openrouter/free",
        }
    )


def completion(content, model="some/free-routed-model", cost=0):
    return httpx.Response(
        200,
        json={
            "model": model,
            "provider": "ExampleProvider",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 10},
                "cost": cost,
            },
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        },
    )


async def test_structured_success_auth_routed_model_and_summary(or_settings, caplog):
    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/api/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer router-test"
        assert body["model"] == "openrouter/free"
        assert body["provider"]["max_price"] == {
            "prompt": 0,
            "completion": 0,
            "request": 0,
            "image": 0,
        }
        assert body["provider"]["require_parameters"] is False
        assert body["response_format"]["json_schema"]["strict"] is True
        assert "reasoning_effort" not in body
        assert body["reasoning"] == {"effort": "none", "enabled": False}
        return completion('{"teaser":"done","timestamps":[],"reasoning":""}')

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        client = OpenRouterClient(or_settings, http)
        agent = StructuredAgent(client, PromptLibrary(or_settings.prompts_dir), or_settings)
        with caplog.at_level("INFO"), recording_usage(60, "test") as usage:
            result = await agent.request(ChannelTeaser, system="s", user="u")
        assert result.teaser == "done"
        assert usage.requests == 1 and usage.input_tokens == 100 and usage.output_tokens == 50
    assert "some/free-routed-model" in caplog.text
    assert "reasoning_tokens=10" in caplog.text and "reported_cost=0" in caplog.text
    assert "duration_seconds=60" in caplog.text


@pytest.mark.parametrize("bad", ["not json", '{"teaser":', "[]", '{"teaser":""}'])
async def test_bounded_json_and_pydantic_repair(or_settings, bad):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return completion(
            bad if len(seen) == 1 else '{"teaser":"fixed","timestamps":[],"reasoning":""}'
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(or_settings, http), PromptLibrary(or_settings.prompts_dir), or_settings
        )
        assert (await agent.request(ChannelTeaser, system="s", user="u")).teaser == "fixed"
    assert len(seen) == 2


async def test_invalid_json_stops_after_two(or_settings):
    seen = []

    def handler(req):
        seen.append(req)
        return completion("bad")

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(or_settings, http), PromptLibrary(or_settings.prompts_dir), or_settings
        )
        with pytest.raises(LLMGenerationError):
            await agent.request(ChannelTeaser, system="s", user="u")
    assert len(seen) == 2


async def test_truncation_logs_model_and_remains_bounded(or_settings, caplog):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        assert body["reasoning"] == {"effort": "none", "enabled": False}
        return httpx.Response(
            200,
            json={
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "usage": {
                    "prompt_tokens": 838,
                    "completion_tokens": 3000,
                    "completion_tokens_details": {"reasoning_tokens": 2891},
                    "cost": 0,
                },
                "choices": [{"finish_reason": "length", "message": {"content": '{"teaser":'}}],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(or_settings, http), PromptLibrary(or_settings.prompts_dir), or_settings
        )
        with caplog.at_level("INFO"), pytest.raises(LLMGenerationError, match="output_budget=3000"):
            with recording_usage(105, "truncated") as usage:
                await agent.request(ChannelTeaser, system="s", user="u", max_tokens=3000)
    assert len(seen) == usage.requests == 2
    assert usage.output_tokens == 6000
    assert "finish_reason=length" in caplog.text and "reasoning_tokens=2891" in caplog.text


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
async def test_reasoning_opt_in_uses_unified_control(or_settings, effort):
    configured = or_settings.model_copy(update={"openrouter_reasoning_effort": effort})

    def handler(req):
        assert json.loads(req.content)["reasoning"] == {"effort": effort}
        return completion("{}")

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        assert await OpenRouterClient(configured, http).chat_json(system="s", user="u") == {}


async def test_unsupported_reasoning_is_not_silently_removed(or_settings):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "param": "reasoning",
                    "message": "reasoning none not supported for this model",
                }
            },
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(or_settings, http), PromptLibrary(or_settings.prompts_dir), or_settings
        )
        with pytest.raises(LLMError, match="reasoning none not supported"):
            await agent.request(ChannelTeaser, system="s", user="u")
    assert len(seen) == 1


@pytest.mark.parametrize("html_error", [False, True])
async def test_429_honors_retry_after_and_is_bounded(or_settings, monkeypatch, html_error):
    waits, seen = [], []

    async def sleep(delay):
        waits.append(delay)

    async def retry(operation, **kwargs):
        return await retry_async(operation, **kwargs, sleep=sleep)

    monkeypatch.setattr("app.services.openrouter_client.retry_async", retry)

    def handler(req):
        seen.append(json.loads(req.content))
        if html_error:
            return httpx.Response(429, headers={"retry-after": "125"}, text="<h1>Quota</h1>")
        return httpx.Response(
            429, headers={"retry-after": "125"}, json={"error": {"message": "free quota exhausted"}}
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        with (
            recording_usage(60, "failed") as usage,
            pytest.raises(LLMError, match="retries exhausted"),
        ):
            await OpenRouterClient(or_settings, http).chat_json(system="s", user="u")
    assert len(seen) == 3 and waits == [125, 125] and usage.requests == 3
    assert all(body["model"] == "openrouter/free" for body in seen)


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_unrelated_errors_never_downgrade_strict_schema(or_settings, status):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(status, json={"error": {"message": "Invalid request or schema"}})

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(or_settings, http), PromptLibrary(or_settings.prompts_dir), or_settings
        )
        with pytest.raises(LLMError):
            await agent.request(ChannelTeaser, system="s", user="u")
    assert len(seen) == 1
    assert seen[0]["response_format"]["json_schema"]["strict"] is True


async def test_request_cap_includes_failed_json_attempts(or_settings, caplog):
    configured = or_settings.model_copy(update={"openrouter_max_requests_per_recording": 1})
    seen = []

    def handler(req):
        seen.append(req)
        return completion("not json")

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(configured, http), PromptLibrary(configured.prompts_dir), configured
        )
        with caplog.at_level("INFO"), pytest.raises(LLMError, match="request limit"):
            with recording_usage(60, "failed-cap") as usage:
                await agent.request(ChannelTeaser, system="s", user="u")
    assert len(seen) == usage.requests == 1
    assert "Recording usage recording=failed-cap" in caplog.text
    assert "input_tokens=100" in caplog.text


async def test_no_free_models_does_not_downgrade_or_spend(or_settings):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(
            404, json={"error": {"message": "No endpoints found for free models"}}
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(LLMError):
            await OpenRouterClient(or_settings, http).chat_json(
                system="s", user="u", schema=json_schema_for(ChannelTeaser)
            )
    assert len(seen) == 1 and "models" not in seen[0]


@pytest.mark.parametrize("model", ["openai/gpt-4o", "openrouter/auto", "anthropic/claude-sonnet-4"])
def test_paid_disabled_blocks_before_http(or_settings, model):
    with pytest.raises(LLMError, match="Paid OpenRouter model blocked"):
        OpenRouterClient(or_settings.model_copy(update={"openrouter_model": model}))


async def test_explicit_paid_model_requires_opt_in(or_settings):
    configured = or_settings.model_copy(
        update={"openrouter_allow_paid": True, "openrouter_model": "some/paid-model"}
    )

    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "some/paid-model"
        assert "max_price" not in body["provider"]
        assert body["provider"]["require_parameters"] is True
        return completion("{}", cost=0.001)

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        assert await OpenRouterClient(configured, http).chat_json(system="s", user="u") == {}


async def test_explicit_unsupported_schema_keeps_free_model(or_settings):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "param": "response_format",
                        "message": "response_format not supported",
                    }
                },
            )
        assert "response_format" not in body
        assert body["reasoning"] == {"effort": "none", "enabled": False}
        assert "Return only JSON matching" in body["messages"][0]["content"]
        return completion('{"teaser":"ok"}')

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        await OpenRouterClient(or_settings, http).chat_json(
            system="s", user="u", schema=json_schema_for(ChannelTeaser)
        )
    assert len(seen) == 2
    assert all(
        b["model"] == "openrouter/free" and b["provider"]["max_price"]["request"] == 0 for b in seen
    )


async def test_direct_free_model_requires_parameters_and_disables_reasoning(or_settings):
    configured = or_settings.model_copy(
        update={"openrouter_model": "nvidia/nemotron-3-super-120b-a12b:free"}
    )

    def handler(req):
        body = json.loads(req.content)
        assert body["provider"]["require_parameters"] is True
        assert body["provider"]["max_price"]["completion"] == 0
        assert body["reasoning"] == {"effort": "none", "enabled": False}
        assert body["response_format"]["json_schema"]["strict"] is True
        return completion('{"teaser":"ok","timestamps":[],"reasoning":""}')

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            OpenRouterClient(configured, http), PromptLibrary(configured.prompts_dir), configured
        )
        assert (await agent.request(ChannelTeaser, system="s", user="u")).teaser == "ok"


async def test_hour_recording_has_bounded_calls_without_groq_text(or_settings, monkeypatch):
    from app.services.groq_client import GroqClient
    from app.services.tpm import RollingTPM

    async def forbidden(*args, **kwargs):
        pytest.fail("Groq text/TPM used on OpenRouter path")

    monkeypatch.setattr(GroqClient, "chat_json", forbidden)
    monkeypatch.setattr(RollingTPM, "reserve", forbidden)
    names, counts = [], {"extraction": 0}

    def handler(req):
        body = json.loads(req.content)
        name = body["response_format"]["json_schema"]["name"]
        assert body["reasoning"] == {"effort": "none", "enabled": False}
        names.append(name)
        if name == "AtomExtraction":
            counts["extraction"] += 1
            n = counts["extraction"]
            data = {
                "atoms": [
                    {
                        "title": f"unique{n}x{i}",
                        "idea": f"thought{n}x{i}",
                        "type": "business_insight",
                        "start_seconds": (n - 1) * 690,
                        "end_seconds": (n - 1) * 690 + 30,
                    }
                    for i in range(8)
                ]
            }
        elif name == "ChannelTeaser":
            data = {"teaser": "done", "timestamps": [], "reasoning": ""}
        else:
            data = {"candidates": [], "rejected": []}
        return completion(json.dumps(data))

    transcript = Transcript(
        duration=3600,
        segments=[
            TranscriptSegment(start=i * 30, end=(i + 1) * 30, text=f"idea{i}") for i in range(120)
        ],
    )
    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    ) as http:
        client = OpenRouterClient(or_settings, http)
        pipeline, _, _ = build_pipeline(or_settings, llm=client)
        with recording_usage(3600, "hour") as usage:
            result = await pipeline.analyse(make_job_request(), transcript, 1, time.monotonic())
    assert counts["extraction"] == 6
    assert names.count("ThreadsBatch") == 2 and names.count("ReelsBatch") == 2
    assert names.count("ChannelTeaser") == 1 and "AtomEnrichment" not in names
    assert usage.requests == 11 and len(result.atoms) == 48
