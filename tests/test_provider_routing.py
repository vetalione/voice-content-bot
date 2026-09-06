import json
import sys
import time
import types
from typing import ClassVar

import httpx
import pytest

from app.agents.base import StructuredAgent
from app.models.content import ChannelTeaser
from app.services.claude_client import ClaudeAgentClient
from app.services.llm import LLMError
from app.services.openrouter_client import OpenRouterClient
from app.services.prompts import PromptLibrary
from app.services.provider_routing import ModelCatalog, TextRouter
from app.services.usage import recording_usage


@pytest.fixture
def configured(settings):
    return settings.model_copy(
        update={
            "text_provider": "openrouter",
            "semantic_pipeline_enabled": True,
            "openrouter_primary_model": "primary/model",
            "openrouter_escalation_model": "quality/model",
            "openrouter_allow_paid": True,
            "openrouter_api_key": "fake",
        }
    )


def info(id, price="0.000001", params=None):
    return {
        "id": id,
        "context_length": 262144,
        "supported_parameters": params
        if params is not None
        else ["reasoning", "structured_outputs", "response_format"],
        "pricing": {"prompt": price, "completion": price},
    }


def response():
    return httpx.Response(
        200,
        json={
            "model": "primary/model",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "cost": 0.0001},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"teaser":"ok","timestamps":[],"reasoning":""}'},
                }
            ],
        },
    )


def router(settings, models, handler):
    http = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    )
    catalog = ModelCatalog(http)
    catalog.models = {m["id"]: m for m in models}
    catalog.loaded_at = time.monotonic()
    return TextRouter(
        settings,
        catalog=catalog,
        client_factory=lambda settings, **kwargs: OpenRouterClient(settings, http, **kwargs),
    )


async def test_missing_primary_does_not_use_free_or_escalation(configured):
    seen = []
    client = router(
        configured, [info("quality/model"), info("openrouter/free")], lambda r: seen.append(r)
    )
    try:
        with pytest.raises(LLMError, match="Primary unavailable"):
            await client.chat_json(system="s", user="u")
        assert not seen
    finally:
        await client.aclose()


async def test_explicit_fallback_used_after_catalog_disappearance(configured, caplog):
    s = configured.model_copy(
        update={
            "openrouter_primary_fallback_model": "fallback/model",
            "openrouter_allow_escalation": True,
        }
    )

    def handler(req):
        assert json.loads(req.content)["model"] == "fallback/model"
        return response()

    client = router(s, [info("fallback/model")], handler)
    try:
        await client.chat_json(system="s", user="u")
        assert "Configured primary model unavailable" in caplog.text
    finally:
        await client.aclose()


async def test_no_automatic_paid_escalation_after_429(configured, monkeypatch):
    s = configured.model_copy(
        update={"openrouter_max_retries": 0, "openrouter_allow_escalation": True}
    )
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["model"])
        return httpx.Response(429, json={"error": {"message": "quota"}})

    client = router(s, [info("primary/model"), info("quality/model")], handler)
    try:
        with pytest.raises(LLMError):
            await client.chat_json(system="s", user="u")
        assert seen == ["primary/model"]
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "params,expected",
    [
        (["reasoning", "structured_outputs", "response_format"], "json_schema"),
        (["response_format"], "json_object"),
        ([], None),
    ],
)
async def test_model_capabilities_choose_structured_method(configured, params, expected):
    def handler(req):
        body = json.loads(req.content)
        assert body.get("response_format", {}).get("type") == expected
        assert ("reasoning" in body) == ("reasoning" in params)
        return response()

    client = router(configured, [info("primary/model", params=params)], handler)
    try:
        result = await StructuredAgent(
            client, PromptLibrary(configured.prompts_dir), configured
        ).request(ChannelTeaser, system="s", user="u", request_label="semantic_extraction")
        assert result.teaser == "ok"
    finally:
        await client.aclose()


async def test_fallback_on_endpoint_404_only_once(configured):
    s = configured.model_copy(update={"openrouter_primary_fallback_model": "fallback/model"})
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["model"])
        return httpx.Response(404, json={"error": {"message": "model unavailable"}})

    client = router(s, [info("primary/model"), info("fallback/model")], handler)
    try:
        with pytest.raises(LLMError):
            await client.chat_json(system="s", user="u")
        assert seen == ["primary/model", "fallback/model"]
    finally:
        await client.aclose()


async def test_more_expensive_fallback_is_not_silent(configured):
    s = configured.model_copy(update={"openrouter_primary_fallback_model": "fallback/model"})
    client = router(
        s,
        [info("primary/model"), info("fallback/model", "0.001")],
        lambda r: httpx.Response(404, json={"error": {"message": "unavailable"}}),
    )
    try:
        with pytest.raises(LLMError, match="More expensive"):
            await client.chat_json(system="s", user="u")
    finally:
        await client.aclose()


async def test_free_router_available_when_explicitly_selected(configured):
    client = router(
        configured.model_copy(update={"openrouter_primary_model": "openrouter/free"}),
        [info("openrouter/free")],
        lambda r: response(),
    )
    try:
        assert await client.ensure_primary() == "openrouter/free"
        assert await client.chat_json(system="s", user="u")
    finally:
        await client.aclose()


async def test_catalog_health_and_authoritative_monthly_usage(configured):
    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [info("primary/model")]})
        return httpx.Response(200, json={"data": {"usage_monthly": 10.25}})

    client = router(configured, [], handler)
    try:
        health = await client.preflight()
        assert health["status"] == "catalog_available"
        assert health["models"]["primary/model"]["exists"]
        assert await client.monthly_usage() == 10.25
    finally:
        await client.aclose()


async def test_claude_sdk_oauth_isolated_and_validated(configured, monkeypatch):
    seen = []

    class Result:
        is_error = False
        structured_output: ClassVar[dict] = {
            "teaser": "Claude audit",
            "timestamps": [],
            "reasoning": "",
        }
        usage: ClassVar[dict] = {"input_tokens": 100, "output_tokens": 50}
        total_cost_usd = 0.002
        num_turns = 1

    async def query(**kwargs):
        seen.append(kwargs)
        yield Result()

    module = types.SimpleNamespace(
        query=query, ClaudeAgentOptions=lambda **kw: kw, ResultMessage=Result
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-use")
    s = configured.model_copy(
        update={
            "claude_enabled": True,
            "claude_model": "claude-sonnet-5",
            "claude_code_oauth_token": "oauth-test",
        }
    )
    with recording_usage(105, "claude") as usage:
        result = await StructuredAgent(
            ClaudeAgentClient(s), PromptLibrary(s.prompts_dir), s
        ).request(ChannelTeaser, system="s", user="u")
    opts = seen[0]["options"]
    assert (
        opts["env"]["ANTHROPIC_API_KEY"] == ""
        and opts["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-test"
    )
    assert opts["tools"] == [] and opts["setting_sources"] == [] and opts["mcp_servers"] == {}
    assert not opts.get("fallback_model")
    assert result.teaser == "Claude audit" and usage.claude_requests == 1
    assert (
        usage.events[0]["reported_cost"] is None and usage.events[0]["sdk_estimated_cost"] == 0.002
    )


async def test_claude_missing_token_never_uses_api_key(configured, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-use")
    with pytest.raises(LLMError, match="no API key fallback"):
        await ClaudeAgentClient(
            configured.model_copy(
                update={"claude_enabled": True, "claude_model": "claude-sonnet-5"}
            )
        ).chat_json(system="s", user="u")


async def test_premium_writer_is_explicit_and_guarded(configured):
    s = configured.model_copy(
        update={"writing_provider": "escalation", "openrouter_allow_escalation": False}
    )
    client = router(s, [info("primary/model"), info("quality/model")], lambda r: response())
    try:
        with pytest.raises(LLMError, match="escalation is disabled"):
            await client.chat_json(system="s", user="u", label="threads_editor#1")
    finally:
        await client.aclose()
