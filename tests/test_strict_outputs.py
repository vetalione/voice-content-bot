"""Strict schemas and explicit-only compatibility downgrade."""

import json

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from app.agents.base import StructuredAgent, json_schema_for
from app.models.atoms import ContentAtomSet
from app.models.content import ChannelTeaser, ReelsBatch, ThreadsBatch
from app.services.groq_client import GroqClient, GroqError, GroqSchemaError
from app.services.prompts import PromptLibrary
from tests.conftest import default_llm_responses

STAGES = [
    (ContentAtomSet, "content_miner"),
    (ChannelTeaser, "channel_teaser"),
    (ThreadsBatch, "threads_editor"),
    (ReelsBatch, "reels_editor"),
]


def check_closed(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for value in node.values():
            check_closed(value)
    elif isinstance(node, list):
        for value in node:
            check_closed(value)


@pytest.mark.parametrize("model,stage", STAGES)
async def test_all_stages_send_strict_and_validate(settings, model, stage):
    schema = json_schema_for(model)
    check_closed(schema)
    # Serializing a validated fixture fills every defaulted field, as strict output must.
    result = model.model_validate(default_llm_responses()[stage]).model_dump(mode="json")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"] == schema
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    settings = settings.model_copy(update={"groq_use_json_schema": False})
    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        agent = StructuredAgent(
            GroqClient(settings, client=http), PromptLibrary(settings.prompts_dir), settings
        )
        actual = await agent.request(model, system="s", user="u", max_tokens=1200)
    assert isinstance(actual, model)
    assert actual.model_dump(mode="json") == result
    assert len(seen) == 1


def test_nullable_nested_fields_and_property_names_preserved():
    class Child(BaseModel):
        title: str | None = None
        description: str = ""

    class Envelope(BaseModel):
        child: Child | None = None
        children: list[Child] = []

    schema = json_schema_for(Envelope)
    check_closed(schema)
    assert {"type": "null"} in schema["properties"]["child"]["anyOf"]
    assert {"type": "null"} in schema["$defs"]["Child"]["properties"]["title"]["anyOf"]
    assert "description" in schema["$defs"]["Child"]["required"]
    assert (
        Envelope.model_validate(
            {"child": None, "children": [{"title": None, "description": ""}]}
        ).child
        is None
    )


@pytest.mark.parametrize(
    "status,detail",
    [
        (400, {"message": "bad temperature"}),
        (404, {"message": "model not found"}),
        (
            422,
            {"message": "unsupported schema keyword in json_schema", "code": "invalid_json_schema"},
        ),
        (400, {"code": "json_validate_failed", "failed_generation": ""}),
        (429, {"message": "TPM Limit 8000 Requested 8202"}),
        (500, {"message": "internal error"}),
    ],
)
async def test_unrelated_errors_never_downgrade(settings, status, detail):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["response_format"])
        return httpx.Response(status, json={"error": detail})

    settings = settings.model_copy(update={"groq_max_retries": 0})
    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(GroqError):
            await GroqClient(settings, client=http).chat_json(
                system="s", user="u", schema=json_schema_for(ChannelTeaser), max_tokens=400
            )
    assert len(seen) == 1
    assert seen[0]["type"] == "json_schema"
    assert seen[0]["json_schema"]["strict"] is True


async def test_invalid_schema_is_configuration_error(settings):
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "invalid_json_schema",
                    "message": "json_schema is not supported by this model due to bad schema",
                }
            },
        )

    async with httpx.AsyncClient(
        base_url=settings.groq_base_url, transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(GroqSchemaError):
            await GroqClient(settings, client=http).chat_json(
                system="s", user="u", schema=json_schema_for(ChannelTeaser), max_tokens=400
            )


def test_pydantic_bounds_remain_enforced():
    schema = json_schema_for(ChannelTeaser)
    assert "maxLength" not in schema["properties"]["teaser"]
    with pytest.raises(ValidationError):
        ChannelTeaser.model_validate({"teaser": "x" * 1501, "timestamps": [], "reasoning": ""})
