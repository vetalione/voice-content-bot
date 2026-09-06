"""Agent behaviour: windowing, atom dedupe, prompts, structured-output repair."""

from __future__ import annotations

import pytest

from app.agents.base import AgentError, StructuredAgent, json_schema_for
from app.agents.content_miner import (
    ContentMinerAgent,
    build_windows,
    deduplicate_atoms,
)
from app.agents.reels_editor import rank_for_reels
from app.agents.threads_editor import rank_for_threads
from app.models.atoms import ContentAtom, ContentAtomSet
from app.models.transcript import Transcript, TranscriptSegment
from app.services.prompts import PromptLibrary, PromptNotFoundError
from tests.conftest import FakeLLM


def atom(**kwargs) -> ContentAtom:
    data = {
        "id": "a1",
        "label": "l",
        "start_seconds": 0,
        "end_seconds": 60,
        "description": "d",
        "key_claim": "k",
        "supporting_context": "s",
        "categories": ["business_insight"],
        "business_score": 5,
        "personal_score": 5,
        "novelty_score": 5,
        "confidence": 0.9,
    }
    data.update(kwargs)
    return ContentAtom.model_validate(data)


# ------------------------------------------------------------------- windows ---
def test_short_transcript_is_a_single_window():
    transcript = Transcript(
        segments=[TranscriptSegment(start=0, end=100, text="одна мысль")],
        duration=100,
    )
    windows = build_windows(transcript, window_seconds=1200, overlap_seconds=60)
    assert len(windows) == 1
    assert windows[0].segments


def test_long_transcript_is_split_into_overlapping_windows():
    segments = [
        TranscriptSegment(start=index * 60, end=index * 60 + 59, text=f"мысль {index}")
        for index in range(60)  # 60 minutes
    ]
    transcript = Transcript(segments=segments, duration=3600)
    windows = build_windows(transcript, window_seconds=1200, overlap_seconds=60)

    assert len(windows) >= 3
    assert windows[0].start == 0.0
    assert windows[-1].end >= 3500
    # Windows carry absolute timestamps into the prompt.
    assert "[20:00]" in windows[1].text or "[19:00]" in windows[1].text


def test_empty_transcript_yields_no_windows():
    assert build_windows(Transcript(), 1200, 60) == []


# ------------------------------------------------------------- atom de-dupe ---
def test_duplicate_atoms_from_overlapping_windows_are_merged():
    kept = deduplicate_atoms(
        [
            atom(
                id="w1a1",
                label="Религиозные сообщества вокруг AI",
                key_claim="вокруг AI появятся религиозные сообщества",
                description="прогноз",
                start_seconds=1150,
                end_seconds=1250,
                novelty_score=9,
            ),
            atom(
                id="w2a1",
                label="Религиозные сообщества вокруг AI",
                key_claim="вокруг AI появятся религиозные сообщества",
                description="прогноз",
                start_seconds=1160,
                end_seconds=1260,
                novelty_score=6,
            ),
        ]
    )
    assert len(kept) == 1
    assert kept[0].novelty_score == 9, "the stronger duplicate survives"


def test_distinct_atoms_are_all_kept():
    kept = deduplicate_atoms(
        [
            atom(id="a1", label="AI религии", key_claim="про AI", start_seconds=0, end_seconds=60),
            atom(
                id="a2",
                label="Аэропорт",
                key_claim="почти опоздал на самолёт",
                description="история",
                start_seconds=30,
                end_seconds=90,
            ),
        ]
    )
    assert len(kept) == 2


def test_atoms_at_different_times_are_never_merged():
    kept = deduplicate_atoms(
        [
            atom(id="a1", start_seconds=0, end_seconds=60),
            atom(id="a2", start_seconds=600, end_seconds=660),
        ]
    )
    assert len(kept) == 2


# ------------------------------------------------------------------ ranking ---
def test_threads_ranking_prefers_professional_atoms():
    business = atom(id="b", categories=["business_insight"], business_score=9, personal_score=1)
    personal = atom(id="p", categories=["personal_story"], business_score=1, personal_score=9)
    assert rank_for_threads([personal, business])[0].id == "b"


def test_reels_ranking_prefers_story_atoms():
    story = atom(
        id="s",
        categories=["travel_incident", "funny_incident"],
        personal_score=9,
        novelty_score=8,
    )
    dry = atom(id="d", categories=["framework"], personal_score=1, novelty_score=4)
    assert rank_for_reels([dry, story])[0].id == "s"


# ------------------------------------------------------------------ prompts ---
def test_all_prompt_files_exist_and_are_non_empty(settings):
    library = PromptLibrary(settings.prompts_dir)
    for name in ("content_miner", "channel_teaser", "threads_editor", "reels_editor"):
        assert len(library.load(name)) > 200, name
    assert "Tone" in library.voice_style()


def test_missing_prompt_raises():
    library = PromptLibrary(settings_dir := __import__("pathlib").Path("/nonexistent"))
    assert settings_dir
    with pytest.raises(PromptNotFoundError):
        library.load("nope")


def test_placeholders_are_filled_and_unknown_ones_survive():
    assert PromptLibrary.fill("a {{x}} b {{y}}", x="1") == "a 1 b {{y}}"


def test_agents_inject_the_voice_style_into_the_system_prompt(settings):
    library = PromptLibrary(settings.prompts_dir)
    agent = ContentMinerAgent(FakeLLM(), library, settings)  # type: ignore[arg-type]
    rendered = agent.system_prompt()
    assert "{{voice_style}}" not in rendered
    assert "VOICE STYLE" in rendered


# -------------------------------------------------------- structured output ---
def test_json_schema_is_generated_without_a_root_schema_key():
    schema = json_schema_for(ContentAtomSet)
    assert "$schema" not in schema
    assert schema["type"] == "object"


async def test_invalid_payload_is_repaired_on_the_second_attempt(settings):
    class Flaky(FakeLLM):
        def __init__(self) -> None:
            super().__init__(responses={})
            self.n = 0

        async def chat_json(self, **kwargs):
            self.n += 1
            self.calls.append(kwargs)
            if self.n == 1:
                return {"atoms": [{"id": "a1"}]}  # missing required fields
            return {
                "atoms": [
                    {
                        "id": "a1",
                        "label": "ok",
                        "start_seconds": 0,
                        "end_seconds": 10,
                        "confidence": 0.9,
                    }
                ]
            }

    llm = Flaky()

    class Dummy(StructuredAgent):
        name = "dummy"
        prompt_file = "content_miner.md"

    agent = Dummy(llm, PromptLibrary(settings.prompts_dir), settings)  # type: ignore[arg-type]
    result = await agent.request(ContentAtomSet, system="s", user="u")

    assert llm.n == 2
    assert result.atoms[0].label == "ok"
    assert "did not validate" in llm.calls[1]["user"]


async def test_repeatedly_invalid_payload_raises_agent_error(settings):
    class Broken(FakeLLM):
        async def chat_json(self, **kwargs):
            self.calls.append(kwargs)
            return {"atoms": [{"nope": 1}]}

    agent_cls = type("Dummy2", (StructuredAgent,), {"name": "d", "prompt_file": "content_miner.md"})
    agent = agent_cls(Broken(), PromptLibrary(settings.prompts_dir), settings)
    with pytest.raises(AgentError):
        await agent.request(ContentAtomSet, system="s", user="u")


async def test_miner_renumbers_atoms_and_clamps_timestamps(settings, transcript):
    llm = FakeLLM()
    # The model claims a timestamp beyond the window it was shown.
    llm.responses["content_miner"]["atoms"][0]["end_seconds"] = 99999
    miner = ContentMinerAgent(llm, PromptLibrary(settings.prompts_dir), settings)  # type: ignore[arg-type]

    result = await miner.mine(transcript)

    assert [item.id for item in result.atoms] == ["a1", "a2"]
    assert all(item.end_seconds <= transcript.duration for item in result.atoms)


async def test_miner_returns_nothing_for_an_empty_transcript(settings):
    miner = ContentMinerAgent(FakeLLM(), PromptLibrary(settings.prompts_dir), settings)  # type: ignore[arg-type]
    result = await miner.mine(Transcript())
    assert result.atoms == []
