"""Schema contracts for content atoms and editor output."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.atoms import AtomCategory, ContentAtom, ContentAtomSet
from app.models.content import ChannelTeaser, ReelsCandidate, TeaserTimestamp, ThreadsBatch


def minimal_atom(**overrides) -> dict:
    data = {
        "id": "a1",
        "label": "Тестовый атом",
        "start_seconds": 30,
        "end_seconds": 90,
        "description": "описание",
        "key_claim": "утверждение",
        "supporting_context": "цитата из записи",
        "categories": ["business_insight"],
        "business_score": 8,
        "personal_score": 2,
        "novelty_score": 7,
        "confidence": 0.9,
        "should_ignore": False,
    }
    data.update(overrides)
    return data


def test_atom_parses_and_exposes_a_timecode():
    atom = ContentAtom.model_validate(minimal_atom())
    assert atom.timecode == "00:30–01:30"
    assert atom.duration == 60
    assert atom.is_professional is True
    assert atom.usable(min_confidence=0.5) is True


def test_atom_rejects_out_of_range_scores():
    with pytest.raises(ValidationError):
        ContentAtom.model_validate(minimal_atom(business_score=42))
    with pytest.raises(ValidationError):
        ContentAtom.model_validate(minimal_atom(novelty_score=-1))


def test_atom_requires_id_and_label():
    with pytest.raises(ValidationError):
        ContentAtom.model_validate(minimal_atom(id=""))
    with pytest.raises(ValidationError):
        ContentAtom.model_validate(minimal_atom(label=""))


def test_unknown_category_falls_back_to_other():
    atom = ContentAtom.model_validate(
        minimal_atom(categories=["Business Insight", "hot-take", "prediction"])
    )
    assert atom.categories == [
        AtomCategory.BUSINESS_INSIGHT,
        AtomCategory.OTHER,
        AtomCategory.PREDICTION,
    ]


def test_confidence_is_normalized_from_percent_and_ten_point_scales():
    assert ContentAtom.model_validate(minimal_atom(confidence=85)).confidence == 0.85
    assert ContentAtom.model_validate(minimal_atom(confidence=9)).confidence == 0.9
    assert ContentAtom.model_validate(minimal_atom(confidence=0.4)).confidence == 0.4


def test_inverted_timestamps_are_clamped():
    atom = ContentAtom.model_validate(minimal_atom(start_seconds=100, end_seconds=10))
    assert atom.end_seconds == atom.start_seconds


def test_ignored_and_low_confidence_atoms_are_not_usable():
    ignored = ContentAtom.model_validate(minimal_atom(should_ignore=True))
    unsure = ContentAtom.model_validate(minimal_atom(confidence=0.2))
    assert ignored.usable(0.5) is False
    assert unsure.usable(0.5) is False


def test_atom_set_filters_by_confidence():
    atom_set = ContentAtomSet.model_validate(
        {
            "atoms": [
                minimal_atom(id="a1", confidence=0.9),
                minimal_atom(id="a2", confidence=0.3),
                minimal_atom(id="a3", confidence=0.8, should_ignore=True),
            ]
        }
    )
    assert [atom.id for atom in atom_set.usable(0.5)] == ["a1"]


def test_teaser_render_includes_timestamps_only_when_asked():
    teaser = ChannelTeaser(
        teaser="Две сентенции подряд.",
        timestamps=[TeaserTimestamp(seconds=95, label="про сообщества")],
    )
    assert "01:35 — про сообщества" in teaser.render(include_timestamps=True)
    assert teaser.render(include_timestamps=False) == "Две сентенции подряд."


def test_threads_batch_orders_by_readiness():
    batch = ThreadsBatch.model_validate(
        {
            "candidates": [
                {"atom_id": "a1", "draft": "слабый", "readiness_score": 3},
                {"atom_id": "a2", "draft": "сильный", "readiness_score": 9},
            ]
        }
    )
    assert [item.draft for item in batch.best(2)] == ["сильный", "слабый"]
    assert [item.draft for item in batch.best(1)] == ["сильный"]


def test_reels_candidate_coerces_on_screen_text_from_a_string():
    candidate = ReelsCandidate.model_validate(
        {
            "atom_id": "a1",
            "concept": "Аэропорт",
            "script": "текст",
            "on_screen_text": "минус одна минута",
        }
    )
    assert candidate.on_screen_text == ["минус одна минута"]


def test_reels_candidate_requires_a_script():
    with pytest.raises(ValidationError):
        ReelsCandidate.model_validate({"concept": "Аэропорт", "script": ""})
