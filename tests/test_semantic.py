"""Semantic orchestration regressions use synthetic sources and mocked providers.

These prove preservation/routing contracts, not empirical model recall quality.
"""

import json

import pytest

from app.agents.rendering import render_atoms
from app.agents.semantic_miner import (
    SemanticMinerAgent,
    apply_merge,
    choose_auditor,
    editorial_atoms,
    escalation_reasons,
)
from app.models.semantic import (
    ContentRoute,
    CoverageDiagnostics,
    CoveragePatch,
    DuplicateLink,
    MergePatch,
    SemanticAtom,
    SemanticExtraction,
)
from app.models.transcript import Transcript, TranscriptSegment
from app.services.checkpoints import SQLiteStore, checkpoint_scope
from app.services.llm import LLMError
from app.services.prompts import PromptLibrary


@pytest.fixture
def semantic_settings(settings):
    return settings.model_copy(
        update={
            "semantic_pipeline_enabled": True,
            "text_provider": "openrouter",
            "quality_auditor": "none",
            "openrouter_primary_model": "primary/test",
            "openrouter_allow_paid": True,
        }
    )


def atom(
    id="taste",
    claim="ИИ-агенты выработают собственный эстетический вкус.",
    kind="primary",
    start=0,
    end=30,
    **kwargs,
):
    return SemanticAtom(
        id=id,
        title=claim[:100],
        claim=claim,
        kind=kind,
        sources=[{"start_seconds": start, "end_seconds": end}],
        **kwargs,
    )


def art_transcript(duration=105):
    return Transcript(
        duration=duration,
        segments=[
            TranscriptSegment(
                start=0, end=30, text="ИИ-агенты выработают собственный эстетический вкус."
            ),
            TranscriptSegment(
                start=30,
                end=60,
                text="Художники смогут создавать искусство специально для агентов.",
            ),
            TranscriptSegment(
                start=60,
                end=105,
                text="Богатые люди станут покупать такое искусство: предпочтение агентов станет сигналом статуса и стоимости.",
            ),
        ],
    )


def art_atoms():
    return [
        atom(),
        atom(
            "artists",
            "Художники будут создавать искусство для вкуса агентов.",
            "supporting",
            30,
            60,
            supports_atom_ids=["taste"],
        ),
        atom(
            "status",
            "Предпочтение агента становится сигналом статуса и стоимости для богатых коллекционеров.",
            "implication",
            60,
            105,
            derived_from_atom_ids=["taste", "artists"],
        ),
    ]


class SemanticFake:
    cache_identity = "semantic-fake"

    def __init__(self, extracted=None, audit=None, merge=None):
        self.extracted = extracted if extracted is not None else [atom()]
        self.audit = audit or CoveragePatch(
            diagnostics=CoverageDiagnostics(coverage_confidence=0.9)
        )
        self.merge = merge or MergePatch()
        self.calls = []
        self.roles = []
        self.extract_pages = []

    def audit_view(self, role):
        self.roles.append(role)
        return self

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["schema_name"]
        if name == "SemanticExtraction":
            if self.extract_pages:
                return self.extract_pages.pop(0).model_dump(mode="json")
            return SemanticExtraction(
                atoms=self.extracted, diagnostics=CoverageDiagnostics(coverage_confidence=0.9)
            ).model_dump(mode="json")
        if name == "MergePatch":
            return self.merge.model_dump(mode="json")
        if name == "CoveragePatch":
            return self.audit.model_dump(mode="json")
        if name == "RoutingResult":
            return {
                "routes": [
                    {"atom_id": a["id"], "route": "ARCHIVE_ONLY", "reason": "Keep all thoughts"}
                    for a in json.loads(kwargs["user"])
                ]
            }
        raise AssertionError(name)


def miner(settings, fake):
    return SemanticMinerAgent(fake, PromptLibrary(settings.prompts_dir), settings)


async def test_short_simple_recording_stays_one_atom(semantic_settings):
    fake = SemanticFake()
    result = await miner(semantic_settings, fake).mine(art_transcript())
    assert len(result.atoms) == 1
    assert len(fake.calls) == 4  # extraction, global merge, primary audit, routing


async def test_multiple_atoms_same_topic_preserve_support_and_implication(semantic_settings):
    result = await miner(semantic_settings, SemanticFake(art_atoms())).mine(art_transcript())
    assert len(result.atoms) == 3
    assert [a.kind for a in result.atoms] == ["primary", "supporting", "implication"]
    assert result.atoms[1].supports_atom_ids == [result.atoms[0].id]
    assert result.atoms[2].derived_from_atom_ids == [result.atoms[0].id, result.atoms[1].id]
    assert "статуса" in result.atoms[2].claim


def test_dedupe_does_not_collapse_semantic_roles():
    atoms = [atom(), atom("support", kind="supporting"), atom("implication", kind="implication")]
    patch = MergePatch(
        duplicates=[
            DuplicateLink(duplicate_id="support", keep_id="taste", reason="same topic"),
            DuplicateLink(duplicate_id="implication", keep_id="taste", reason="same topic"),
        ]
    )
    assert len(apply_merge(atoms, patch)) == 3


def test_actual_duplicates_merge_sources_and_remap_relations():
    atoms = [
        atom(),
        atom("repeat", start=80, end=90),
        atom("effect", "Следствие", "implication", derived_from_atom_ids=["repeat"]),
    ]
    result = apply_merge(
        atoms,
        MergePatch(
            duplicates=[
                DuplicateLink(duplicate_id="repeat", keep_id="taste", reason="identical claim")
            ]
        ),
    )
    assert len(result) == 2 and len(result[0].sources) == 2
    assert result[1].derived_from_atom_ids == ["taste"]


async def test_audit_recovers_economic_implication(semantic_settings):
    fake = SemanticFake(
        art_atoms()[:2],
        audit=CoveragePatch(
            omissions=[
                atom(
                    "missing",
                    "Вкус агента стал сигналом статуса для коллекционеров.",
                    "implication",
                    60,
                    105,
                    derived_from_atom_ids=["w1p1.taste"],
                )
            ],
            diagnostics=CoverageDiagnostics(coverage_confidence=0.95),
        ),
    )
    result = await miner(semantic_settings, fake).mine(art_transcript())
    assert len(result.atoms) == 3 and result.auditor_added_atom_count == 1
    assert result.atoms[-1].derived_from_atom_ids == ["w1p1.taste"]
    assert "60.0-105.0" in next(
        c["user"] for c in fake.calls if c["schema_name"] == "CoveragePatch"
    )


async def test_distant_stories_get_separate_linked_synthesis(semantic_settings):
    s = semantic_settings.model_copy(update={"extraction_window_minutes": 65})
    stories = [
        atom("dog", "История с собакой", "story", 0, 100),
        atom("film", "История с фильмом", "story", 1000, 1100),
        atom("job", "История с новой работой", "story", 2000, 2100),
    ]
    synthesis = atom(
        "law",
        "Все три случая показывают: сначала проверь ожидания, затем делай вывод.",
        "synthesis",
        3000,
        3100,
        derived_from_atom_ids=["w1p1.dog", "w1p1.film", "w1p1.job"],
    )
    synthesis.sources += [r for a in stories for r in a.sources]
    transcript = Transcript(
        duration=3600,
        segments=[
            TranscriptSegment(
                start=a.sources[0].start_seconds, end=a.sources[0].end_seconds, text=a.claim
            )
            for a in [*stories, synthesis]
        ],
    )
    result = await miner(
        s, SemanticFake(stories, merge=MergePatch(synthesis_atoms=[synthesis]))
    ).mine(transcript)
    assert len(result.atoms) == 4
    assert result.atoms[-1].kind == "synthesis" and len(result.atoms[-1].derived_from_atom_ids) == 3
    assert len(result.atoms[-1].sources) == 4


async def test_overflow_continues_and_preserves_atom_eleven(semantic_settings):
    fake = SemanticFake()
    fake.extract_pages = [
        SemanticExtraction(
            atoms=[atom(f"a{i}", f"Отдельная мысль {i}") for i in range(10)],
            diagnostics=CoverageDiagnostics(overflow=True, coverage_confidence=0.6),
            overflow_hints=["мысль 11"],
        ),
        SemanticExtraction(
            atoms=[atom("eleven", "Экономическое следствие 11", "implication")],
            diagnostics=CoverageDiagnostics(coverage_confidence=0.9),
        ),
    ]
    result = await miner(semantic_settings, fake).mine(art_transcript())
    assert len(result.atoms) == 11
    requests = [c for c in fake.calls if c["schema_name"] == "SemanticExtraction"]
    assert len(requests) == 2 and "мысль 11" in requests[1]["user"]


async def test_overflow_exhaustion_visible_not_silent(semantic_settings):
    fake = SemanticFake()
    fake.extract_pages = [
        SemanticExtraction(
            atoms=[atom(str(i), f"Мысль {i}")],
            diagnostics=CoverageDiagnostics(overflow=True),
            overflow_hints=["ещё мысль"],
        )
        for i in range(2)
    ]
    result = await miner(semantic_settings, fake).mine(art_transcript())
    assert result.diagnostics.overflow and result.warnings


def test_low_coverage_needs_evidence_not_just_small_count(semantic_settings):
    simple = art_transcript()
    diag = CoverageDiagnostics(coverage_confidence=0.95)
    assert not escalation_reasons(semantic_settings, simple, [atom()], diag)
    longtext = Transcript(
        duration=1200, segments=[TranscriptSegment(start=0, end=1200, text="мысль " * 2000)]
    )
    diag = CoverageDiagnostics(
        coverage_confidence=0.5, topic_shift_count=7, semantic_density="dense", broad_atoms=True
    )
    reasons = escalation_reasons(semantic_settings, longtext, [atom()], diag)
    assert "suspiciously_low_coverage" in reasons and "broad_atoms_dense_source" in reasons


@pytest.mark.parametrize(
    "mode,expected",
    [("none", None), ("kimi_k3", "escalation"), ("claude", "claude"), ("auto", "escalation")],
)
async def test_quality_selection_is_one_audit_not_both(semantic_settings, mode, expected):
    s = semantic_settings.model_copy(
        update={
            "quality_auditor": mode,
            "claude_enabled": True,
            "openrouter_allow_escalation": True,
        }
    )
    fake = SemanticFake()
    await miner(s, fake).mine(art_transcript(1900))
    assert fake.roles == ([expected] if expected else [])
    assert len([c for c in fake.calls if c["label"].startswith("coverage_audit/")]) == (
        2 if expected else 1
    )


def test_manual_escalation_requires_explicit_permission(semantic_settings):
    with pytest.raises(LLMError):
        choose_auditor(semantic_settings.model_copy(update={"quality_auditor": "kimi_k3"}), [])


async def test_completed_semantic_requests_reused_after_restart(semantic_settings, tmp_path):
    path = tmp_path / "checkpoints.db"
    store = SQLiteStore(path)
    fake = SemanticFake(art_atoms())
    with checkpoint_scope(store, "recording"):
        original = await miner(semantic_settings, fake).mine(art_transcript())
    newstore = SQLiteStore(path)
    newfake = SemanticFake(art_atoms())
    with checkpoint_scope(newstore, "recording"):
        replay = await miner(semantic_settings, newfake).mine(art_transcript())
    assert not newfake.calls and replay == original


async def test_archive_atoms_preserved_and_support_passed_to_writer(semantic_settings):
    result = await miner(semantic_settings, SemanticFake(art_atoms())).mine(art_transcript())
    assert all(r.route == ContentRoute.ARCHIVE_ONLY for r in result.routes)
    editorial = editorial_atoms(result, art_transcript())
    assert len(editorial) == 3
    text = render_atoms([editorial[0]], related_atoms=editorial)
    assert "статуса" in text and "SUPPORT / IMPLICATIONS" in text


async def test_hour_has_six_large_extraction_windows(semantic_settings):
    fake = SemanticFake(extracted=[])
    transcript = Transcript(
        duration=3600,
        segments=[
            TranscriptSegment(start=i * 60, end=(i + 1) * 60, text=f"Текст {i}") for i in range(60)
        ],
    )
    await miner(semantic_settings, fake).mine(transcript)
    assert len([c for c in fake.calls if c["schema_name"] == "SemanticExtraction"]) == 6
    assert len(fake.calls) == 9  # 6 extraction + global merge/audit/routing; writers separate


async def test_global_count_over_soft_target_is_visible(semantic_settings):
    s = semantic_settings.model_copy(update={"target_atom_limit": 2})
    result = await miner(s, SemanticFake(art_atoms())).mine(art_transcript())
    assert len(result.atoms) == 3 and result.diagnostics.overflow
    assert "overflow" in result.escalation_reasons
