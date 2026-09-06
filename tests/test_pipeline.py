"""End-to-end pipeline behaviour with every external dependency faked."""

from __future__ import annotations

import pytest

from app.agents.channel_teaser import ChannelTeaserAgent
from app.agents.content_miner import ContentMinerAgent
from app.agents.reels_editor import ReelsEditorAgent
from app.agents.threads_editor import ThreadsEditorAgent
from app.config import Settings
from app.models.media import SourceMode
from app.models.transcript import TranscriptSegment
from app.services.dedupe import TTLDedupeStore
from app.services.pipeline import ContentPipeline, EmptyTranscriptError
from app.services.processor import RecordingProcessor
from app.services.prompts import PromptLibrary
from tests.conftest import (
    FakeAudioProcessor,
    FakeDelivery,
    FakeDownloader,
    FakeLLM,
    FakeTranscriber,
    make_job_request,
)


def build_pipeline(
    settings: Settings,
    llm: FakeLLM | None = None,
    transcriber: FakeTranscriber | None = None,
    delivery: FakeDelivery | None = None,
    audio: FakeAudioProcessor | None = None,
):
    llm = llm or FakeLLM()
    delivery = delivery or FakeDelivery()
    prompts = PromptLibrary(settings.prompts_dir)
    pipeline = ContentPipeline(
        settings=settings,
        downloader=FakeDownloader(),
        audio=audio or FakeAudioProcessor(),  # type: ignore[arg-type]
        transcriber=transcriber or FakeTranscriber(),
        miner=ContentMinerAgent(llm, prompts, settings),  # type: ignore[arg-type]
        teaser_agent=ChannelTeaserAgent(llm, prompts, settings),  # type: ignore[arg-type]
        threads_agent=ThreadsEditorAgent(llm, prompts, settings),  # type: ignore[arg-type]
        reels_agent=ReelsEditorAgent(llm, prompts, settings),  # type: ignore[arg-type]
        delivery=delivery,  # type: ignore[arg-type]
    )
    return pipeline, llm, delivery


async def test_channel_job_publishes_a_teaser_and_reports_privately(settings):
    pipeline, _, delivery = build_pipeline(settings)
    processor = RecordingProcessor(settings, pipeline, delivery, TTLDedupeStore())  # type: ignore[arg-type]

    result = await processor.process(make_job_request(SourceMode.CHANNEL))

    assert delivery.published is True
    published_text, reply_to = delivery.channel_posts[0]
    assert "AI" in published_text
    assert reply_to == 77, "the teaser replies to the original channel post"
    assert result.metadata.published is True
    assert result.threads and result.reels
    assert delivery.owner_messages, "owner always gets the report"


async def test_private_job_never_publishes_to_the_channel(settings):
    pipeline, _, delivery = build_pipeline(settings)
    processor = RecordingProcessor(settings, pipeline, delivery, TTLDedupeStore())  # type: ignore[arg-type]

    result = await processor.process(make_job_request(SourceMode.PRIVATE))

    assert delivery.channel_posts == [], "private mode must never post publicly"
    assert result.metadata.published is False
    assert delivery.owner_messages, "but the owner still gets the full report"
    assert result.teaser is not None, "teaser is generated for inspection only"


async def test_candidate_timecodes_come_from_the_referenced_atom(settings):
    """A draft must link back to where it was actually said in the recording."""
    pipeline, _, _ = build_pipeline(settings)
    result = await pipeline.run(make_job_request(SourceMode.PRIVATE))

    by_id = {atom.id: atom for atom in result.atoms}
    for candidate in [*result.threads, *result.reels]:
        atom = by_id[candidate.atom_id]
        assert candidate.start_seconds == atom.start_seconds
        assert candidate.end_seconds == atom.end_seconds

    # The AI prediction atom (30-90s) is the one Threads used.
    assert result.threads[0].timecode == "00:30–01:30"
    # The airport story (0-30s) is the one Reels used.
    assert result.reels[0].timecode == "00:00–00:30"


async def test_owner_report_contains_metadata_threads_and_reels(settings):
    pipeline, _, delivery = build_pipeline(settings)
    processor = RecordingProcessor(settings, pipeline, delivery, TTLDedupeStore())  # type: ignore[arg-type]

    await processor.process(make_job_request(SourceMode.PRIVATE))
    report = "\n".join(delivery.owner_messages)

    assert "Запись обработана" in report
    assert "Threads" in report
    assert "Reels" in report
    assert "СЦЕНАРИЙ" in report


async def test_transcript_is_not_sent_by_default(settings):
    pipeline, _, delivery = build_pipeline(settings)
    processor = RecordingProcessor(settings, pipeline, delivery, TTLDedupeStore())  # type: ignore[arg-type]

    await processor.process(make_job_request(SourceMode.PRIVATE))
    assert delivery.owner_documents == []


async def test_transcript_document_is_sent_when_enabled(settings):
    settings = settings.model_copy(update={"enable_full_transcript": True})
    pipeline, _, delivery = build_pipeline(settings)
    processor = RecordingProcessor(settings, pipeline, delivery, TTLDedupeStore())  # type: ignore[arg-type]

    await processor.process(make_job_request(SourceMode.PRIVATE))

    assert len(delivery.owner_documents) == 1
    filename, content, _ = delivery.owner_documents[0]
    assert filename.endswith(".txt")
    assert "самолёт" in content.decode("utf-8")


async def test_dry_run_publish_runs_everything_but_posts_nothing(settings):
    settings = settings.model_copy(update={"dry_run_publish": True})

    class DryDelivery(FakeDelivery):
        async def publish_to_channel(self, text, reply_to_message_id=None):
            return None  # mirrors TelegramDelivery under DRY_RUN_PUBLISH

    delivery = DryDelivery()
    pipeline, _, _ = build_pipeline(settings, delivery=delivery)
    processor = RecordingProcessor(settings, pipeline, delivery, TTLDedupeStore())  # type: ignore[arg-type]

    result = await processor.process(make_job_request(SourceMode.CHANNEL))
    assert result.metadata.published is False


async def test_empty_transcript_raises_instead_of_publishing_nonsense(settings):
    transcriber = FakeTranscriber(segments=[TranscriptSegment(start=0, end=1, text="")])
    pipeline, _, delivery = build_pipeline(settings, transcriber=transcriber)

    with pytest.raises(EmptyTranscriptError):
        await pipeline.run(make_job_request(SourceMode.CHANNEL))
    assert delivery.channel_posts == []


async def test_multi_chunk_recording_offsets_atom_timestamps(settings):
    audio = FakeAudioProcessor(duration=2400.0, chunks=2)
    transcriber = FakeTranscriber()
    pipeline, _, _ = build_pipeline(settings, transcriber=transcriber, audio=audio)

    transcript, chunks = await pipeline.transcribe(
        settings.work_dir / "x.oga", settings.work_dir, reported_duration=2400.0
    )

    assert chunks == 2
    assert len(transcriber.calls) == 2
    assert transcriber.calls[1].offset_seconds == 1200.0
    assert max(seg.end for seg in transcript.segments) > 1200.0


async def test_temp_workspace_is_cleaned_up(settings):
    pipeline, _, _ = build_pipeline(settings)
    await pipeline.run(make_job_request(SourceMode.PRIVATE))

    leftovers = list(settings.work_dir.glob("*")) if settings.work_dir.exists() else []
    assert leftovers == [], f"scratch files left behind: {leftovers}"


async def test_low_confidence_atoms_produce_a_warning(settings):
    llm = FakeLLM()
    llm.responses["content_miner"]["atoms"][0]["confidence"] = 0.1
    llm.responses["content_miner"]["atoms"][1]["confidence"] = 0.1
    pipeline, _, _ = build_pipeline(settings, llm=llm)

    result = await pipeline.run(make_job_request(SourceMode.PRIVATE))
    assert result.warnings
