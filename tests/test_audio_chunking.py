"""Chunk planning arithmetic (no ffmpeg execution)."""

from __future__ import annotations

from itertools import pairwise

from app.services.audio import MIN_CHUNK_SECONDS, AudioProcessor, resolve_ffmpeg


def test_chunk_plan_covers_the_whole_recording(settings):
    processor = AudioProcessor(settings)
    duration = 3600.0
    plan = processor.plan_chunks(duration, bytes_per_second=4000.0)

    assert plan[0][0] == 0.0
    last_offset, last_length, _ = plan[-1]
    assert last_offset + last_length >= duration - 1.0


def test_chunks_overlap_by_the_configured_amount(settings):
    settings = settings.model_copy(
        update={"audio_chunk_minutes": 10.0, "audio_chunk_overlap_seconds": 5.0}
    )
    processor = AudioProcessor(settings)
    plan = processor.plan_chunks(3600.0, bytes_per_second=100.0)

    assert plan[0][2] == 0.0, "the first chunk has no overlap"
    assert all(overlap == 5.0 for _, _, overlap in plan[1:])
    # Each chunk starts `overlap` seconds before the previous one ended.
    for (offset_a, length_a, _), (offset_b, _, overlap_b) in pairwise(plan):
        assert abs((offset_a + length_a) - (offset_b + overlap_b)) < 1e-6


def test_chunk_length_is_capped_by_the_upload_limit(settings):
    settings = settings.model_copy(update={"max_stt_upload_mb": 5.0, "audio_chunk_minutes": 60.0})
    processor = AudioProcessor(settings)
    bytes_per_second = 4000.0  # 32 kbps
    plan = processor.plan_chunks(3600.0, bytes_per_second)

    limit_bytes = settings.max_stt_upload_bytes
    for _, length, _ in plan:
        assert length * bytes_per_second <= limit_bytes
    assert len(plan) > 1


def test_short_recording_is_a_single_chunk(settings):
    processor = AudioProcessor(settings)
    plan = processor.plan_chunks(300.0, bytes_per_second=4000.0)
    assert len(plan) == 1
    assert plan[0] == (0.0, 300.0, 0.0)


def test_zero_duration_still_returns_one_chunk(settings):
    processor = AudioProcessor(settings)
    plan = processor.plan_chunks(0.0, bytes_per_second=0.0)
    assert len(plan) == 1


def test_chunk_length_never_drops_below_the_floor(settings, caplog):
    """An absurd byte budget must warn, not shred speech into 2-second pieces."""
    settings = settings.model_copy(update={"max_stt_upload_mb": 0.01})
    processor = AudioProcessor(settings)
    plan = processor.plan_chunks(600.0, bytes_per_second=4000.0)

    assert all(length >= MIN_CHUNK_SECONDS for _, length, _ in plan[:-1])
    assert any("Byte budget allows only" in record.message for record in caplog.records)


def test_ffmpeg_resolution_prefers_the_explicit_override():
    assert resolve_ffmpeg("/opt/custom/ffmpeg") == "/opt/custom/ffmpeg"


def test_ffmpeg_is_resolvable_in_this_environment():
    """PATH or the bundled imageio-ffmpeg binary must provide ffmpeg."""
    assert resolve_ffmpeg("") is not None
