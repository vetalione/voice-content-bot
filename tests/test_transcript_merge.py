"""Chunk timestamp offsets and overlap de-duplication."""

from __future__ import annotations

from app.models.transcript import ChunkTranscript, TranscriptSegment
from app.services.transcript_merge import merge_chunk_transcripts


def chunk(index, offset, overlap, duration, pairs) -> ChunkTranscript:
    return ChunkTranscript(
        index=index,
        offset_seconds=offset,
        overlap_seconds=overlap,
        duration=duration,
        language="ru",
        segments=[TranscriptSegment(start=start, end=end, text=text) for start, end, text in pairs],
    )


def test_single_chunk_is_passed_through():
    merged = merge_chunk_transcripts(
        [chunk(0, 0, 0, 60, [(0, 10, "первая мысль"), (10, 20, "вторая мысль")])]
    )
    assert [seg.text for seg in merged.segments] == ["первая мысль", "вторая мысль"]
    assert merged.language == "ru"


def test_chunk_timestamps_are_offset_to_the_full_recording():
    """A segment at 0-10s of chunk 2 must land at 600-610s of the recording."""
    merged = merge_chunk_transcripts(
        [
            chunk(0, 0, 0, 600, [(0, 10, "начало"), (500, 590, "конец первого чанка")]),
            chunk(1, 600, 5, 600, [(0, 10, "начало второго"), (100, 120, "дальше")]),
        ]
    )
    starts = [seg.start for seg in merged.segments]
    ends = [seg.end for seg in merged.segments]
    assert starts == [0.0, 500.0, 600.0, 700.0]
    assert ends == [10.0, 590.0, 610.0, 720.0]
    assert merged.duration >= 1200.0


def test_offsets_survive_out_of_order_input():
    merged = merge_chunk_transcripts(
        [
            chunk(1, 300, 5, 300, [(0, 10, "второй чанк")]),
            chunk(0, 0, 0, 300, [(0, 10, "первый чанк")]),
        ]
    )
    assert [seg.text for seg in merged.segments] == ["первый чанк", "второй чанк"]
    assert [seg.start for seg in merged.segments] == [0.0, 300.0]


def test_overlap_segment_repeated_verbatim_is_dropped():
    """The tail of chunk 1 is re-transcribed at the head of chunk 2."""
    tail = "и вот именно поэтому я решил всё поменять"
    merged = merge_chunk_transcripts(
        [
            chunk(0, 0, 0, 300, [(0, 290, "долгий монолог"), (290, 300, tail)]),
            # chunk 2 starts 10s early, so its first segment repeats `tail`
            chunk(1, 290, 10, 300, [(0, 10, tail), (10, 30, "новая мысль")]),
        ]
    )
    texts = [seg.text for seg in merged.segments]
    assert texts.count(tail) == 1
    assert "новая мысль" in texts


def test_partial_word_overlap_is_trimmed_at_the_seam():
    """Whisper split the seam differently, so trim the duplicated word run."""
    merged = merge_chunk_transcripts(
        [
            chunk(0, 0, 0, 300, [(280, 300, "а потом я понял что деньги это не цель")]),
            chunk(
                1,
                290,
                10,
                300,
                [(0, 12, "деньги это не цель а следствие внимания")],
            ),
        ]
    )
    joined = merged.text
    assert joined.count("деньги это не цель") == 1
    assert "а следствие внимания" in joined


def test_timestamps_never_go_backwards():
    merged = merge_chunk_transcripts(
        [
            chunk(0, 0, 0, 300, [(0, 300, "первый")]),
            chunk(1, 290, 10, 300, [(0, 20, "второй совершенно другой текст")]),
        ]
    )
    starts = [seg.start for seg in merged.segments]
    assert starts == sorted(starts)


def test_empty_chunks_produce_empty_transcript():
    merged = merge_chunk_transcripts([])
    assert merged.segments == []
    assert merged.text == ""


def test_three_chunks_accumulate_offsets():
    merged = merge_chunk_transcripts(
        [
            chunk(0, 0, 0, 600, [(0, 5, "раз")]),
            chunk(1, 595, 5, 600, [(0, 5, "два")]),
            chunk(2, 1190, 5, 600, [(0, 5, "три")]),
        ]
    )
    assert [round(seg.start) for seg in merged.segments] == [0, 595, 1190]
    assert [seg.text for seg in merged.segments] == ["раз", "два", "три"]
