"""Long Telegram output splitting."""

from __future__ import annotations

from app.telegram.formatting import (
    TELEGRAM_MESSAGE_LIMIT,
    escape,
    open_tag_stack,
    split_html_message,
)


def test_short_text_is_not_split():
    assert split_html_message("короткий отчёт") == ["короткий отчёт"]


def test_empty_text_returns_no_messages():
    assert split_html_message("") == []
    assert split_html_message("   ") == []


def test_every_part_respects_the_limit():
    text = "\n\n".join(f"Блок номер {index}. " + "слово " * 120 for index in range(40))
    parts = split_html_message(text)
    assert len(parts) > 1
    assert all(len(part) <= TELEGRAM_MESSAGE_LIMIT for part in parts)


def test_split_prefers_blank_line_boundaries():
    blocks = [f"<b>Блок {index}</b>\nтело блока " + "x" * 900 for index in range(10)]
    parts = split_html_message("\n\n".join(blocks))
    assert len(parts) > 1
    for part in parts:
        assert part.startswith("<b>") or part.startswith("Блок") or "<b>" in part


def test_open_tags_are_closed_and_reopened_across_the_seam():
    long_body = "текст " * 1500
    text = f"<b>заголовок {long_body}</b>"
    parts = split_html_message(text)
    assert len(parts) > 1
    for part in parts:
        assert part.count("<b>") == part.count("</b>"), part[:80]
    assert parts[0].endswith("</b>")
    assert parts[1].startswith("<b>")


def test_nested_tags_are_balanced_in_every_part():
    body = "<b><i>" + ("длинная мысль " * 800) + "</i></b>"
    parts = split_html_message(body)
    assert len(parts) > 1
    for part in parts:
        assert part.count("<b>") == part.count("</b>")
        assert part.count("<i>") == part.count("</i>")


def test_no_part_ends_inside_a_tag():
    text = "<code>" + ("abcdefgh " * 900) + "</code>"
    for part in split_html_message(text):
        assert part.count("<") == part.count(">")
        assert not part.rstrip().endswith("<")


def test_single_unbroken_run_is_hard_split():
    text = "я" * 12000
    parts = split_html_message(text)
    assert len(parts) >= 3
    assert all(len(part) <= TELEGRAM_MESSAGE_LIMIT for part in parts)
    assert "".join(parts) == text


def test_content_is_preserved_across_the_split():
    marker_count = 60
    text = "\n\n".join(f"МАРКЕР{index} " + "слово " * 100 for index in range(marker_count))
    joined = " ".join(split_html_message(text))
    for index in range(marker_count):
        assert f"МАРКЕР{index}" in joined


def test_open_tag_stack_tracks_unclosed_tags():
    assert open_tag_stack("<b>жирный") == [("b", "<b>")]
    assert open_tag_stack("<b>жирный</b>") == []
    assert [name for name, _ in open_tag_stack("<b><i>оба")] == ["b", "i"]
    assert open_tag_stack('<a href="https://x.dev">ссылка') == [("a", '<a href="https://x.dev">')]


def test_escape_protects_html_special_chars():
    assert escape("5 < 10 & 20 > 3") == "5 &lt; 10 &amp; 20 &gt; 3"
    assert escape(None) == ""


def test_custom_limit_is_honoured():
    text = "\n\n".join(f"строка {index} " + "y" * 200 for index in range(20))
    parts = split_html_message(text, limit=600)
    assert all(len(part) <= 600 for part in parts)
    assert len(parts) > 4
