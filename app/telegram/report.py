"""Renders the private owner report as Telegram HTML.

Deliberately never includes the raw transcript: it is long, it is the least
useful part of the report, and it is not what the owner needs to act. The
transcript goes out as a separate .txt document only when
``ENABLE_FULL_TRANSCRIPT=true``.
"""

from __future__ import annotations

from app.models.content import PipelineResult, ReelsCandidate, ThreadsCandidate
from app.models.media import RecordingMetadata, SourceMode
from app.telegram.formatting import escape
from app.utils.timecode import format_timecode

_MODE_LABELS = {
    SourceMode.CHANNEL: "канал (продакшн)",
    SourceMode.PRIVATE: "приватный тест/архив",
}


def render_metadata(meta: RecordingMetadata) -> str:
    lines = [
        "<b>🎙 Запись обработана</b>",
        f"Режим: {escape(_MODE_LABELS.get(meta.mode, meta.mode.value))}",
        f"Тип: {escape(meta.kind)}" + (" (форвард)" if meta.forwarded else ""),
        f"Длительность: {format_timecode(meta.duration_seconds)}",
        f"Источник: <code>{meta.source_chat_id}</code> / msg <code>{meta.source_message_id}</code>",
    ]
    if meta.chat_title:
        lines.insert(2, f"Канал: {escape(meta.chat_title)}")
    if meta.posted_at:
        lines.append(f"Опубликовано: {meta.posted_at:%Y-%m-%d %H:%M} UTC")
    lines.append(
        f"Транскрипт: {meta.transcript_words} слов / {meta.transcript_chars} симв."
        f" · чанков: {meta.chunks}"
    )
    if meta.detected_language:
        lines.append(f"Язык: {escape(meta.detected_language)}")
    lines.append(f"Атомы: найдено {meta.atoms_found}, использовано {meta.atoms_used}")
    if meta.mode is SourceMode.CHANNEL:
        if meta.published:
            lines.append(f"Тизер опубликован ✅ (msg {meta.published_message_id})")
        else:
            lines.append("Тизер <b>не</b> опубликован ⚠️")
    else:
        lines.append("В канал ничего не отправлялось (приватный режим)")
    lines.append(f"Время обработки: {meta.processing_seconds:.0f} c")
    return "\n".join(lines)


def render_teaser_preview(result: PipelineResult) -> str:
    if result.teaser is None:
        return ""
    body = result.teaser.render(include_timestamps=True)
    header = (
        "<b>📣 Тизер для канала</b>"
        if result.metadata.mode is SourceMode.CHANNEL
        else "<b>📣 Тизер (только для проверки, не опубликован)</b>"
    )
    chunks = [header, f"<blockquote>{escape(body)}</blockquote>"]
    if result.teaser.reasoning:
        chunks.append(f"<i>{escape(result.teaser.reasoning)}</i>")
    return "\n".join(chunks)


def render_threads(candidates: list[ThreadsCandidate]) -> str:
    if not candidates:
        return "<b>🧵 Threads</b>\nНичего не дотянуло до поста в этой записи."
    blocks = [f"<b>🧵 Threads — {len(candidates)} кандидат(ов)</b>"]
    for index, item in enumerate(candidates, start=1):
        parts = [
            f"<b>{index}. {escape(item.angle) or 'без названия'}</b>",
            f"⏱ {item.timecode}  ·  готовность {item.readiness_score:.0f}/10"
            + (f"  ·  <code>{escape(item.atom_id)}</code>" if item.atom_id else ""),
        ]
        if item.why_it_works:
            parts.append(f"<i>{escape(item.why_it_works)}</i>")
        parts.append("")
        parts.append(escape(item.draft))
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def render_reels(candidates: list[ReelsCandidate]) -> str:
    if not candidates:
        return "<b>🎬 Reels</b>\nНи один фрагмент не годится для Reels."
    blocks = [f"<b>🎬 Reels — {len(candidates)} кандидат(ов)</b>"]
    for index, item in enumerate(candidates, start=1):
        parts = [
            f"<b>{index}. {escape(item.concept)}</b>",
            f"⏱ {item.timecode}  ·  {item.score:.0f}/10  ·  "
            f"~{item.target_duration_seconds} сек"
            + (f"  ·  <code>{escape(item.atom_id)}</code>" if item.atom_id else ""),
        ]
        if item.why_it_works:
            parts.append(f"<i>{escape(item.why_it_works)}</i>")
        beats = [
            ("HOOK", item.hook),
            ("SETUP", item.setup),
            ("РАЗВИТИЕ", item.development),
            ("ПЕЙОФФ", item.payoff),
            ("ФИНАЛ", item.ending),
        ]
        for label, value in beats:
            if value:
                parts.append(f"<b>{label}:</b> {escape(value)}")
        if item.on_screen_text:
            captions = " · ".join(escape(text) for text in item.on_screen_text)
            parts.append(f"<b>Текст на экране:</b> {captions}")
        parts.append(f"<b>СЦЕНАРИЙ</b>\n{escape(item.script)}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def render_report(result: PipelineResult) -> str:
    """Full owner report, ready for :func:`split_html_message`."""
    sections = [render_metadata(result.metadata)]

    teaser = render_teaser_preview(result)
    if teaser:
        sections.append(teaser)

    sections.append(render_threads(result.threads))
    sections.append(render_reels(result.reels))

    if result.warnings:
        warnings = "\n".join(f"• {escape(item)}" for item in result.warnings)
        sections.append(f"<b>⚠️ Предупреждения</b>\n{warnings}")

    return "\n\n".join(sections)


def render_failure(mode: SourceMode, chat_id: int, message_id: int, error: BaseException) -> str:
    """Owner notification for a job that ultimately failed."""
    return "\n".join(
        [
            "<b>❌ Обработка записи не удалась</b>",
            f"Режим: {escape(_MODE_LABELS.get(mode, mode.value))}",
            f"Источник: <code>{chat_id}</code> / msg <code>{message_id}</code>",
            f"Ошибка: <code>{escape(type(error).__name__)}</code>",
            f"<blockquote>{escape(str(error)[:1500])}</blockquote>",
            "",
            "Ничего в канал не опубликовано. Перешли голосовое боту в личку, "
            "чтобы попробовать снова.",
        ]
    )
