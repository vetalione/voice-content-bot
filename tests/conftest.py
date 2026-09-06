"""Shared fixtures and offline fakes.

No test in this suite performs a network call: the Telegram Bot session, the
Groq client, the file downloader and ffmpeg are all replaced with fakes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Message, Update

from app.config import Settings
from app.models.media import JobRequest, MediaKind, MediaRef, SourceMode
from app.models.transcript import ChunkTranscript, Transcript, TranscriptSegment
from app.services.audio import AudioChunk
from app.services.dedupe import TTLDedupeStore
from app.services.jobs import JobRunner
from app.telegram.handlers import JobSubmitter, build_router

# A syntactically valid token; no request is ever made with it.
FAKE_TOKEN = "123456789:AAHfYh_ThisIsAFakeTokenForTestsOnly12345"
CHANNEL_ID = -1001234567890
OWNER_ID = 424242
ALLOWED_USER_ID = 555000
STRANGER_ID = 999999


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        text_provider="groq",
        bot_token=FAKE_TOKEN,
        allowed_channel_id=CHANNEL_ID,
        owner_telegram_id=OWNER_ID,
        allowed_user_ids=f"{ALLOWED_USER_ID}",
        groq_api_key="test-key",
        work_dir=tmp_path / "work",
        prompts_dir=Path(__file__).resolve().parent.parent / "prompts",
        log_level="WARNING",
        _env_file=None,
    )


@pytest.fixture
def offline_bot(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A real Bot object with the transport stubbed out.

    ``Bot.__call__`` is where every aiogram API request funnels through, so
    patching it guarantees no test can reach api.telegram.org. Sent method
    objects are recorded for assertions.
    """
    sent: list[Any] = []

    async def fake_call(self: Bot, method: Any, request_timeout: int | None = None) -> None:
        sent.append(method)
        return None

    monkeypatch.setattr(Bot, "__call__", fake_call, raising=True)
    bot = Bot(token=FAKE_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    return SimpleNamespace(bot=bot, sent=sent)


# --------------------------------------------------------------------- fakes ---
class FakeDelivery:
    """Records every outbound Telegram action instead of performing it."""

    def __init__(self) -> None:
        self.owner_messages: list[str] = []
        self.owner_documents: list[tuple[str, bytes, str]] = []
        self.plain_messages: list[tuple[int, str]] = []
        self.channel_posts: list[tuple[str, int | None]] = []
        self.next_message_id = 5001

    async def send_owner_html(self, text: str) -> None:
        self.owner_messages.append(text)

    async def send_owner_document(self, filename: str, content: bytes, caption: str = "") -> None:
        self.owner_documents.append((filename, content, caption))

    async def send_plain(self, chat_id: int, text: str) -> None:
        self.plain_messages.append((chat_id, text))

    async def publish_to_channel(
        self, text: str, reply_to_message_id: int | None = None
    ) -> int | None:
        self.channel_posts.append((text, reply_to_message_id))
        self.next_message_id += 1
        return self.next_message_id

    @property
    def published(self) -> bool:
        return bool(self.channel_posts)


class FakeDownloader:
    """Writes a stub file so the pipeline has something on disk."""

    def __init__(self) -> None:
        self.calls: list[MediaRef] = []

    async def download(self, media: MediaRef, directory: Path) -> Path:
        self.calls.append(media)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "source.oga"
        path.write_bytes(b"fake-audio-bytes")
        return path


class FakeAudioProcessor:
    """Skips ffmpeg entirely: one chunk, no transcoding."""

    def __init__(self, duration: float = 1200.0, chunks: int = 1) -> None:
        self.duration = duration
        self.chunks = chunks
        self.ffmpeg_available = True
        self.calls = 0

    async def prepare(
        self, source: Path, workdir: Path, reported_duration: float | None = None
    ) -> Any:
        from app.services.audio import PreparedAudio

        self.calls += 1
        per_chunk = self.duration / self.chunks
        return PreparedAudio(
            chunks=[
                AudioChunk(
                    index=index,
                    path=source,
                    offset_seconds=index * per_chunk,
                    duration=per_chunk,
                )
                for index in range(self.chunks)
            ],
            duration=self.duration,
            transcoded=False,
            split=self.chunks > 1,
        )


class FakeTranscriber:
    """Returns canned segments; can be told to raise instead."""

    def __init__(
        self,
        segments: list[TranscriptSegment] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.segments = segments or [
            TranscriptSegment(start=0.0, end=30.0, text="Я почти опоздал на самолёт."),
            TranscriptSegment(
                start=30.0,
                end=90.0,
                text="Вокруг AI скоро появятся почти религиозные сообщества.",
            ),
        ]
        self.error = error
        self.calls: list[AudioChunk] = []

    async def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript:
        self.calls.append(chunk)
        if self.error is not None:
            raise self.error
        return ChunkTranscript(
            index=chunk.index,
            offset_seconds=chunk.offset_seconds,
            overlap_seconds=chunk.overlap_seconds,
            duration=chunk.duration,
            language="ru",
            segments=list(self.segments),
            text=" ".join(item.text for item in self.segments),
        )


class FakeLLM:
    """Serves scripted JSON payloads keyed by the agent label prefix."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self.responses = responses or default_llm_responses()
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        label = str(kwargs.get("label", ""))
        if label.startswith("content_extraction"):
            return {
                "atoms": [
                    {
                        "title": a["label"],
                        "start_seconds": a["start_seconds"],
                        "end_seconds": a["end_seconds"],
                        "idea": a["key_claim"],
                        "type": "idea",
                    }
                    for a in self.responses["content_miner"]["atoms"]
                ]
            }
        if label.startswith("content_enrichment"):
            a = next(
                (
                    a
                    for a in self.responses["content_miner"]["atoms"]
                    if a["label"] in kwargs["user"]
                ),
                self.responses["content_miner"]["atoms"][0],
            )
            return {
                k: a[k]
                for k in (
                    "categories",
                    "business_score",
                    "personal_score",
                    "novelty_score",
                    "confidence",
                    "should_ignore",
                    "ignore_reason",
                )
            }
        for key, payload in self.responses.items():
            if label.startswith(key):
                return json.loads(json.dumps(payload))
        return {}


def default_llm_responses() -> dict[str, dict[str, Any]]:
    return {
        "content_miner": {
            "atoms": [
                {
                    "id": "a1",
                    "label": "Почти опоздал на самолёт",
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "description": "Личная история про аэропорт.",
                    "key_claim": "Почти опоздал на самолёт.",
                    "supporting_context": "Я почти опоздал на самолёт.",
                    "categories": ["travel_incident", "personal_story"],
                    "business_score": 2,
                    "personal_score": 8,
                    "novelty_score": 6,
                    "confidence": 0.8,
                    "should_ignore": False,
                    "ignore_reason": "",
                },
                {
                    "id": "a2",
                    "label": "Религиозные сообщества вокруг AI",
                    "start_seconds": 30,
                    "end_seconds": 90,
                    "description": "Прогноз о том, что вокруг AI появятся квазирелигиозные сообщества.",
                    "key_claim": "Вокруг AI сформируются почти религиозные сообщества.",
                    "supporting_context": "Вокруг AI скоро появятся почти религиозные сообщества.",
                    "categories": ["prediction", "culture_observation"],
                    "business_score": 8,
                    "personal_score": 3,
                    "novelty_score": 9,
                    "confidence": 0.9,
                    "should_ignore": False,
                    "ignore_reason": "",
                },
            ],
            "notes": "",
        },
        "channel_teaser": {
            "teaser": "Здесь неожиданно сошлись аэропорт и мысль про новые сообщества вокруг AI.",
            "timestamps": [],
            "reasoning": "Столкновение бытового и большого прогноза.",
        },
        "threads_editor": {
            "candidates": [
                {
                    "atom_id": "a2",
                    "angle": "Религии вокруг AI",
                    "why_it_works": "Сильный неочевидный прогноз.",
                    "readiness_score": 8,
                    "draft": "Вокруг AI начнут собираться сообщества, которые ведут себя как религии.",
                }
            ],
            "rejected": ["a1 — личная история без профессионального вывода"],
        },
        "reels_editor": {
            "candidates": [
                {
                    "atom_id": "a1",
                    "concept": "Аэропорт",
                    "why_it_works": "Есть сцена и напряжение.",
                    "score": 7,
                    "target_duration_seconds": 45,
                    "hook": "Я стоял у гейта, который уже закрывали.",
                    "setup": "Рейс был последним в тот день.",
                    "development": "Очередь не двигалась.",
                    "payoff": "Успел за минуту до закрытия.",
                    "ending": "С тех пор выезжаю на час раньше.",
                    "on_screen_text": ["минус одна минута"],
                    "script": "Я стоял у гейта, который уже закрывали. И успел.",
                }
            ],
            "rejected": [],
        },
    }


# ------------------------------------------------------------------ builders ---
def make_media(kind: MediaKind = MediaKind.VOICE, duration: int = 1200) -> MediaRef:
    return MediaRef(
        kind=kind,
        file_id="file-id-1",
        file_unique_id="unique-1",
        file_size=3 * 1024 * 1024,
        duration_seconds=duration,
        mime_type="audio/ogg" if kind is MediaKind.VOICE else "audio/mpeg",
        file_name=None if kind is MediaKind.VOICE else "old-voice.mp3",
    )


def make_job_request(
    mode: SourceMode = SourceMode.CHANNEL,
    *,
    message_id: int = 77,
    forwarded: bool = False,
) -> JobRequest:
    return JobRequest(
        mode=mode,
        chat_id=CHANNEL_ID if mode is SourceMode.CHANNEL else ALLOWED_USER_ID,
        message_id=message_id,
        media=make_media(),
        requester_id=None if mode is SourceMode.CHANNEL else ALLOWED_USER_ID,
        posted_at=datetime(2026, 9, 6, 12, 0, tzinfo=UTC),
        forwarded=forwarded,
    )


def voice_payload(duration: int = 1200) -> dict[str, Any]:
    return {
        "duration": duration,
        "mime_type": "audio/ogg",
        "file_id": "voice-file-id",
        "file_unique_id": "voice-unique-id",
        "file_size": 2_500_000,
    }


def audio_payload(duration: int = 1500) -> dict[str, Any]:
    return {
        "duration": duration,
        "mime_type": "audio/mpeg",
        "file_id": "audio-file-id",
        "file_unique_id": "audio-unique-id",
        "file_size": 6_000_000,
        "file_name": "old-column.mp3",
        "title": "Old column",
    }


def channel_post_update(
    update_id: int = 1,
    chat_id: int = CHANNEL_ID,
    message_id: int = 501,
    media: dict[str, Any] | None = None,
    media_key: str = "voice",
    extra: dict[str, Any] | None = None,
) -> Update:
    post: dict[str, Any] = {
        "message_id": message_id,
        "date": 1_757_000_000,
        "chat": {"id": chat_id, "type": "channel", "title": "My column"},
    }
    if media is not None:
        post[media_key] = media
    if extra:
        post.update(extra)
    return Update.model_validate({"update_id": update_id, "channel_post": post})


def private_message_update(
    update_id: int = 1,
    user_id: int = ALLOWED_USER_ID,
    message_id: int = 901,
    media: dict[str, Any] | None = None,
    media_key: str = "voice",
    forwarded: bool = False,
    text: str | None = None,
) -> Update:
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": 1_757_000_000,
        "chat": {"id": user_id, "type": "private", "first_name": "Owner"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Owner"},
    }
    if media is not None:
        message[media_key] = media
    if text is not None:
        message["text"] = text
    if forwarded:
        message["forward_origin"] = {
            "type": "channel",
            "date": 1_756_000_000,
            "chat": {"id": CHANNEL_ID, "type": "channel", "title": "My column"},
            "message_id": 12,
        }
    return Update.model_validate({"update_id": update_id, "message": message})


class RecordingProcessorSpy:
    """Stands in for RecordingProcessor and records what was asked of it."""

    def __init__(self) -> None:
        self.processed: list[JobRequest] = []
        self.failures: list[tuple[JobRequest, BaseException]] = []

    async def process(self, request: JobRequest) -> None:
        self.processed.append(request)

    async def notify_failure(self, request: JobRequest, error: BaseException) -> None:
        self.failures.append((request, error))


@pytest.fixture
def wiring(settings: Settings):
    """A dispatcher wired to spies, with no Groq/Telegram/ffmpeg access."""

    class Wiring:
        def __init__(self) -> None:
            self.settings = settings
            self.processor = RecordingProcessorSpy()
            self.delivery = FakeDelivery()
            self.dedupe = TTLDedupeStore(ttl_seconds=3600, max_entries=100)
            self.runner = JobRunner(concurrency=1, queue_size=8)
            self.submitter = JobSubmitter(
                settings,
                self.runner,
                self.processor,  # type: ignore[arg-type]
                self.dedupe,
                self.delivery,  # type: ignore[arg-type]
            )
            self.dispatcher = Dispatcher()
            self.dispatcher.include_router(build_router(settings, self.submitter, self.runner))

    return Wiring()


@pytest.fixture
def bot_free_message():
    """Build a bare channel Message for filter tests (no Bot instance needed)."""

    def make(chat_id: int, message_id: int = 1, media: dict[str, Any] | None = None):
        data: dict[str, Any] = {
            "message_id": message_id,
            "date": 1_757_000_000,
            "chat": {"id": chat_id, "type": "channel", "title": "Some channel"},
        }
        if media:
            data.update(media)
        return Message.model_validate(data)

    return make


@pytest.fixture
def bot_free_private_message():
    """Build a bare private Message for filter tests."""

    def make(
        user_id: int | None,
        chat_type: str = "private",
        media: dict[str, Any] | None = None,
    ):
        data: dict[str, Any] = {
            "message_id": 1,
            "date": 1_757_000_000,
            "chat": {"id": user_id or 1, "type": chat_type, "first_name": "X"},
        }
        if user_id is not None:
            data["from"] = {"id": user_id, "is_bot": False, "first_name": "X"}
        if media:
            data.update(media)
        return Message.model_validate(data)

    return make


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(start=0.0, end=30.0, text="Я почти опоздал на самолёт."),
            TranscriptSegment(
                start=30.0,
                end=90.0,
                text="Вокруг AI скоро появятся почти религиозные сообщества.",
            ),
        ],
        language="ru",
        duration=90.0,
    )


@pytest.fixture(autouse=True)
def virtual_tpm_clock(monkeypatch):
    """Admission tests and the full suite never spend real minutes sleeping."""
    from app.services.tpm import RollingTPM

    original = RollingTPM.__init__
    now = [0.0]

    async def sleep(delay):
        now[0] += delay

    def init(self, limit, *, clock=None, sleep=None):
        original(self, limit, clock=clock or (lambda: now[0]), sleep=sleep or virtual_sleep)

    virtual_sleep = sleep
    monkeypatch.setattr(RollingTPM, "__init__", init)
