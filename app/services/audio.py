"""Audio preprocessing: ffmpeg discovery, downsampling and overlapped chunking.

Strategy for a 15-60 minute recording:

1. Probe the real duration (ffprobe, else parse ffmpeg output, else trust the
   duration Telegram reported).
2. If the original file already fits under ``MAX_STT_UPLOAD_MB`` and has a
   format Groq accepts, upload it untouched — a Telegram ``.oga`` voice note of
   20 minutes is only a few MB, so this is the common path.
3. Otherwise transcode to mono / 16 kHz / low-bitrate (defaults to mp3, whose
   encoder is present in every ffmpeg build). 60 minutes lands around 14 MB.
4. If it is *still* too large, split into chunks with a small overlap so no word
   is lost at a boundary. Chunk length is the smaller of the configured value
   and whatever fits the upload limit at the measured bitrate.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)

# Container/extension list accepted by Groq's transcription endpoint.
GROQ_SUPPORTED_EXTENSIONS = frozenset(
    {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".oga", ".opus", ".wav", ".webm"}
)

# Shorter chunks than this fragment sentences badly and hurt transcription.
MIN_CHUNK_SECONDS = 15.0

_TIME_RE = re.compile(r"time=(\d+):(\d{2}):(\d{2})(?:\.(\d+))?")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})(?:\.(\d+))?")


class AudioProcessingError(RuntimeError):
    """ffmpeg failed or is unavailable when it was required."""


class AudioTooLongError(AudioProcessingError):
    """The recording exceeds ``MAX_AUDIO_DURATION_MINUTES``."""


@dataclass(slots=True)
class AudioChunk:
    """One uploadable piece of audio, positioned on the original timeline."""

    index: int
    path: Path
    offset_seconds: float
    duration: float
    overlap_seconds: float = 0.0

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


@dataclass(slots=True)
class PreparedAudio:
    chunks: list[AudioChunk]
    duration: float
    transcoded: bool
    split: bool

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def resolve_ffmpeg(override: str = "") -> str | None:
    """Locate an ffmpeg binary.

    Order: explicit override, ffmpeg on PATH (Docker/apt), then the static
    binary shipped by ``imageio-ffmpeg`` so the plain Render Python runtime
    works too.
    """
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        logger.debug("imageio-ffmpeg unavailable")
        return None


def _hhmmss_to_seconds(match: re.Match[str]) -> float:
    hours, minutes, seconds = (int(match.group(i)) for i in (1, 2, 3))
    fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


class AudioProcessor:
    """Wraps every ffmpeg invocation this project needs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ffmpeg = resolve_ffmpeg(settings.ffmpeg_binary)
        if self._ffmpeg is None:
            logger.warning(
                "ffmpeg not found; oversized recordings cannot be processed. "
                "Install ffmpeg or the imageio-ffmpeg package."
            )
        else:
            logger.info("Using ffmpeg at %s", self._ffmpeg)

    @property
    def ffmpeg_available(self) -> bool:
        return self._ffmpeg is not None

    async def _run(self, args: list[str]) -> tuple[int, str]:
        if self._ffmpeg is None:
            raise AudioProcessingError("ffmpeg binary is not available")
        command = [self._ffmpeg, "-nostdin", "-hide_banner", *args]
        logger.debug("ffmpeg: %s", " ".join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self._settings.ffmpeg_timeout_seconds
            )
        except TimeoutError as error:
            process.kill()
            raise AudioProcessingError("ffmpeg timed out") from error
        return process.returncode or 0, (stdout or b"").decode("utf-8", "replace")

    # ------------------------------------------------------------------ probing
    async def probe_duration(self, path: Path, fallback: float | None = None) -> float:
        """Best-effort duration in seconds."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            process = await asyncio.create_subprocess_exec(
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            try:
                value = float(stdout.decode().strip())
                if value > 0:
                    return value
            except (ValueError, UnicodeDecodeError):
                logger.debug("ffprobe returned no usable duration for %s", path)

        if self.ffmpeg_available:
            # `-f null -` decodes the file and prints the final timestamp.
            _, output = await self._run(["-i", str(path), "-f", "null", "-"])
            seconds = 0.0
            for match in _TIME_RE.finditer(output):
                seconds = _hhmmss_to_seconds(match)
            if seconds > 0:
                return seconds
            duration_match = _DURATION_RE.search(output)
            if duration_match:
                return _hhmmss_to_seconds(duration_match)

        if fallback and fallback > 0:
            logger.info("Falling back to Telegram-reported duration %.0fs", fallback)
            return float(fallback)
        return 0.0

    # ---------------------------------------------------------------- transcode
    async def transcode(self, source: Path, destination: Path) -> Path:
        settings = self._settings
        code, output = await self._run(
            [
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(settings.audio_target_sample_rate),
                "-c:a",
                settings.audio_target_codec,
                "-b:a",
                settings.audio_target_bitrate,
                str(destination),
            ]
        )
        if code != 0 or not destination.exists():
            raise AudioProcessingError(f"ffmpeg transcode failed (exit {code}): {output[-600:]}")
        return destination

    async def extract_chunk(
        self, source: Path, destination: Path, offset: float, length: float
    ) -> Path:
        settings = self._settings
        code, output = await self._run(
            [
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-t",
                f"{length:.3f}",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(settings.audio_target_sample_rate),
                "-c:a",
                settings.audio_target_codec,
                "-b:a",
                settings.audio_target_bitrate,
                str(destination),
            ]
        )
        if code != 0 or not destination.exists():
            raise AudioProcessingError(
                f"ffmpeg chunk extraction failed (exit {code}): {output[-600:]}"
            )
        return destination

    # ------------------------------------------------------------------- public
    def plan_chunks(
        self, duration: float, bytes_per_second: float
    ) -> list[tuple[float, float, float]]:
        """Return ``(offset, length, overlap)`` triples covering ``duration``.

        Chunk length respects both ``AUDIO_CHUNK_MINUTES`` and the byte budget
        implied by ``MAX_STT_UPLOAD_MB``. A floor of ``MIN_CHUNK_SECONDS`` stops
        us from shredding speech into unusable fragments; if even that floor
        cannot fit the upload limit, we warn instead of failing silently, since
        lowering the bitrate is the correct fix at that point.
        """
        settings = self._settings
        limit = settings.max_stt_upload_bytes * 0.9
        chunk_seconds = settings.audio_chunk_seconds
        if bytes_per_second > 0:
            chunk_seconds = min(chunk_seconds, limit / bytes_per_second)
        if chunk_seconds < MIN_CHUNK_SECONDS:
            logger.warning(
                "Byte budget allows only %.1fs per chunk; clamping to %.0fs. Lower "
                "AUDIO_TARGET_BITRATE or raise MAX_STT_UPLOAD_MB — chunks may be "
                "rejected by Groq as too large.",
                chunk_seconds,
                MIN_CHUNK_SECONDS,
            )
        chunk_seconds = max(MIN_CHUNK_SECONDS, chunk_seconds)

        overlap = min(settings.audio_chunk_overlap_seconds, chunk_seconds / 4)
        step = max(1.0, chunk_seconds - overlap)

        plan: list[tuple[float, float, float]] = []
        offset = 0.0
        while offset < duration - 0.5:
            length = min(chunk_seconds, duration - offset)
            plan.append((offset, length, 0.0 if not plan else overlap))
            if offset + length >= duration - 0.5:
                break
            offset += step
        if not plan:
            plan.append((0.0, max(duration, 1.0), 0.0))
        return plan

    async def prepare(
        self,
        source: Path,
        workdir: Path,
        reported_duration: float | None = None,
    ) -> PreparedAudio:
        """Turn a downloaded file into a list of uploadable chunks."""
        settings = self._settings
        duration = await self.probe_duration(source, fallback=reported_duration)
        max_seconds = settings.max_audio_duration_minutes * 60
        if duration and duration > max_seconds:
            raise AudioTooLongError(
                f"Recording is {duration / 60:.0f} min, limit is "
                f"{settings.max_audio_duration_minutes:.0f} min"
            )

        size = source.stat().st_size
        limit = settings.max_stt_upload_bytes
        extension = source.suffix.lower()

        if size <= limit and extension in GROQ_SUPPORTED_EXTENSIONS:
            logger.info(
                "Uploading original audio as-is (%.1f MB, %.0fs)",
                size / 1024 / 1024,
                duration,
            )
            return PreparedAudio(
                chunks=[AudioChunk(index=0, path=source, offset_seconds=0.0, duration=duration)],
                duration=duration,
                transcoded=False,
                split=False,
            )

        if not self.ffmpeg_available:
            raise AudioProcessingError(
                f"Audio is {size / 1024 / 1024:.1f} MB / format {extension or 'unknown'}, "
                "which needs ffmpeg, but no ffmpeg binary was found."
            )

        converted = workdir / f"normalized.{settings.audio_target_ext}"
        await self.transcode(source, converted)
        converted_size = converted.stat().st_size
        logger.info(
            "Transcoded %.1f MB -> %.1f MB",
            size / 1024 / 1024,
            converted_size / 1024 / 1024,
        )
        if not duration:
            duration = await self.probe_duration(converted, fallback=reported_duration)

        if converted_size <= limit:
            return PreparedAudio(
                chunks=[AudioChunk(index=0, path=converted, offset_seconds=0.0, duration=duration)],
                duration=duration,
                transcoded=True,
                split=False,
            )

        bytes_per_second = converted_size / duration if duration else 0.0
        plan = self.plan_chunks(duration, bytes_per_second)
        logger.info("Splitting into %s chunks", len(plan))

        chunks: list[AudioChunk] = []
        for index, (offset, length, overlap) in enumerate(plan):
            target = workdir / f"chunk-{index:03d}.{settings.audio_target_ext}"
            await self.extract_chunk(converted, target, offset, length)
            chunks.append(
                AudioChunk(
                    index=index,
                    path=target,
                    offset_seconds=offset,
                    duration=length,
                    overlap_seconds=overlap,
                )
            )
        return PreparedAudio(chunks=chunks, duration=duration, transcoded=True, split=True)
