"""Application configuration.

All tunables live here and are driven by environment variables so that the
deployment (Render) can be reconfigured without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS_DIR = PROJECT_ROOT / "prompts"


# Substrings that mark an unedited value copied straight from .env.example.
_PLACEHOLDER_MARKERS = ("replace-me", "replace_me", "your-", "xxxx", "...")


def _is_placeholder(value: str) -> bool:
    lowered = (value or "").strip().lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _parse_id_list(raw: str | list[int] | None) -> list[int]:
    """Parse a comma/space separated list of Telegram ids."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(item) for item in raw]
    cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    return [int(part) for part in cleaned.split() if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- Telegram
    bot_token: str = Field(default="", description="Telegram bot token.")
    allowed_channel_id: int = Field(
        default=0,
        description="The single broadcast channel whose posts are processed.",
    )
    owner_telegram_id: int = Field(
        default=0, description="User id that receives all private reports."
    )
    # Kept as a raw string because env vars cannot express list[int] cleanly;
    # `allowed_user_ids` below is the parsed accessor.
    allowed_user_ids_raw: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ALLOWED_USER_IDS", "allowed_user_ids", "allowed_user_ids_raw"
        ),
        description="Comma separated user ids allowed to use private mode.",
    )
    webhook_secret: str = Field(
        default="",
        description=(
            "Optional value for X-Telegram-Bot-Api-Secret-Token. When set, "
            "requests without a matching header are rejected."
        ),
    )
    webhook_path: str = Field(default="/telegram/webhook")
    public_base_url: str = Field(
        default="",
        description="Public https base url of the service, used by the webhook script.",
    )

    # -------------------------------------------------------------------- Groq
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_whisper_model: str = Field(default="whisper-large-v3")
    groq_llm_model: str = Field(default="openai/gpt-oss-120b")
    groq_llm_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    groq_mining_max_tokens: int = Field(default=1200, gt=0)
    groq_teaser_max_tokens: int = Field(default=400, gt=0)
    groq_threads_max_tokens: int = Field(default=1200, gt=0)
    groq_reels_max_tokens: int = Field(default=1600, gt=0)
    groq_tpm_limit: int = Field(default=8000, gt=0)
    threads_batch_size: int = Field(default=2, ge=1, le=3)
    reels_batch_size: int = Field(default=2, ge=1, le=3)
    groq_llm_max_tokens: int = Field(default=1200, gt=0)
    groq_timeout_seconds: float = Field(default=600.0, gt=0)
    groq_max_retries: int = Field(default=3, ge=0, le=10)
    groq_retry_base_delay: float = Field(default=5.0, ge=0.0)
    groq_retry_max_delay: float = Field(default=120.0, ge=0.0)
    groq_use_json_schema: bool = Field(
        default=True,
        description="Enable strict schemas for other models; GPT-OSS 120B always uses strict schemas.",
    )
    transcript_language: str = Field(
        default="ru",
        description="Language hint for Groq STT. Empty string means auto-detect.",
    )

    # ------------------------------------------------------------------- Audio
    max_stt_upload_mb: float = Field(default=24.0, gt=0)
    audio_chunk_minutes: float = Field(default=10.0, gt=0)
    audio_chunk_overlap_seconds: float = Field(default=5.0, ge=0)
    audio_target_codec: str = Field(default="libmp3lame")
    audio_target_ext: str = Field(default="mp3")
    audio_target_bitrate: str = Field(default="32k")
    audio_target_sample_rate: int = Field(default=16000, gt=0)
    ffmpeg_binary: str = Field(default="", description="Override path to ffmpeg.")
    ffmpeg_timeout_seconds: float = Field(default=900.0, gt=0)
    max_audio_duration_minutes: float = Field(
        default=75.0,
        gt=0,
        description="Safety valve; longer recordings are rejected with a notice.",
    )
    # Bot API getFile cannot download files larger than 20 MB.
    telegram_max_download_mb: float = Field(default=20.0, gt=0)

    # ----------------------------------------------------------------- Content
    miner_window_minutes: float = Field(
        default=20.0,
        gt=0,
        description="Transcript is mined window by window so long audio stays deep.",
    )
    miner_window_overlap_minutes: float = Field(default=1.0, ge=0)
    max_threads_candidates: int = Field(default=5, ge=1, le=20)
    max_reels_candidates: int = Field(default=5, ge=1, le=20)
    min_atom_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    teaser_include_timestamps: bool = Field(default=True)
    teaser_reply_to_source: bool = Field(default=True)

    # ------------------------------------------------------------------ Runtime
    enable_full_transcript: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    job_queue_size: int = Field(default=16, gt=0)
    job_concurrency: int = Field(default=1, ge=1, le=4)
    dedupe_ttl_seconds: float = Field(default=6 * 3600.0, gt=0)
    dedupe_max_entries: int = Field(default=2048, gt=0)
    prompts_dir: Path = Field(default=DEFAULT_PROMPTS_DIR)
    work_dir: Path = Field(
        default=Path("/tmp/voice-content-bot"),
        description="Scratch space for downloads and audio chunks.",
    )
    keep_temp_files: bool = Field(default=False)
    dry_run_publish: bool = Field(
        default=False,
        description="When true, the public teaser is reported but never published.",
    )

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("webhook_path")
    @classmethod
    def _leading_slash(cls, value: str) -> str:
        return value if value.startswith("/") else f"/{value}"

    @property
    def allowed_user_ids(self) -> frozenset[int]:
        """Users allowed to drive private mode (owner is always included)."""
        ids = set(_parse_id_list(self.allowed_user_ids_raw))
        if self.owner_telegram_id:
            ids.add(self.owner_telegram_id)
        return frozenset(ids)

    @property
    def max_stt_upload_bytes(self) -> int:
        return int(self.max_stt_upload_mb * 1024 * 1024)

    @property
    def telegram_max_download_bytes(self) -> int:
        return int(self.telegram_max_download_mb * 1024 * 1024)

    @property
    def audio_chunk_seconds(self) -> float:
        return self.audio_chunk_minutes * 60.0

    @property
    def webhook_url(self) -> str:
        base = self.public_base_url.rstrip("/")
        return f"{base}{self.webhook_path}" if base else ""

    def missing_required(self) -> list[str]:
        """Names of required settings that are empty or still a placeholder.

        Placeholder detection matters: copying ``.env.example`` to ``.env`` and
        forgetting to edit it would otherwise report a healthy service that
        fails on the first real update.
        """
        missing: list[str] = []
        if not self.bot_token or _is_placeholder(self.bot_token):
            missing.append("BOT_TOKEN")
        if not self.groq_api_key or _is_placeholder(self.groq_api_key):
            missing.append("GROQ_API_KEY")
        if not self.owner_telegram_id:
            missing.append("OWNER_TELEGRAM_ID")
        if not self.allowed_channel_id:
            missing.append("ALLOWED_CHANNEL_ID")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
