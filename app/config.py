"""Application configuration.

All tunables live here and are driven by environment variables so that the
deployment (Render) can be reconfigured without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
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

    transcription_provider: Literal["groq"] = "groq"
    text_provider: Literal["openrouter", "groq"] = "openrouter"
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_allow_paid: bool = False
    openrouter_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "none"
    openrouter_timeout_seconds: float = Field(default=120, gt=0)
    openrouter_max_retries: int = Field(default=2, ge=0, le=3)
    openrouter_max_requests_per_recording: int = Field(default=60, ge=1, le=100)
    # Optional single ceiling. None delegates completion sizing to the provider.
    # Legacy per-stage fields below remain accepted but do not constrain OpenRouter.
    openrouter_max_output_tokens: int | None = Field(default=None, gt=0)
    text_max_input_tokens: int = Field(default=12000, ge=1000)
    text_extraction_max_tokens: int = Field(default=3000, gt=0)
    text_teaser_max_tokens: int = Field(default=1800, gt=0)
    text_threads_max_tokens: int = Field(default=5000, gt=0)
    text_reels_max_tokens: int = Field(default=6000, gt=0)

    # Semantic editor. Model IDs deliberately have no application defaults.
    semantic_pipeline_enabled: bool = True
    openrouter_primary_model: str = ""
    openrouter_primary_fallback_model: str = ""
    openrouter_escalation_model: str = ""
    openrouter_allow_escalation: bool = False
    quality_auditor: Literal["none", "kimi_k3", "claude", "auto"] = "auto"
    writing_provider: Literal["primary", "escalation", "claude"] = "primary"
    force_quality_audit: bool = False
    extraction_window_minutes: float = Field(default=12, gt=0)
    extraction_overlap_minutes: float = Field(default=2, ge=0)
    target_atom_limit: int = Field(default=10, ge=1)
    overflow_max_passes: int = Field(default=2, ge=1, le=5)
    semantic_max_input_tokens: int = Field(default=90000, ge=4000)
    semantic_max_output_tokens: int = Field(default=8000, ge=1000)
    semantic_reasoning_effort: Literal["none", "low", "medium", "high"] = "medium"
    writing_reasoning_effort: Literal["none", "low", "medium", "high"] = "low"
    escalate_if_duration_minutes: float = Field(default=30, gt=0)
    coverage_confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    claude_enabled: bool = False
    claude_model: str = ""
    claude_code_oauth_token: str = Field(default="", repr=False)
    claude_timeout_seconds: float = Field(default=300, gt=0)
    claude_max_budget_usd: float = Field(default=0.5, gt=0)
    checkpoint_backend: Literal["none", "sqlite", "supabase"] = "sqlite"
    checkpoint_sqlite_path: Path = Path("/tmp/voice-content-bot/checkpoints.sqlite3")
    supabase_url: str = ""
    supabase_service_role_key: str = Field(default="", repr=False)
    monthly_llm_budget_usd: float = Field(default=15, gt=0)
    soft_budget_warning_usd: float = Field(default=10, ge=0)
    stop_on_budget_exceeded: bool = False

    # -------------------------------------------------------------------- Groq
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_whisper_model: str = Field(default="whisper-large-v3")
    groq_llm_model: str = Field(default="openai/gpt-oss-120b")
    groq_llm_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    groq_extraction_max_tokens: int = Field(default=1500, ge=700, le=1600)
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
        default=12.0,
        gt=0,
        description="Transcript is mined window by window so long audio stays deep.",
    )
    miner_window_overlap_minutes: float = Field(default=0.5, ge=0)
    max_threads_candidates: int = Field(default=4, ge=1, le=5)
    max_reels_candidates: int = Field(default=4, ge=1, le=5)
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

    @model_validator(mode="after")
    def _semantic_windows(self):
        if self.extraction_overlap_minutes >= self.extraction_window_minutes:
            raise ValueError(
                "EXTRACTION_OVERLAP_MINUTES must be smaller than EXTRACTION_WINDOW_MINUTES"
            )
        return self

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
        if self.text_provider == "openrouter" and (
            not self.openrouter_api_key or _is_placeholder(self.openrouter_api_key)
        ):
            missing.append("OPENROUTER_API_KEY")
        if (
            self.semantic_pipeline_enabled
            and self.text_provider == "openrouter"
            and not self.openrouter_primary_model
        ):
            missing.append("OPENROUTER_PRIMARY_MODEL")
        if self.checkpoint_backend == "supabase":
            if not self.supabase_url:
                missing.append("SUPABASE_URL")
            if not self.supabase_service_role_key:
                missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if self.claude_enabled:
            if not self.claude_model:
                missing.append("CLAUDE_MODEL")
            if not self.claude_code_oauth_token:
                missing.append("CLAUDE_CODE_OAUTH_TOKEN")
        if not self.owner_telegram_id:
            missing.append("OWNER_TELEGRAM_ID")
        if not self.allowed_channel_id:
            missing.append("ALLOWED_CHANNEL_ID")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
