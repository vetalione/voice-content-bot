#!/usr/bin/env python3
"""One tiny OpenRouter request: no Whisper, Telegram, retries or paid routing."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pydantic import BaseModel

from app.agents.base import json_schema_for
from app.config import Settings
from app.services.openrouter_client import OpenRouterClient
from app.services.usage import recording_usage


class SmokeResult(BaseModel):
    ok: bool


async def main():
    # Ignore unrelated Telegram IDs and provider overrides for this text-only check.
    import os

    from dotenv import dotenv_values

    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(".env").get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env; no request was sent.")
    settings = Settings(
        _env_file=None,
        openrouter_api_key=key,
        openrouter_model="openrouter/free",
        openrouter_allow_paid=False,
        openrouter_max_retries=0,
        openrouter_max_requests_per_recording=1,
    )
    client = OpenRouterClient(settings)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        with recording_usage(0, "text-only-smoke"):
            raw = await client.chat_json(
                system="Return a JSON object with ok=true.",
                user="Health test.",
                schema=json_schema_for(SmokeResult),
                schema_name="SmokeResult",
                max_tokens=1024,
            )
            result = SmokeResult.model_validate(raw)
            if not result.ok:
                raise ValueError("Model returned ok=false")
            print("PASS: free-only OpenRouter response validated.")
    except Exception as error:
        raise SystemExit(
            f"{type(error).__name__}: {str(error).replace(key, '[REDACTED]')}"
        ) from None
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
