#!/usr/bin/env python3
"""Register, inspect or delete the Telegram webhook.

Usage (from the repo root, with .env filled in):

    python scripts/set_webhook.py set
    python scripts/set_webhook.py info
    python scripts/set_webhook.py delete

``set`` uses PUBLIC_BASE_URL + WEBHOOK_PATH, and passes WEBHOOK_SECRET as
``secret_token`` when it is configured. ``--url`` overrides PUBLIC_BASE_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings

# channel_post is what production mode needs; message covers private mode.
ALLOWED_UPDATES = ["message", "channel_post", "edited_channel_post"]


async def call(token: str, method: str, **params: object) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"https://api.telegram.org/bot{token}/{method}", json=params)
        payload = response.json()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload.get("ok"):
        raise SystemExit(1)
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["set", "info", "delete"])
    parser.add_argument("--url", default="", help="Override PUBLIC_BASE_URL")
    parser.add_argument(
        "--drop-pending",
        action="store_true",
        help="Discard updates Telegram queued while the service was down",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN is not set (check your .env)")

    if args.action == "info":
        await call(settings.bot_token, "getWebhookInfo")
        return

    if args.action == "delete":
        await call(
            settings.bot_token,
            "deleteWebhook",
            drop_pending_updates=args.drop_pending,
        )
        return

    base = (args.url or settings.public_base_url).rstrip("/")
    if not base:
        raise SystemExit("Set PUBLIC_BASE_URL in .env or pass --url https://...")
    if not base.startswith("https://"):
        raise SystemExit("Telegram requires an https webhook URL")

    url = f"{base}{settings.webhook_path}"
    params: dict[str, object] = {
        "url": url,
        "allowed_updates": ALLOWED_UPDATES,
        "drop_pending_updates": args.drop_pending,
        "max_connections": 10,
    }
    if settings.webhook_secret:
        params["secret_token"] = settings.webhook_secret

    print(f"Setting webhook to {url}")
    await call(settings.bot_token, "setWebhook", **params)
    await call(settings.bot_token, "getWebhookInfo")


if __name__ == "__main__":
    asyncio.run(main())
