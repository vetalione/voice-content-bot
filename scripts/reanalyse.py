#!/usr/bin/env python3
"""Re-analyse a saved transcript privately; never calls Whisper or Telegram."""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agents.semantic_miner import SemanticMinerAgent
from app.config import Settings
from app.models.transcript import Transcript
from app.services.checkpoints import build_store, checkpoint_scope
from app.services.claude_client import ClaudeAgentClient
from app.services.prompts import PromptLibrary
from app.services.provider_routing import TextRouter
from app.services.usage import recording_usage, restore_usage


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="Recording ID from the private report")
    parser.add_argument("--auditor", choices=["none", "kimi_k3", "claude", "auto"], default="none")
    parser.add_argument(
        "--primary-model", help="Explicit A/B primary override; no automatic model change"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overrides = {"quality_auditor": args.auditor}
    if args.primary_model:
        overrides["openrouter_primary_model"] = args.primary_model
    settings = Settings(**overrides)
    store = build_store(settings)
    router = TextRouter(
        settings, claude=ClaudeAgentClient(settings) if settings.claude_enabled else None
    )
    try:
        used = await router.monthly_usage()
        if used is not None and used >= settings.soft_budget_warning_usd:
            print(f"OpenRouter monthly reported usage: ${used:.2f}")
        if (
            used is not None
            and used >= settings.monthly_llm_budget_usd
            and settings.stop_on_budget_exceeded
        ):
            raise SystemExit("Configured monthly budget reached")
        saved = await store.get(args.recording, "transcript")
        if not saved:
            raise SystemExit("Saved transcript not found. Check recording ID/storage settings.")
        transcript = Transcript.model_validate(saved["transcript"])
        with (
            checkpoint_scope(store, args.recording),
            recording_usage(transcript.duration, args.recording) as usage,
        ):
            await restore_usage(store, args.recording, usage)
            result = await SemanticMinerAgent(
                router, PromptLibrary(settings.prompts_dir), settings
            ).mine(transcript)
        data = result.model_dump(mode="json")
        await store.put(args.recording, f"comparison:{args.auditor}:{time.time_ns()}", data)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"Saved {len(result.atoms)} atoms to {args.output}; no public publication.")
    finally:
        await router.aclose()
        await store.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
