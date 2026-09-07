#!/usr/bin/env python3
"""Resume a saved private recording; inspect by default, never call Whisper.

Requires unchanged prompts/models for completed checkpoints. Use --run to
generate unfinished stages and deliver the result to the configured owner.
Do not run while the same recording is processing on Render.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.container import build_container
from app.models.media import JobRequest, SourceMode
from app.models.transcript import Transcript
from app.services.checkpoints import SupabaseStore, checkpoint_scope, recording_id
from app.services.usage import recording_usage, restore_usage


class ResumeBlocked(RuntimeError):
    pass


class NextStage(RuntimeError):
    pass


class InspectionStore:
    def __init__(self, store):
        self.store = store

    async def get(self, job, stage):
        return await self.store.get(job, stage)

    async def put(self, *args):
        pass


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--primary-model", required=True)
    parser.add_argument("--escalation-model", default="")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = Settings(
        checkpoint_backend="supabase",
        text_provider="openrouter",
        semantic_pipeline_enabled=True,
        openrouter_primary_model=args.primary_model,
        openrouter_escalation_model=args.escalation_model,
        openrouter_allow_paid=args.allow_paid,
        openrouter_allow_escalation=bool(args.escalation_model) and args.allow_paid,
        claude_enabled=False,
    )
    container = build_container(settings)
    store = container.pipeline.checkpoints
    assert isinstance(store, SupabaseStore)
    try:
        request = JobRequest.model_validate(await store.get(args.recording, "request"))
        if (
            request.mode is not SourceMode.PRIVATE
            or request.chat_id != settings.owner_telegram_id
            or recording_id(request) != args.recording
        ):
            raise ResumeBlocked("Only the configured owner's saved private recording can resume")
        saved = await store.get(args.recording, "transcript")
        if not saved:
            raise ResumeBlocked("Saved transcript missing; no Whisper call will be made")
        transcript = Transcript.model_validate(saved["transcript"])
        status = await store.get(args.recording, "status") or {}
        if status.get("status") == "complete":
            raise ResumeBlocked("Recording is already complete; refusing duplicate delivery")
        response = await store._request(
            "GET", params={"job": f"eq.{args.recording}", "select": "stage,data"}
        )
        completed = {
            row["data"]["stage"]
            for row in response.json()
            if row["data"].get("status") == "complete" and row["data"].get("stage")
        }
        logging.info(
            "Saved duration=%ss completed_stages=%s", transcript.duration, sorted(completed)
        )
        router = container.text
        original_request = router.request

        async def guarded_request(role, method="chat_json", **kwargs):
            label = kwargs.get("label", "").split("#", 1)[0]
            if label in completed:
                raise ResumeBlocked(
                    f"Completed stage {label} cache does not match settings/prompts; refusing regeneration"
                )
            if not args.run:
                raise NextStage(f"Next unfinished stage: {label}; no API generation was performed")
            return await original_request(role, method=method, **kwargs)

        router.request = guarded_request
        if args.run:
            await container.processor._check_budget(enforce=True)
        active_store = store if args.run else InspectionStore(store)
        with (
            checkpoint_scope(active_store, args.recording),
            recording_usage(transcript.duration, args.recording) as usage,
        ):
            await restore_usage(store, args.recording, usage)
            # analyse accepts an existing transcript: download/audio/STT are unreachable.
            result = await container.pipeline.analyse(
                request, transcript, saved["chunks"], time.monotonic()
            )
            if args.run:
                result.recording_id = args.recording
                result.usage_diagnostics = await store.get(args.recording, "usage") or {}
                await store.put(args.recording, "analysis", result.model_dump(mode="json"))
                await store.put(args.recording, "latest_result", result.model_dump(mode="json"))
                await container.processor._deliver(request, result)
                await store.put(args.recording, "status", {"status": "complete"})
                logging.info("Resume complete; private owner report delivered")
    except NextStage as error:
        logging.info("%s", error)
    finally:
        await container.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # HTTP URLs for Telegram contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    asyncio.run(main())
