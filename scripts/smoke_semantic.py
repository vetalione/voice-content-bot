#!/usr/bin/env python3
"""One paid-primary extraction smoke, only when explicitly configured. No STT/TG."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agents.base import StructuredAgent
from app.config import Settings
from app.models.semantic import SemanticExtraction
from app.services.prompts import PromptLibrary
from app.services.provider_routing import TextRouter
from app.services.usage import recording_usage


async def main():
    settings = Settings(
        openrouter_max_retries=0,
        openrouter_max_requests_per_recording=1,
        openrouter_allow_escalation=False,
        quality_auditor="none",
    )
    router = TextRouter(settings)
    prompts = PromptLibrary(settings.prompts_dir)
    try:
        with recording_usage(105, "semantic-smoke"):
            result = await StructuredAgent(router, prompts, settings).request(
                SemanticExtraction,
                system=prompts.render("semantic_extraction.md", target_atom_limit=10),
                user="[0-30] ИИ-агенты могут развить собственный эстетический вкус. "
                "[30-60] Тогда художники смогут делать искусство специально для агентов. "
                "[60-105] Богатые люди могут начать покупать такие работы именно потому, "
                "что их ценят агенты: предпочтение агента станет новым сигналом статуса.",
                request_label="semantic_extraction/smoke",
                repair_attempts=1,
            )
            print(f"Validated {len(result.atoms)} atoms. Inspect claims:")
            for atom in result.atoms:
                print(f"{atom.kind}: {atom.claim}")
    finally:
        await router.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
