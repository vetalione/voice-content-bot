#!/usr/bin/env python3
"""One tiny OpenRouter request: no Whisper, Telegram, retries or paid routing."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pydantic import BaseModel

from app.agents.base import json_schema_for
from app.agents.content_miner import ContentMinerAgent
from app.config import Settings
from app.models.transcript import Transcript, TranscriptSegment
from app.services.openrouter_client import OpenRouterClient
from app.services.prompts import PromptLibrary
from app.services.usage import recording_usage


class SmokeResult(BaseModel):
    ok: bool


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="openrouter/free", help="Only free model IDs are allowed"
    )
    parser.add_argument(
        "--extraction", action="store_true", help="Extract atoms from synthetic Russian text"
    )
    args = parser.parse_args()
    # Ignore unrelated Telegram IDs and provider overrides for this text-only check.
    import os

    from dotenv import dotenv_values

    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(".env").get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env; no request was sent.")
    settings = Settings(
        _env_file=None,
        text_provider="openrouter",
        openrouter_api_key=key,
        openrouter_model=args.model,
        openrouter_allow_paid=False,
        openrouter_max_retries=0,
        openrouter_max_requests_per_recording=1,
    )
    client = OpenRouterClient(settings)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        with recording_usage(0, "text-only-smoke"):
            if args.extraction:
                texts = [
                    "Мы тестировали новый продукт. Сначала хотели потратить месяц на разработку, "
                    "но решили провести десять интервью с потенциальными клиентами. Оказалось, "
                    "что их основная проблема совершенно другая. Разговоры помогли сэкономить "
                    "месяц работы. Поэтому перед разработкой я теперь проверяю проблему, а не "
                    "спрашиваю, нравится ли человеку моя идея. Комплимент продукту не равен покупке.",
                    "На втором проекте мы сделали лендинг и предложили предзаказ. Из ста посетителей "
                    "трое оставили заявку, но никто не заплатил. Это полезнее сотни положительных "
                    "комментариев. Мы поменяли предложение, добавили конкретный результат и срок. "
                    "После этого появились первые оплаты. Проверять спрос нужно действием клиента, "
                    "а не только его словами. Даже маленький платёж показывает больше интереса.",
                    "Ещё заметил, что постоянная работа без отдыха мешает принимать решения. "
                    "Однажды я три часа исправлял ошибку и только усложнял код. После прогулки "
                    "вернулся и увидел простое решение за десять минут. Теперь в расписании есть "
                    "перерывы, даже когда срок сдачи близко. Отдых для меня стал частью работы, "
                    "а не наградой за полностью закрытый список задач.",
                ]
                transcript = Transcript(
                    duration=105,
                    segments=[
                        TranscriptSegment(start=i * 35, end=(i + 1) * 35, text=text)
                        for i, text in enumerate(texts)
                    ],
                )
                agent = ContentMinerAgent(client, PromptLibrary(settings.prompts_dir), settings)
                result = await agent.mine(transcript)
                if not result.atoms:
                    raise ValueError("No atoms extracted from synthetic source")
                print(
                    f"PASS: validated {len(result.atoms)} extracted atoms from synthetic Russian text."
                )
                return
            raw = await client.chat_json(
                system="Return a JSON object with ok=true.",
                user="Health test.",
                schema=json_schema_for(SmokeResult),
                schema_name="SmokeResult",
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
