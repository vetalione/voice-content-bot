"""Every structured stage recovers one generation failure without restarting STT."""

import pytest

from app.services.groq_client import GroqGenerationError
from tests.conftest import FakeLLM, FakeTranscriber, make_job_request
from tests.test_pipeline import build_pipeline


@pytest.mark.parametrize(
    "stage", ["content_enrichment", "channel_teaser", "threads_editor", "reels_editor"]
)
async def test_generation_failure_retries_only_failed_stage(settings, stage):
    class FailsOnce(FakeLLM):
        def __init__(self):
            super().__init__()
            self.failed = False
            self.attempted = []

        async def chat_json(self, **kwargs):
            self.attempted.append(kwargs)
            if kwargs["label"].startswith(stage) and not self.failed:
                self.failed = True
                raise GroqGenerationError("json_validate_failed")
            return await super().chat_json(**kwargs)

    llm = FailsOnce()
    stt = FakeTranscriber()
    pipeline, _, _ = build_pipeline(settings, llm=llm, transcriber=stt)
    result = await pipeline.run(make_job_request())
    assert result.atoms and result.threads and result.reels
    attempts = [c for c in llm.attempted if c["label"].startswith(stage)]
    assert attempts[0]["label"].endswith("#1")
    assert attempts[1]["label"].endswith("#2")
    assert attempts[0]["user"] == attempts[1]["user"]
    assert attempts[0]["max_tokens"] == attempts[1]["max_tokens"]
    assert len(stt.calls) == 1


async def test_enrichment_retries_are_bounded(settings):
    class Broken(FakeLLM):
        attempts = 0

        async def chat_json(self, **kwargs):
            if kwargs["label"].startswith("content_enrichment"):
                self.attempts += 1
                raise GroqGenerationError("json_validate_failed")
            return await super().chat_json(**kwargs)

    llm = Broken()
    pipeline, _, _ = build_pipeline(settings, llm=llm)
    with pytest.raises(GroqGenerationError):
        await pipeline.run(make_job_request())
    assert llm.attempts == 2
