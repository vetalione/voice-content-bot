"""Exercise real asyncio scheduling and safe worker inspection."""

import asyncio
import time

import pytest

from app.services.jobs import JobRunner
from app.services.tpm import RollingTPM


async def test_real_timer_resumes_and_job_diagnostics_are_safe():
    runner = JobRunner()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def job():
        entered.set()
        await release.wait()

    await runner.start()
    try:
        runner.submit("private job identifier", job)
        await asyncio.wait_for(entered.wait(), 1)
        info = runner.diagnostics()
        assert info["pid"] > 0
        assert any(t["name"] == "job-worker-0" and "job" in t["await_chain"] for t in info["tasks"])
        assert "private job identifier" not in str(info)
        scheduler = RollingTPM(8000, clock=time.monotonic, sleep=asyncio.sleep)
        scheduler.events.append((time.monotonic() - 59.98, 7000))
        await asyncio.wait_for(
            scheduler.reserve(2000, label="real-timer", attempt=1, input_tokens=500, output=1500), 1
        )
        release.set()
        assert await runner.drain(timeout=1)
        assert not any(
            t["name"].startswith("job-heartbeat-") for t in runner.diagnostics()["tasks"]
        )
    finally:
        release.set()
        await runner.stop()


def test_extraction_budget_accepts_1600(settings):
    from app.config import Settings

    assert settings.groq_extraction_max_tokens == 1500
    assert (
        Settings(_env_file=None, groq_extraction_max_tokens=1600).groq_extraction_max_tokens == 1600
    )
    with pytest.raises(ValueError):
        Settings(_env_file=None, groq_extraction_max_tokens=1601)
