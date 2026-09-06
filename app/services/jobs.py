"""In-process background job runner.

Why not Celery/RQ/Redis: the cost brief forbids paid infrastructure, and this is
a single-user bot. A bounded ``asyncio.Queue`` plus N worker tasks inside the
same FastAPI process is enough, and it keeps the webhook response instant.

Trade-offs, all documented in the README:
  * jobs live in memory — a redeploy or a Render Free spin-down loses the queue;
  * Render Free has no always-on guarantee, so a job can be killed mid-flight;
  * throughput is capped by ``JOB_CONCURRENCY`` (default 1) to stay inside the
    Groq free-tier rate limits.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

JobCallable = Callable[[], Awaitable[None]]


class QueueFullError(RuntimeError):
    """Raised when the runner cannot accept more work."""


@dataclass(slots=True)
class _Job:
    name: str
    run: JobCallable
    on_error: Callable[[BaseException], Awaitable[None]] | None = None
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class RunnerStats:
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0


class JobRunner:
    """Fire-and-forget async worker pool with a bounded queue."""

    def __init__(self, concurrency: int = 1, queue_size: int = 16) -> None:
        self._concurrency = max(1, concurrency)
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=queue_size)
        self._workers: list[asyncio.Task[None]] = []
        self._stats = RunnerStats()
        self._running = False

    @property
    def stats(self) -> RunnerStats:
        self._stats.queued = self._queue.qsize()
        return self._stats

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"job-worker-{index}")
            for index in range(self._concurrency)
        ]
        logger.info("Job runner started with %s worker(s)", self._concurrency)

    async def stop(self, timeout: float = 5.0) -> None:
        """Signal workers to finish and wait briefly for the current job."""
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            # A full queue on shutdown is fine: the timeout below still cancels.
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)
        if self._workers:
            done, pending = await asyncio.wait(self._workers, timeout=timeout)
            for task in pending:
                task.cancel()
            logger.info("Job runner stopped (%s finished, %s cancelled)", len(done), len(pending))
        self._workers = []

    def submit(
        self,
        name: str,
        run: JobCallable,
        on_error: Callable[[BaseException], Awaitable[None]] | None = None,
    ) -> None:
        """Enqueue work without blocking the caller (the webhook request)."""
        job = _Job(name=name, run=run, on_error=on_error)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as error:
            raise QueueFullError(
                f"Job queue is full ({self._queue.maxsize} items); dropped {name}"
            ) from error
        logger.info("Queued job %s (depth=%s)", name, self._queue.qsize())

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                self._stats.running += 1
                started = time.monotonic()
                logger.info(
                    "Worker %s picked up %s (waited %.1fs)",
                    index,
                    job.name,
                    started - job.submitted_at,
                )
                try:
                    await job.run()
                    self._stats.completed += 1
                    logger.info("Job %s finished in %.1fs", job.name, time.monotonic() - started)
                except asyncio.CancelledError:  # pragma: no cover
                    logger.warning("Job %s cancelled", job.name)
                    raise
                except BaseException as error:
                    self._stats.failed += 1
                    logger.exception("Job %s failed: %s", job.name, error)
                    if job.on_error is not None:
                        try:
                            await job.on_error(error)
                        except Exception:
                            logger.exception("Error handler for %s failed", job.name)
                finally:
                    self._stats.running -= 1
            finally:
                self._queue.task_done()

    async def drain(self, timeout: float = 30.0) -> bool:
        """Wait for the queue to empty. Used by tests, not by production code."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except TimeoutError:  # pragma: no cover
            return False
