"""Conservative in-process rolling TPM admission, including failed attempts.

One shared GroqClient is used by the pipeline. Multiple processes/other API users
still require provider headers and 429 handling; this is not a distributed quota.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque

logger = logging.getLogger(__name__)


def duration_seconds(raw: str | None) -> float:
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        parts = re.findall(r"([\d.]+)(ms|s|m|h)", raw)
        return sum(float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit] for n, unit in parts)


class RollingTPM:
    def __init__(self, limit: int, *, clock=None, sleep=None):
        self.limit = limit
        self.clock = clock or time.monotonic
        self.sleep = sleep or asyncio.sleep
        self.events: deque[tuple[float, int]] = deque()
        self.lock = asyncio.Lock()
        self.blocked_until = 0.0
        self.remote_remaining: int | None = None
        self.remote_until = 0.0
        self.wait_state: dict = {}
        self.wake_count = 0

    async def reserve(
        self, tokens: int, *, label: str, attempt: int, input_tokens: int, output: int
    ):
        if tokens > self.limit:
            raise ValueError("Single request exceeds TPM capacity")
        async with self.lock:
            while True:
                now = self.clock()
                while self.events and self.events[0][0] + 60 <= now:
                    self.events.popleft()
                used = sum(n for _, n in self.events)
                wait = max(0.0, self.blocked_until - now)
                if used + tokens > self.limit:
                    remaining = used
                    for stamp, size in self.events:
                        remaining -= size
                        if remaining + tokens <= self.limit:
                            wait = max(wait, stamp + 60 - now)
                            break
                if (
                    self.remote_until > now
                    and self.remote_remaining is not None
                    and tokens > self.remote_remaining
                ):
                    wait = max(wait, self.remote_until - now)
                logger.info(
                    "LLM schedule stage/window=%s attempt=%s input_tokens=%s output_budget=%s rolling_tpm_estimate=%s wait_before_request=%.2fs",
                    label,
                    attempt,
                    input_tokens,
                    output,
                    used,
                    wait,
                )
                if wait > 0:
                    interval = min(wait, 10.0)
                    loop = asyncio.get_running_loop()
                    self.wait_state = {
                        "started_monotonic": self.clock(),
                        "started_loop_time": loop.time(),
                        "sleep_seconds": interval,
                        "requested_wait_seconds": wait,
                    }
                    await self.sleep(interval)
                    self.wake_count += 1
                    self.wait_state = {}
                    continue
                reservation = (self.clock(), tokens)
                self.events.append(reservation)
                if self.remote_until > now and self.remote_remaining is not None:
                    self.remote_remaining = max(0, self.remote_remaining - tokens)
                return reservation

    def diagnostics(self):
        now = self.clock()
        state = dict(self.wait_state)
        if state:
            state["elapsed_monotonic"] = round(now - state["started_monotonic"], 3)
            state["elapsed_loop_time"] = round(
                asyncio.get_running_loop().time() - state["started_loop_time"], 3
            )
        return {
            "wake_count": self.wake_count,
            "waiting": state,
            "lock_held": self.lock.locked(),
            "cooldown_remaining": max(0, self.blocked_until - now),
            "remote_reset_remaining": max(0, self.remote_until - now),
            "remote_remaining": self.remote_remaining,
        }

    def settle(self, reservation, usage):
        """Replace a successful reservation with reported usage; errors retain it."""
        if not isinstance(usage, dict):
            return
        actual = usage.get("total_tokens")
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
            return
        for index, event in enumerate(self.events):
            if event is reservation:
                self.events[index] = (event[0], actual)
                logger.info("TPM usage reconciled reserved=%s actual=%s", event[1], actual)
                return

    def observe(self, headers, *, generation_failed=False):
        now = self.clock()
        reset = duration_seconds(headers.get("x-ratelimit-reset-tokens"))
        try:
            self.remote_remaining = int(headers["x-ratelimit-remaining-tokens"])
            self.remote_until = now + (reset or 60.0)
        except (KeyError, ValueError):
            pass
        # Keep reservations on errors. A failed generation is never free capacity.
        if generation_failed:
            retry = duration_seconds(headers.get("retry-after"))
            self.blocked_until = max(self.blocked_until, now + max(60.0, reset, retry))
