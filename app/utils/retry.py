"""Bounded async retry with exponential backoff and jitter.

Deliberately conservative: Groq free-tier rate limits are the main reason we
retry at all, and we never fall back to a different (paid) provider.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Raised by callers to signal that another attempt may succeed.

    ``retry_after`` mirrors the HTTP ``Retry-After`` header when available.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    label: str = "operation",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``operation``, retrying only on :class:`RetryableError`."""
    total = max(1, attempts)
    last_error: RetryableError | None = None
    for attempt in range(1, total + 1):
        try:
            return await operation()
        except RetryableError as error:
            last_error = error
            if attempt >= total:
                break
            delay = error.retry_after
            if delay is None:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0, base_delay / 2)
            delay = max(0.0, float(delay))
            logger.warning(
                "%s failed (attempt %s/%s): %s — retrying in %.1fs",
                label,
                attempt,
                total,
                error,
                delay,
            )
            await sleep(delay)
    assert last_error is not None
    raise last_error
