"""In-process duplicate-update protection.

Telegram redelivers a webhook update if it does not receive a prompt 200, and a
retried delivery must not trigger a second transcription. v1 has no database, so
this is a bounded TTL set living in the worker process.

Limitation (documented in the README): the guard is lost on restart, redeploy or
a Render Free spin-down. Because the same set is also used to key the seen
``update_id``s, a cold start immediately after a delivery could in principle
reprocess one message. Swapping this class for a Redis/Postgres-backed
implementation is the intended upgrade path — everything else depends only on
the :class:`DedupeStore` interface.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DedupeStore(Protocol):
    """Minimal interface a persistent implementation would need to satisfy."""

    def claim(self, key: str) -> bool:
        """Atomically mark ``key`` as seen. Returns False if already claimed."""

    def seen(self, key: str) -> bool: ...

    def release(self, key: str) -> None:
        """Forget ``key`` so a failed job can be retried."""


class TTLDedupeStore:
    """Bounded, time-limited set of processed keys."""

    def __init__(
        self,
        ttl_seconds: float = 6 * 3600.0,
        max_entries: int = 2048,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._clock = clock
        self._entries: OrderedDict[str, float] = OrderedDict()

    def _purge(self) -> None:
        now = self._clock()
        expired = [key for key, ts in self._entries.items() if now - ts > self._ttl]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def claim(self, key: str) -> bool:
        self._purge()
        if key in self._entries:
            logger.info("Duplicate update ignored: %s", key)
            return False
        self._entries[key] = self._clock()
        return True

    def seen(self, key: str) -> bool:
        self._purge()
        return key in self._entries

    def release(self, key: str) -> None:
        self._entries.pop(key, None)

    def __len__(self) -> int:  # pragma: no cover - diagnostics only
        self._purge()
        return len(self._entries)
