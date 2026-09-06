"""Logging configuration kept in one place."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.setLevel(level.upper())
    if not _CONFIGURED:
        root.handlers = [handler]
        _CONFIGURED = True
    else:
        for existing in root.handlers:
            existing.setFormatter(logging.Formatter(_FORMAT))

    # These are chatty and rarely useful at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
