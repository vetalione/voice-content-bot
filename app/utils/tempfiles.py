"""Scratch directory lifecycle for downloads and audio chunks."""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def workspace(root: Path, prefix: str = "job", keep: bool = False) -> Iterator[Path]:
    """Create an isolated directory and always remove it afterwards.

    ``keep=True`` leaves the files behind for debugging.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        base = root
    except OSError:  # pragma: no cover - read-only filesystem fallback
        base = Path(tempfile.gettempdir())
        logger.warning("Cannot use work dir %s, falling back to %s", root, base)

    path = base / f"{prefix}-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if keep:
            logger.info("Keeping temp workspace %s", path)
        else:
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.debug("Removed temp workspace %s", path)
            except OSError:  # pragma: no cover
                logger.exception("Failed to clean temp workspace %s", path)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
