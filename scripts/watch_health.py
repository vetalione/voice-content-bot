#!/usr/bin/env python3
"""External health monitor. Run on an awake computer, not inside Render itself."""

import argparse
import fcntl
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger("health_watch")


def probe(url):
    try:
        response = httpx.get(url, timeout=45, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        workers = data.get("worker_diagnostics") or {}
        queue = data.get("queue") or {}
        logger.info(
            "status=%s uptime=%s running=%s queued=%s completed=%s failed=%s",
            data.get("status"),
            workers.get("uptime_seconds"),
            queue.get("running"),
            queue.get("queued"),
            queue.get("completed"),
            queue.get("failed"),
        )
    except (httpx.HTTPError, TimeoutError, ValueError, OSError) as error:
        logger.warning("Health check failed (%s); will retry next interval", type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Public HTTPS /health URL; no credentials")
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--lock-file", type=Path, default=Path("/tmp/voice-content-health-watch.lock")
    )
    args = parser.parse_args()
    if not args.url.startswith("https://") or args.interval < 30:
        parser.error("Use HTTPS and an interval of at least 30 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with args.lock_file.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(message="A health monitor is already running.\n")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        logger.info("Monitor started pid=%s interval_seconds=%s", os.getpid(), args.interval)
        while True:
            started = time.monotonic()
            probe(args.url)
            if args.once:
                return
            time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
