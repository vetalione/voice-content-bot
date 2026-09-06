"""FastAPI entrypoint.

The webhook endpoint is deliberately trivial: parse the update, dispatch it to
aiogram (whose handlers only enqueue work) and return 200 immediately. Telegram
never waits for transcription.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from aiogram.types import Update
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.container import Container, build_container
from app.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.container = None
        application.state.startup_error = ""
        container: Container | None = None
        try:
            container = build_container(settings)
        except Exception as error:
            # A bad/missing BOT_TOKEN would otherwise crash-loop the service with
            # no way to inspect it. Come up degraded instead, so /health explains
            # the problem and the webhook answers 503 rather than timing out.
            application.state.startup_error = f"{type(error).__name__}: {error}"
            logger.exception("Startup failed; serving /health in degraded mode")
            yield
            return

        application.state.container = container
        await container.startup()
        logger.info("Application started; webhook path is %s", settings.webhook_path)
        try:
            yield
        finally:
            await container.shutdown()
            logger.info("Application stopped")

    app = FastAPI(
        title="Voice Content Bot",
        version="1.0.0",
        summary="Turns Telegram voice posts into a channel teaser, Threads posts and Reels scripts.",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def container_of(request: Request) -> Container:
        container: Container | None = getattr(request.app.state, "container", None)
        if container is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=getattr(request.app.state, "startup_error", "")
                or "Application is still starting",
            )
        return container

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Liveness probe. Also handy for keeping Render Free warm."""
        container: Container | None = getattr(request.app.state, "container", None)
        startup_error = getattr(request.app.state, "startup_error", "")
        stats = container.runner.stats if container else None
        logger.warning(
            "Health probe pid=%s uptime=%s running=%s",
            os.getpid(),
            container.runner.diagnostics()["uptime_seconds"] if container else None,
            stats.running if stats else 0,
        )
        return {
            "runtime": {
                "revision": os.getenv("RENDER_GIT_COMMIT", "unknown"),
                "event_loop": type(asyncio.get_running_loop()).__module__,
                "unix_time": time.time(),
            },
            "tpm_diagnostics": {
                model: limiter.diagnostics() for model, limiter in container.groq._tpm.items()
            }
            if container
            else {},
            "status": "ok" if container is not None else "degraded",
            "text_provider_health": getattr(container.text, "health", {"status": "legacy"})
            if container
            else None,
            "checkpoint_backend": settings.checkpoint_backend,
            "configured": not settings.missing_required(),
            "missing_env": settings.missing_required(),
            "startup_error": startup_error,
            "ffmpeg": bool(container and container.audio.ffmpeg_available),
            "worker_diagnostics": container.runner.diagnostics() if container else None,
            "queue": {
                "queued": stats.queued if stats else 0,
                "running": stats.running if stats else 0,
                "completed": stats.completed if stats else 0,
                "failed": stats.failed if stats else 0,
            },
        }

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "voice-content-bot", "health": "/health"}

    @app.post(settings.webhook_path)
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> JSONResponse:
        container = container_of(request)

        if settings.webhook_secret and x_telegram_bot_api_secret_token != settings.webhook_secret:
            logger.warning("Rejected webhook call with a bad secret token")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

        try:
            payload = await request.json()
        except Exception as error:
            logger.warning("Malformed webhook body: %s", error)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="invalid json"
            ) from error

        try:
            update = Update.model_validate(payload, context={"bot": container.bot})
        except Exception as error:
            # Never 4xx an update we cannot parse: Telegram would retry forever.
            logger.warning("Unparseable update ignored: %s", error)
            return JSONResponse({"ok": True, "ignored": "unparseable"})

        # Duplicate guard at the update level, in addition to the per-message
        # guard inside the handlers.
        if not container.dedupe.claim(f"update:{update.update_id}"):
            logger.info("Duplicate update_id %s ignored", update.update_id)
            return JSONResponse({"ok": True, "duplicate": True})

        try:
            # Handlers only validate + enqueue, so this returns in milliseconds.
            await container.dispatcher.feed_update(container.bot, update)
        except Exception:
            logger.exception("Update %s failed during dispatch", update.update_id)

        return JSONResponse({"ok": True})

    return app


app = create_app()
