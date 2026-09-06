"""Small persistent stage store. No credentials, audio blobs or hidden fallbacks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol

import httpx


class CheckpointError(RuntimeError):
    pass


class CheckpointStore(Protocol):
    async def get(self, job: str, stage: str) -> dict | None: ...
    async def put(self, job: str, stage: str, data: dict) -> None: ...
    async def aclose(self) -> None: ...


class NullStore:
    async def get(self, job, stage):
        return None

    async def put(self, job, stage, data):
        pass

    async def aclose(self):
        pass


class SQLiteStore:
    def __init__(self, path):
        self.path = Path(path)

    def _run(self, job, stage, data=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=15) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints (job TEXT, stage TEXT, data TEXT NOT NULL, PRIMARY KEY(job,stage))"
            )
            if data is None:
                row = db.execute(
                    "SELECT data FROM checkpoints WHERE job=? AND stage=?", (job, stage)
                ).fetchone()
                return json.loads(row[0]) if row else None
            db.execute(
                "INSERT INTO checkpoints VALUES (?,?,?) ON CONFLICT(job,stage) DO UPDATE SET data=excluded.data",
                (job, stage, json.dumps(data, ensure_ascii=False)),
            )

    async def get(self, job, stage):
        return await asyncio.to_thread(self._run, job, stage)

    async def put(self, job, stage, data):
        await asyncio.to_thread(self._run, job, stage, data)

    async def aclose(self):
        pass


class SupabaseStore:
    def __init__(self, url, key, client=None):
        if not url.startswith("https://") or not key:
            raise CheckpointError("Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        self.client = client or httpx.AsyncClient(
            base_url=url.rstrip("/") + "/rest/v1/", timeout=30
        )
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    async def _request(self, method, **kwargs):
        try:
            response = await self.client.request(
                method, "voice_checkpoints", headers=self.headers, **kwargs
            )
        except httpx.HTTPError as error:
            raise CheckpointError(f"Checkpoint network error: {type(error).__name__}") from None
        if response.is_error:
            raise CheckpointError(
                f"Checkpoint store HTTP {response.status_code}; check migration and service credentials"
            )
        return response

    async def get(self, job, stage):
        response = await self._request(
            "GET", params={"job": f"eq.{job}", "stage": f"eq.{stage}", "select": "data"}
        )
        rows = response.json()
        return rows[0]["data"] if rows else None

    async def put(self, job, stage, data):
        # PUT is expressed through PostgREST upsert.
        try:
            response = await self.client.post(
                "voice_checkpoints",
                params={"on_conflict": "job,stage"},
                headers={**self.headers, "Prefer": "resolution=merge-duplicates"},
                json={"job": job, "stage": stage, "data": data},
            )
            if response.is_error:
                raise CheckpointError(f"Checkpoint write HTTP {response.status_code}")
        except httpx.HTTPError as error:
            raise CheckpointError(f"Checkpoint write failed: {type(error).__name__}") from None

    async def aclose(self):
        await self.client.aclose()


def build_store(settings):
    if settings.checkpoint_backend == "supabase":
        return SupabaseStore(settings.supabase_url, settings.supabase_service_role_key)
    if settings.checkpoint_backend == "sqlite":
        return SQLiteStore(settings.checkpoint_sqlite_path)
    return NullStore()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def recording_id(request):
    return fingerprint([request.mode.value, request.chat_id, request.media.file_unique_id])[:32]


active_checkpoint: ContextVar[tuple | None] = ContextVar("active_checkpoint", default=None)


@contextmanager
def checkpoint_scope(store, job):
    token = active_checkpoint.set((store, job))
    try:
        yield
    finally:
        active_checkpoint.reset(token)
