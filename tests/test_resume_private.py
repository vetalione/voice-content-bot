from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.media import SourceMode
from app.services.checkpoints import SupabaseStore, recording_id
from scripts import resume_private
from tests.conftest import make_job_request


@pytest.mark.parametrize("mode", ["inspect", "run", "cache_mismatch", "channel"])
async def test_private_resume_guards_and_inspection(settings, monkeypatch, mode):
    request = make_job_request(SourceMode.PRIVATE).model_copy(
        update={"chat_id": settings.owner_telegram_id}
    )
    if mode == "channel":
        request = request.model_copy(update={"mode": SourceMode.CHANNEL})
    job = recording_id(request)
    data = {
        "request": request.model_dump(mode="json"),
        "transcript": {
            "transcript": {
                "segments": [{"start": 0, "end": 463, "text": "Saved words"}],
                "duration": 463,
            },
            "chunks": 1,
        },
    }

    class Store(SupabaseStore):
        def __init__(self):
            self.put = AsyncMock()

        async def get(self, job, stage):
            return data.get(stage)

        async def _request(self, *args, **kwargs):
            rows = (
                [{"data": {"stage": "threads_editor/atom=a1", "status": "complete"}}]
                if mode == "cache_mismatch"
                else []
            )
            return SimpleNamespace(json=lambda: rows)

    store = Store()
    original_request = AsyncMock(return_value="Draft")
    router = SimpleNamespace(request=original_request)
    result = SimpleNamespace(model_dump=lambda **kwargs: {})

    async def analyse(request, transcript, chunks, started):
        assert transcript.text == "Saved words" and chunks == 1
        await router.request("primary", method="chat_text", label="threads_editor/atom=a1")
        return result

    pipeline = SimpleNamespace(checkpoints=store, analyse=AsyncMock(side_effect=analyse))
    processor = SimpleNamespace(_check_budget=AsyncMock(), _deliver=AsyncMock())
    container = SimpleNamespace(
        text=router, pipeline=pipeline, processor=processor, shutdown=AsyncMock()
    )
    monkeypatch.setattr(resume_private, "Settings", lambda **kw: settings)
    monkeypatch.setattr(resume_private, "build_container", lambda s: container)
    argv = ["resume_private", "--recording", job, "--primary-model", "test:free"]
    if mode != "inspect":
        argv.append("--run")
    monkeypatch.setattr("sys.argv", argv)
    if mode in ("cache_mismatch", "channel"):
        with pytest.raises(resume_private.ResumeBlocked):
            await resume_private.main()
    else:
        await resume_private.main()
    container.shutdown.assert_awaited_once()
    if mode == "run":
        original_request.assert_awaited_once()
        processor._deliver.assert_awaited_once_with(request, result)
        assert store.put.call_args.args == (job, "status", {"status": "complete"})
    else:
        original_request.assert_not_awaited()
        processor._deliver.assert_not_awaited()
        store.put.assert_not_awaited()
