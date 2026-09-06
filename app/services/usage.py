"""Per-recording usage, isolated across concurrent jobs and logged on failure too."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TextUsage:
    events: list[dict] = field(default_factory=list)
    claude_requests: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reported_cost: float = 0.0
    cost_reports: int = 0
    models: set[str] = field(default_factory=set)


current_usage: ContextVar[TextUsage | None] = ContextVar("text_usage", default=None)


@contextmanager
def recording_usage(duration: float, label: str):
    usage = TextUsage()
    token = current_usage.set(usage)
    try:
        yield usage
    finally:
        logger.info(
            "Recording usage recording=%s duration_seconds=%s openrouter_requests=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s models=%s reported_cost=%s cost_reports=%s",
            label,
            duration,
            usage.requests,
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            sorted(usage.models),
            usage.reported_cost if usage.cost_reports else None,
            usage.cost_reports,
        )
        current_usage.reset(token)


async def capture_usage(event):
    """Persist provider metadata even when the later JSON validation fails."""
    from app.services.checkpoints import active_checkpoint

    if event.get("provider") == "claude_agent_sdk":
        logger.info("Claude SDK usage: %s", event)
    context = active_checkpoint.get()
    usage = current_usage.get()
    if usage is not None:
        usage.events.append(event)
    if context:
        store, job = context
        saved = await store.get(job, "usage") or {"events": []}
        saved["events"].append(event)
        await store.put(job, "usage", saved)


async def restore_usage(store, job, usage):
    prior = await store.get(job, "usage") or {"events": []}
    usage.events = list(prior.get("events", []))
    prior_or = [e for e in usage.events if e.get("provider") == "openrouter"]
    attempted = await store.get(job, "openrouter_attempts") or {}
    usage.requests = max(len(prior_or), attempted.get("count", 0))
    usage.claude_requests = sum(e.get("provider") == "claude_agent_sdk" for e in usage.events)
    usage.input_tokens = sum(e.get("input_tokens") or 0 for e in prior_or)
    usage.output_tokens = sum(e.get("output_tokens") or 0 for e in prior_or)
    usage.reasoning_tokens = sum(e.get("reasoning_tokens") or 0 for e in prior_or)
    usage.models = {e["actual_model"] for e in prior_or if e.get("actual_model")}
    costs = [
        e["reported_cost"] for e in prior_or if isinstance(e.get("reported_cost"), int | float)
    ]
    usage.reported_cost = sum(costs)
    usage.cost_reports = len(costs)


async def persist_attempt_count(provider, count):
    """Reserve attempts before sending, including failures without usage metadata."""
    from app.services.checkpoints import active_checkpoint

    context = active_checkpoint.get()
    if context:
        await context[0].put(context[1], provider + "_attempts", {"count": count})
