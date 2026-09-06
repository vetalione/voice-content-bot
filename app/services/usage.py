"""Per-recording usage, isolated across concurrent jobs and logged on failure too."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TextUsage:
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
