"""Conservative estimates for admission control, not tokenizer-exact billing."""

import json
import math


def approximate_tokens(value: object) -> int:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    return math.ceil(len(text.encode("utf-8")) / 3)


def request_tokens(system: str, user: str, schema: dict | None) -> int:
    return (
        64
        + approximate_tokens(system)
        + approximate_tokens(user)
        + approximate_tokens(schema or {})
    )
