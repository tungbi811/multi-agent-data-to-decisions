import json
from collections.abc import Iterator
from typing import Any


KNOWN_EVENT_TYPES = {
    "text",
    "tool_call",
    "tool_response",
    "input_request",
    "run_completion",
}


def next_or_none(events: Iterator[Any]) -> tuple[Any | None, bool]:
    try:
        return next(events), False
    except StopIteration:
        return None, True


def coder_code_from_tool_call(event: Any) -> str | None:
    try:
        if event.content.sender != "Coder":
            return None
        arguments = json.loads(event.content.tool_calls[0].function.arguments)
        code = arguments.get("code")
        return code if isinstance(code, str) else None
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
        return None


def safe_event_type(event: Any) -> str:
    event_type = getattr(event, "type", "unknown")
    return (
        event_type if isinstance(event_type, str) and event_type in KNOWN_EVENT_TYPES else "unknown"
    )
