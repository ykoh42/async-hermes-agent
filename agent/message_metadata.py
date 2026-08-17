"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from collections.abc import MutableMapping
from time import time as wall_time
from typing import Any, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp"})

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: float | None = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time."""
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: float | None = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    stamp_message_timestamp(message, timestamp=timestamp)
    messages.append(message)
    return message
