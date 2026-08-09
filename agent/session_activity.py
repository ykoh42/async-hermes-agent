"""Shared session activity observation contract (#72016 / #72039).

Observation-only: timestamp + bounded description/provenance.
Notification, timeout, kill, and retry policy stay in their own components.
Consumers distinguish work (API / tool / compacting / stalled) from the
description text itself — there is no separate phase enum.

Provenance is a small closed enum of *noun* sources (where the stamp came
from). The default agent activity clock (``_touch_activity``) stamps
``unknown`` unless a caller passes an explicit ``provenance=``; named
values are for special writers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional

ACTIVITY_DESCRIPTION_MAX = 120

# Durable SessionDB activity heartbeat cadence (seconds between writes per
# session). Contract: MUST stay >= 30s — the SessionDB write path is
# contended, and the heartbeat is an observation-only projection that never
# justifies extra write pressure. force_persist (terminal stamps) is the only
# bypass.
SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0


class ActivityProvenance(str, Enum):
    """Where a durable/in-memory activity stamp came from."""

    UNKNOWN = "unknown"
    AGENT_COMPRESSION = "agent.compression"
    AGENT_COMPRESSION_TIMEOUT = "agent.compression_timeout"
    AGENT_COMPRESSION_COOLDOWN = "agent.compression_cooldown"


def bound_activity_description(description: Optional[str]) -> str:
    """Clamp free-form activity text to the shared description budget."""
    text = (description or "").strip()
    if len(text) <= ACTIVITY_DESCRIPTION_MAX:
        return text
    return text[: ACTIVITY_DESCRIPTION_MAX - 1] + "…"


def normalize_activity_provenance(
    provenance: Optional[ActivityProvenance | str],
) -> ActivityProvenance:
    """Return a known provenance, or ``UNKNOWN`` when unset/unrecognized."""
    if isinstance(provenance, ActivityProvenance):
        return provenance
    value = (provenance or "").strip()
    try:
        return ActivityProvenance(value)
    except ValueError:
        return ActivityProvenance.UNKNOWN


def reset_session_activity_persist_window(agent: Any) -> None:
    """Let the next activity projection bypass its cadence window."""
    try:
        agent._session_activity_last_persist_mono = 0.0
    except Exception:
        pass


def build_activity_snapshot(
    *,
    last_activity_at: Optional[float],
    last_activity_description: Optional[str],
    last_activity_provenance: Optional[ActivityProvenance | str] = None,
    now: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the shared activity snapshot plus compatibility aliases."""
    import time as _time

    when = float(last_activity_at) if last_activity_at is not None else None
    clock = float(now if now is not None else _time.time())
    desc = bound_activity_description(last_activity_description)
    prov = normalize_activity_provenance(last_activity_provenance)
    elapsed = round(clock - when, 1) if when is not None else None
    snapshot: dict[str, Any] = {
        "last_activity_at": when,
        "last_activity_description": desc,
        "last_activity_provenance": prov.value,
        "seconds_since_activity": elapsed,
        "last_activity_ts": when,
        "last_activity_desc": desc,
        "description": desc,
        "provenance": prov.value,
    }
    if extra:
        snapshot.update(dict(extra))
    return snapshot
