"""Compatibility namespace for the deferred scheduler subsystem.

The training runtime does not start scheduled jobs.  The scheduler
implementation is intentionally kept out of this checkout's runtime surface,
but the package path remains so an upstream update can restore it without a
module rename.  Callers that still try to start cron receive a clear error
instead of a transitive import failure during normal agent startup.
"""

from __future__ import annotations


def tick(*_args, **_kwargs) -> None:
    """Fail explicitly when the deferred scheduler is invoked."""
    raise RuntimeError(
        "Hermes cron scheduling is not included in the training runtime; "
        "restore the cron scheduler package before enabling scheduled jobs."
    )


__all__ = ["tick"]
