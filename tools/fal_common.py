"""Shared native-async FAL.ai SDK plumbing."""

from __future__ import annotations

from typing import Any


def import_fal_client() -> Any:
    """Import and return the optional ``fal_client`` SDK."""
    import fal_client  # type: ignore[import-not-found]

    return fal_client
