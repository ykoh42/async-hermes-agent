"""Spill oversized plugin-hook context to disk with a bounded preview."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 10_000
DEFAULT_PREVIEW_HEAD = 500
DEFAULT_PREVIEW_TAIL = 500
DEFAULT_ENABLED = True


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return default
    return converted if converted > 0 else default


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return default
    return converted if converted >= 0 else default


async def get_spill_config() -> Dict[str, Any]:
    """Resolve output-spill configuration without blocking or raising."""
    section: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        hooks = config.get("hooks") if isinstance(config, dict) else None
        candidate = hooks.get("output_spill") if isinstance(hooks, dict) else None
        if isinstance(candidate, dict):
            section = candidate
    except asyncio.CancelledError:
        raise
    except Exception:
        pass

    enabled_raw = section.get("enabled", DEFAULT_ENABLED)
    directory = section.get("directory")
    if directory is not None and not isinstance(directory, str):
        directory = None
    return {
        "enabled": (
            bool(enabled_raw) if enabled_raw is not None else DEFAULT_ENABLED
        ),
        "max_chars": _coerce_positive_int(
            section.get("max_chars"), DEFAULT_MAX_CHARS
        ),
        "preview_head": _coerce_non_negative_int(
            section.get("preview_head"), DEFAULT_PREVIEW_HEAD
        ),
        "preview_tail": _coerce_non_negative_int(
            section.get("preview_tail"), DEFAULT_PREVIEW_TAIL
        ),
        "directory": directory,
    }


def _resolve_spill_dir(
    directory_override: Optional[str],
    session_id: Optional[str],
) -> Path:
    if directory_override:
        base = Path(os.path.expanduser(directory_override))
    else:
        from hermes_constants import get_hermes_home

        base = get_hermes_home() / "hook_outputs"
    session_segment = (session_id or "no-session").replace("/", "_")
    session_segment = session_segment.replace("\\", "_").replace("..", "_")
    return base / session_segment


def _build_preview(
    text: str,
    head: int,
    tail: int,
    saved_path: Optional[str],
    *,
    source: str,
) -> str:
    total = len(text)
    parts = [
        f"[{source} output truncated — {total:,} chars; full content "
        + (
            f"saved to {saved_path}]"
            if saved_path
            else "unavailable — spill write failed]"
        )
    ]
    if head > 0:
        parts.extend(("--- head ---", text[:head]))
    if tail > 0 and total > head:
        parts.extend(("--- tail ---", text[-tail:]))
    return "\n".join(parts)


async def spill_if_oversized(
    text: str,
    *,
    session_id: Optional[str] = None,
    source: str = "hook",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Return text unchanged under the cap, otherwise persist and preview it."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    resolved = config if config is not None else await get_spill_config()
    if not resolved.get("enabled", True):
        return text
    max_chars = int(resolved.get("max_chars") or DEFAULT_MAX_CHARS)
    if len(text) <= max_chars:
        return text

    saved_path: Optional[str] = None
    try:
        spill_dir = _resolve_spill_dir(resolved.get("directory"), session_id)
        await aiofiles.os.makedirs(spill_dir, exist_ok=True)
        path = spill_dir / f"{uuid.uuid4().hex}.txt"
        async with aiofiles.open(path, "w", encoding="utf-8") as handle:
            await handle.write(text if text.endswith("\n") else text + "\n")
        saved_path = str(path)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("hook output spill failed: %s", exc)

    return _build_preview(
        text,
        int(resolved.get("preview_head") or 0),
        int(resolved.get("preview_tail") or 0),
        saved_path,
        source=source,
    )


__all__ = [
    "DEFAULT_ENABLED",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_PREVIEW_HEAD",
    "DEFAULT_PREVIEW_TAIL",
    "get_spill_config",
    "spill_if_oversized",
]
