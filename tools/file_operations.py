"""Pure helpers shared by the native async file-tool handlers.

The former shell-backed ``FileOperations`` hierarchy was a synchronous
compatibility layer for terminal backends that async-hermes-agent no longer
ships.  Keeping these small transformations here preserves the upstream file
location while leaving filesystem ownership with :mod:`tools.file_tools`.
"""

from __future__ import annotations

from typing import Any


MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 500

_UTF8_BOM = "\ufeff"


def _detect_line_ending(sample: str) -> str | None:
    """Return the line ending used by *sample*, if it contains one."""
    if not sample:
        return None
    return "\r\n" if "\r\n" in sample else "\n" if "\n" in sample else None


def _normalize_line_endings(text: str, target: str | None) -> str:
    """Normalize newlines to *target* while preserving text content."""
    if not target:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if target == "\n" else normalized.replace("\n", target)


def _strip_bom(text: str) -> tuple[str, bool]:
    """Return ``(text_without_leading_bom, had_bom)``."""
    if text.startswith(_UTF8_BOM):
        return text[len(_UTF8_BOM):], True
    return text, False


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(
    offset: Any = DEFAULT_READ_OFFSET,
    limit: Any = DEFAULT_READ_LIMIT,
) -> tuple[int, int]:
    """Return schema-safe bounds for ``read_file`` pagination."""
    from tools.tool_output_limits import get_max_lines

    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = max(1, _coerce_int(limit, DEFAULT_READ_LIMIT))
    return normalized_offset, min(normalized_limit, get_max_lines())
