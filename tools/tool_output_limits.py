"""Configurable tool-output truncation limits.

Ported from anomalyco/opencode PR #23770 (``feat(truncate): allow
configuring tool output truncation limits``).

OpenCode hardcoded ``MAX_LINES = 2000`` and ``MAX_BYTES = 50 * 1024``
as tool-output truncation thresholds. Hermes-agent had the same
hardcoded constants in two places:

* ``tools/terminal_tool.py`` — ``MAX_OUTPUT_CHARS = 50000`` (terminal
  stdout/stderr cap)
* ``tools/file_operations.py`` — ``MAX_LINES = 2000`` /
  ``MAX_LINE_LENGTH = 2000`` (read_file pagination cap + per-line cap)

This module centralises those values behind a single config section
(``tool_output`` in ``config.yaml``) so power users can tune them
without patching the source. The existing hardcoded numbers remain as
defaults, so behaviour is unchanged when the config key is absent.

Example ``config.yaml``::

    tool_output:
      max_bytes: 100000        # terminal output cap (chars)
      max_lines: 5000          # read_file pagination + truncation cap
      max_line_length: 2000    # per-line length cap before '... [truncated]'

The limits reader is defensive: any error (missing config file, invalid
value type, etc.) falls back to the built-in defaults so tools never
fail because of a malformed config.
"""

from __future__ import annotations

import contextvars
import os
import threading
from typing import Any

import aiofiles.os

from hermes_constants import get_hermes_home

# Hardcoded defaults — these match the pre-existing values, so adding
# this module is behaviour-preserving for users who don't set
# ``tool_output`` in config.yaml.
DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH

# Module-level cache — populated on first call.
# Avoids repeated config file I/O on every tool call.
_cached_limits: dict | None = None
_limits_profile_context: contextvars.ContextVar[
    tuple[str, str] | None
] = contextvars.ContextVar("tool_output_limits_profile", default=None)
_limits_profile_aliases: dict[str, str] = {}
_limits_by_profile: dict[str, dict[str, int]] = {}
_limits_cache_guard = threading.RLock()


def _lexical_limits_profile() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_limits_profile() -> str:
    lexical = _lexical_limits_profile()
    active = _limits_profile_context.get()
    if active is not None and active[0] == lexical:
        return active[1]
    with _limits_cache_guard:
        return _limits_profile_aliases.get(lexical, lexical)


async def _activate_limits_profile() -> str:
    """Activate one canonical HERMES_HOME at the awaited refresh edge."""
    lexical = _lexical_limits_profile()
    active = _limits_profile_context.get()
    if active is not None and active[0] == lexical:
        return active[1]
    with _limits_cache_guard:
        canonical = _limits_profile_aliases.get(lexical)
    if canonical is None:
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        expanded = str(await expanduser(lexical))
        is_absolute = (
            expanded.startswith(("/", "\\\\"))
            or (
                len(expanded) >= 3
                and expanded[1] == ":"
                and expanded[2] in "/\\"
            )
        )
        if not is_absolute:
            expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
        realpath = aiofiles.os.wrap(os.path.realpath)
        canonical = os.path.normcase(str(await realpath(expanded)))
    with _limits_cache_guard:
        _limits_profile_aliases[lexical] = canonical
        if lexical != canonical:
            staged = _limits_by_profile.pop(lexical, None)
            if staged is not None:
                _limits_by_profile.setdefault(canonical, staged)
    _limits_profile_context.set((lexical, canonical))
    return canonical


def _default_limits() -> dict[str, int]:
    return {
        "max_bytes": DEFAULT_MAX_BYTES,
        "max_lines": DEFAULT_MAX_LINES,
        "max_line_length": DEFAULT_MAX_LINE_LENGTH,
    }


def _publish_limits(profile: str, limits: dict[str, int]) -> dict[str, int]:
    """Store one profile value and update the historical private snapshot."""
    global _cached_limits
    with _limits_cache_guard:
        _limits_by_profile[profile] = limits
        _cached_limits = limits
    return limits


def _coerce_positive_int(value: Any, default: int) -> int:
    """Return ``value`` as a positive int, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def get_tool_output_limits() -> dict[str, int]:
    """Return deterministic limits without synchronous configuration I/O."""
    profile = _current_limits_profile()
    with _limits_cache_guard:
        cached = _limits_by_profile.get(profile)
    if cached is not None:
        return cached
    return _publish_limits(profile, _default_limits())


async def _refresh_tool_output_limits() -> dict[str, int]:
    """Refresh the active profile through the native async config loader."""
    profile = await _activate_limits_profile()
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        section = config.get("tool_output") if isinstance(config, dict) else None
        if not isinstance(section, dict):
            section = {}
    except Exception:
        section = {}
    limits = {
        "max_bytes": _coerce_positive_int(section.get("max_bytes"), DEFAULT_MAX_BYTES),
        "max_lines": _coerce_positive_int(section.get("max_lines"), DEFAULT_MAX_LINES),
        "max_line_length": _coerce_positive_int(
            section.get("max_line_length"), DEFAULT_MAX_LINE_LENGTH
        ),
    }
    return _publish_limits(profile, limits)


def _reset_tool_output_limits_cache() -> None:
    """Reset the cached limits — for tests or after config hot-reload."""
    global _cached_limits
    with _limits_cache_guard:
        _limits_by_profile.clear()
        _limits_profile_aliases.clear()
        _cached_limits = None
    _limits_profile_context.set(None)


def get_max_bytes() -> int:
    """Shortcut for terminal-tool callers that only need the byte cap."""
    return get_tool_output_limits()["max_bytes"]


def get_max_lines() -> int:
    """Shortcut for file-ops callers that only need the line cap."""
    return get_tool_output_limits()["max_lines"]


def get_max_line_length() -> int:
    """Shortcut for file-ops callers that only need the per-line cap."""
    return get_tool_output_limits()["max_line_length"]
