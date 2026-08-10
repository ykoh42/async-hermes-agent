#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import asyncio
import contextlib
import difflib
import json
import logging
import os
import posixpath
import re
import stat
import sys
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

import aiofiles
import aiofiles.os

from agent.file_safety import (
    get_cross_profile_warning,
    get_read_block_error,
    get_sandbox_mirror_warning,
    get_write_denied_error,
)
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    LINTERS,
    LINTERS_INPROC,
    LintResult,
    PatchResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    WriteResult,
    _FAIL_CLOSED_INPROC_EXTS,
    _SHELL_LINTER_LSP_REDUNDANT,
    _has_bom,
    _looks_like_linter_unusable,
    _parse_search_context_line,
    _strip_bom,
    normalize_read_pagination,
    normalize_search_pagination,
)
from tools import file_state
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


# Read-loop and external-edit state is process-local metadata, not a file
# backend.  It stays synchronous and tiny; all filesystem operations that feed
# it are awaited by the native handlers below.
_read_tracker_lock = threading.RLock()
_read_tracker: dict[str, dict] = {}
_patch_failure_lock = threading.RLock()
_patch_failure_tracker: dict[str, dict[str, int]] = {}
_READ_HISTORY_CAP = 500
_DEDUP_CAP = 1000
_READ_TIMESTAMPS_CAP = 1000
_NOT_FOUND_CAP = 500
_NOT_FOUND_TTL_SECONDS = 60.0
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from the earlier read_file "
    "result in this conversation is still current — refer to that instead "
    "of re-reading."
)


def _cap_read_tracker_data(task_data: dict) -> None:
    """Enforce the upstream bounds on per-task read-tracker containers."""
    history = task_data.get("read_history")
    if history is not None and len(history) > _READ_HISTORY_CAP:
        for _ in range(len(history) - _READ_HISTORY_CAP):
            try:
                history.pop()
            except KeyError:
                break

    for key, cap in (
        ("dedup", _DEDUP_CAP),
        ("dedup_hits", _DEDUP_CAP),
        ("read_timestamps", _READ_TIMESTAMPS_CAP),
        ("not_found", _NOT_FOUND_CAP),
    ):
        values = task_data.get(key)
        if values is not None and len(values) > cap:
            for _ in range(len(values) - cap):
                try:
                    values.pop(next(iter(values)))
                except (StopIteration, KeyError):
                    break


async def _check_not_found_cache(
    op: str, resolved_str: str, task_id: str
) -> str | None:
    """Return a fresh cached not-found result without blocking the event loop."""
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        not_found = task_data.get("not_found") if task_data else None
        entry = not_found.get((op, resolved_str)) if not_found else None
        if entry is None:
            return None
        timestamp, cached_json = entry
        if time.monotonic() - timestamp > _NOT_FOUND_TTL_SECONDS:
            not_found.pop((op, resolved_str), None)
            return None

    if await aiofiles.os.path.exists(resolved_str):
        with _read_tracker_lock:
            task_data = _read_tracker.get(task_id)
            not_found = task_data.get("not_found") if task_data else None
            if not_found:
                not_found.pop((op, resolved_str), None)
        return None
    return cached_json


def _record_not_found(
    op: str, resolved_str: str, task_id: str, error_json: str
) -> None:
    """Cache a not-found result so repeated misses skip suggestion I/O."""
    state = _read_state(task_id)
    with _read_tracker_lock:
        state.setdefault("not_found", {})[(op, resolved_str)] = (
            time.monotonic(),
            error_json,
        )
        _cap_read_tracker_data(state)


def _read_state(task_id: str) -> dict:
    with _read_tracker_lock:
        return _read_tracker.setdefault(
            task_id,
            {
                "last_key": None,
                "consecutive": 0,
                "read_history": set(),
                "dedup": {},
                "dedup_hits": {},
                "read_timestamps": {},
            },
        )


def _record_patch_failure(task_id: str, resolved_path: str) -> int:
    with _patch_failure_lock:
        failures = _patch_failure_tracker.setdefault(task_id, {})
        if len(failures) >= 64 and resolved_path not in failures:
            failures.pop(next(iter(failures)))
        failures[resolved_path] = failures.get(resolved_path, 0) + 1
        return failures[resolved_path]


def _reset_patch_failures(task_id: str, resolved_paths: list[str]) -> None:
    if not resolved_paths:
        return
    with _patch_failure_lock:
        failures = _patch_failure_tracker.get(task_id)
        if failures:
            for path in resolved_paths:
                failures.pop(path, None)


async def _expand_tilde(path: str) -> str:
    """Expand ``~`` using the effective profile home when available.

    In-process file tools share the gateway process's HOME, which may differ
    from the profile-specific HOME that interactive CLI sessions use.  This
    mirrors ``hermes_constants.get_subprocess_home()`` so that ``~`` resolves
    consistently regardless of whether the tool runs interactively or inside a
    gateway-driven cron job (#48552).
    """
    if not path or "~" not in path:
        return path
    try:
        from hermes_constants import get_subprocess_home

        home = await get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    return await expanduser(path)


# ---------------------------------------------------------------------------
# Read-size guard: cap the character count returned to the model.
# We're model-agnostic so we can't count tokens; characters are a safe proxy.
# 100K chars ≈ 25–35K tokens across typical tokenisers.  Files larger than
# this in a single read are a context-window hazard — the model should use
# offset+limit to read the relevant section.
#
# Configurable via config.yaml:  file_read_max_chars: 200000
# ---------------------------------------------------------------------------
_DEFAULT_MAX_READ_CHARS = 100_000
_max_read_chars_cached: int | None = None


async def _get_max_read_chars() -> int:
    """Return the configured max characters per file read.

    Reads ``file_read_max_chars`` from config.yaml on first call, caches
    the result for the lifetime of the process.  Falls back to the
    built-in default if the config is missing or invalid.
    """
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from hermes_cli.config import load_config_readonly

        cfg = await load_config_readonly()
        val = cfg.get("file_read_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            _max_read_chars_cached = int(val)
            return _max_read_chars_cached
    except Exception:
        pass
    _max_read_chars_cached = _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached


def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to fit a char budget.

    Ported in spirit from nearai/ironclaw#5029 (dual line/byte cap on
    ``read_file``). Where hermes previously hard-rejected an oversized read
    (forcing the model to guess a smaller ``limit`` and burn a round-trip
    returning nothing), this trims the content to the last *complete line*
    that fits within ``max_chars`` and reports how many lines were kept so
    the caller can offer a ``next_offset`` continuation.

    ``content`` is the gutter-rendered text (``LINE_NUM|CONTENT`` joined by
    ``\\n``). Individual lines are already clamped to ``get_max_line_length()``
    upstream, so a single line never blows the whole budget on its own; the
    overflow this handles is the *accumulation* of many lines under the
    line-count limit (logs, wide CSV rows, minified data).

    Returns ``(kept_text, lines_kept, truncated)``. When ``content`` already
    fits, returns it unchanged with ``truncated=False``. If not even the
    first line fits, that single line is clamped on a code-point boundary
    (Python ``str`` slicing never splits a code point) so the read never
    returns empty and the cursor can still advance.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        # +1 for the "\n" that rejoins this line to the previous one.
        addition = len(line) + (1 if kept else 0)
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition

    if not kept:
        # First line alone exceeds the budget. Clamp on a code-point
        # boundary rather than emitting nothing.
        kept.append(lines[0][:max_chars])

    return "\n".join(kept), len(kept), True


def _add_line_numbers(content: str, start_line: int = 1) -> str:
    """Add the compact upstream line gutter and clamp individual lines."""
    from tools.tool_output_limits import get_max_line_length

    max_line_length = get_max_line_length()
    numbered: list[str] = []
    for line_number, line in enumerate(content.split("\n"), start=start_line):
        if len(line) > max_line_length:
            line = line[:max_line_length] + "... [truncated]"
        numbered.append(f"{line_number}|{line}")
    return "\n".join(numbered)


# If the total file size exceeds this AND the caller didn't specify a narrow
# range (limit <= 200), we include a hint encouraging targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000  # 512 KB

# ---------------------------------------------------------------------------
# Device path blocklist — reading these hangs the process (infinite output
# or blocking on input).  Checked by path only (no I/O).
# ---------------------------------------------------------------------------
_BLOCKED_DEVICE_PATHS = frozenset({
    # Infinite output — never reach EOF
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    # Blocks waiting for input
    "/dev/stdin", "/dev/tty", "/dev/console",
    # Nonsensical to read
    "/dev/stdout", "/dev/stderr",
    # fd aliases
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


async def _resolve_path(
    filepath: str, task_id: str = "default"
) -> Path | PurePosixPath:
    """Resolve a path relative to TERMINAL_CWD (the worktree base directory)
    instead of the main repository root.
    """
    return await _resolve_path_for_task(filepath, task_id)


# Sentinel ``TERMINAL_CWD`` values that mean "not configured", NOT a literal
# directory to resolve against. A stale config / .env commonly leaves the
# literal "." here; "auto"/"cwd" are setup-wizard placeholders. Treating any of
# these as a real relative base silently anchors edits to the agent PROCESS cwd
# (e.g. the main repo while a worktree session is active), routing writes to the
# wrong checkout. The gateway sanitizes the same set at import time
# (gateway/run.py); the file/terminal-tool layer must do likewise so CLI
# sessions get the same protection. See references/worktree-cwd-discipline.md.
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})


def _terminal_env_type_for_task(task_id: str = "default") -> str:
    """Return the only terminal backend retained by this training runtime."""
    return "local"


def _uses_container_paths(task_id: str = "default") -> bool:
    try:
        from tools.terminal_tool import _CONTAINER_BACKENDS
        container_backends = _CONTAINER_BACKENDS
    except Exception:
        container_backends = _CONTAINER_PATH_BACKENDS_FALLBACK
    return _terminal_env_type_for_task(task_id) in container_backends


def _normalize_without_host_deref(path: str | Path | PurePosixPath) -> PurePosixPath:
    """Normalize path syntax without following host symlinks.

    Container backends use paths that are meaningful inside the sandbox. Calling
    ``Path.resolve()`` on the host can dereference a host-side symlink such as
    ``/workspace`` and rewrite the path before Docker sees it.
    """
    return PurePosixPath(posixpath.normpath(str(path)))


async def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Normalize a cwd candidate to an absolute, sentinel-free anchor.

    Returns the expanded path only when *raw* is non-empty, not a sentinel (see
    ``_TERMINAL_CWD_SENTINELS``), and absolute. A relative anchor is meaningless
    without knowing which cwd it is relative to — exactly the ambiguity that
    misroutes worktree edits — so relative/sentinel/empty values yield ``None``.
    """
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = await _expand_tilde(raw)
    if not os.path.isabs(expanded):
        return None
    return expanded


async def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real directory anchor.

    Sentinel values (see ``_TERMINAL_CWD_SENTINELS``) and relative paths are
    rejected — a relative anchor is meaningless without knowing which cwd it is
    relative to, which is exactly the ambiguity that misroutes worktree edits.
    Only an absolute, sentinel-free value is honored.
    """
    return await _sentinel_free_abs_cwd(os.environ.get("TERMINAL_CWD"))


async def _registered_task_cwd_override(task_id: str = "default") -> str | None:
    """Return a registered cwd override for the raw task id, when available.

    ``terminal_tool`` intentionally collapses CWD-only task overrides to the
    shared ``"default"`` environment so TUI/dashboard/ACP sessions do not spin
    up isolated sandboxes just because they have different workspaces. The cwd
    value itself is still keyed by the raw session/task id, so file tools must
    read that raw override before falling back to the collapsed container key.
    """
    try:
        from tools.terminal_tool import resolve_task_overrides

        overrides = resolve_task_overrides(task_id)
    except Exception:
        return None

    return await _sentinel_free_abs_cwd(overrides.get("cwd"))


async def _authoritative_workspace_root(task_id: str = "default") -> str | None:
    """Best-effort absolute workspace root for divergence checks.

    Resolution:

      1. The session's own cwd RECORD (``terminal_tool.get_session_cwd``) —
         written on every completed terminal command and seeded by workspace
         registration, keyed by the raw session id. Because the record is
         per-session, one session's ``cd`` can never leak into another
         session's resolution.
      2. A registered task/session cwd override (TUI/Desktop/ACP sessions
         register a raw-keyed cwd before any tool runs). Normally already
         mirrored into the record at registration; kept as a direct fallback
         so a cleared/never-written record still resolves the workspace.
      3. A sentinel-free absolute ``$TERMINAL_CWD`` (the worktree path set by
         ``cli.py``/``main.py`` for ``-w`` sessions).

    Returns ``None`` only when there is genuinely no reliable anchor, in which
    case callers fall back to the process cwd.
    """
    try:
        from tools.terminal_tool import get_session_cwd

        recorded = await get_session_cwd(task_id)
    except Exception:
        recorded = None
    if recorded:
        return recorded
    registered = await _registered_task_cwd_override(task_id)
    if registered:
        return registered
    return await _configured_terminal_cwd()


async def _resolve_base_dir(
    task_id: str = "default",
    *,
    container_paths: bool | None = None,
) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths.

    Resolution order:
      1. The task's live terminal cwd (the directory the agent is actually
         working in — e.g. a git worktree). Authoritative when known.
      2. A registered task/session cwd override (TUI/Desktop/ACP sessions
         register a raw-keyed workspace cwd before any terminal command runs).
      3. A sentinel-free, absolute ``$TERMINAL_CWD`` (the worktree path set by
         ``cli.py``/``main.py`` for ``-w`` sessions). Used even before any
         terminal command has populated the live cwd registry.
      4. The process cwd.

    The returned base is ALWAYS absolute. This is the core invariant that
    prevents the worktree-cwd divergence bug: a relative or sentinel
    ``TERMINAL_CWD`` (commonly the literal ``"."`` from a stale config) is
    meaningless as a resolution anchor — left to ``Path.resolve()`` it silently
    resolves against whatever the agent PROCESS cwd happens to be (e.g. the main
    repo while the terminal is in a worktree), routing edits to the wrong
    checkout. We therefore reject sentinel/relative ``TERMINAL_CWD`` values
    outright (rather than anchoring them to the process cwd) and fall through to
    the process cwd only as a last resort, deterministically.
    """
    root = await _authoritative_workspace_root(task_id)
    if container_paths is None:
        container_paths = _uses_container_paths(task_id)
    if root:
        base_text = await _expand_tilde(root)
    else:
        base_text = await aiofiles.os.getcwd()
    if container_paths:
        if not posixpath.isabs(base_text):
            base_text = posixpath.join(await aiofiles.os.getcwd(), base_text)
        return _normalize_without_host_deref(base_text)
    # Git Bash ``pwd -P`` reports ``/c/Users/...``; translate before Path so
    # relative file-tool paths don't anchor under a nonexistent ``\\c\\Users``.
    from tools.environments.local import _msys_to_windows_path

    base_text = _msys_to_windows_path(base_text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(base_text):
            base_text = ntpath.join(await aiofiles.os.getcwd(), base_text)
        return Path(ntpath.normpath(base_text))
    base = Path(base_text)
    if not base.is_absolute():
        # Last-resort anchoring: a live cwd should already be absolute, but if a
        # terminal backend ever reports a relative cwd, anchor it to the process
        # cwd once, here, so the result no longer depends on cwd at resolve().
        base = Path(await aiofiles.os.getcwd()) / base
    realpath = aiofiles.os.wrap(os.path.realpath)
    return Path(await realpath(base))


async def _resolve_path_for_task(
    filepath: str, task_id: str = "default"
) -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory.

    See :func:`_resolve_base_dir` for how the base is chosen. Absolute input
    paths are returned resolved-but-unanchored.

    On native Windows, Git Bash / MSYS drive paths (``/c/Users/...``) are
    translated to ``C:\\Users\\...`` before resolution so file tools don't
    treat them as relative ``\\c\\Users\\...`` under the process cwd.
    """
    container_paths = _uses_container_paths(task_id)
    if container_paths:
        expanded = await _expand_tilde(filepath)
        if posixpath.isabs(expanded):
            return _normalize_without_host_deref(expanded)
        resolved = await _resolve_base_dir(task_id, container_paths=True) / expanded
        return _normalize_without_host_deref(resolved)

    # Host paths only — never rewrite Linux paths inside a container/WSL env.
    from tools.environments.local import _msys_to_windows_path

    expanded = await _expand_tilde(_msys_to_windows_path(filepath))
    realpath = aiofiles.os.wrap(os.path.realpath)
    if sys.platform == "win32":
        import ntpath

        if ntpath.isabs(expanded):
            return Path(await realpath(ntpath.normpath(expanded)))
        joined = ntpath.join(
            str(await _resolve_base_dir(task_id, container_paths=False)), expanded
        )
        return Path(await realpath(ntpath.normpath(joined)))

    p = Path(expanded)
    if p.is_absolute():
        return Path(await realpath(p))
    resolved = await _resolve_base_dir(task_id, container_paths=False) / p
    return Path(await realpath(resolved))


async def _path_resolution_warning(
    filepath: str,
    resolved: Path,
    task_id: str = "default",
) -> str | None:
    """Warn when a relative path resolves outside the active workspace."""
    try:
        if Path(await _expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = await _authoritative_workspace_root(task_id)
        if not workspace_root:
            return None
        if _uses_container_paths(task_id):
            root = _normalize_without_host_deref(
                Path(await _expand_tilde(workspace_root))
            )
        else:
            realpath = aiofiles.os.wrap(os.path.realpath)
            root = Path(await realpath(await _expand_tilde(workspace_root)))
        try:
            resolved.relative_to(root)
            return None
        except ValueError:
            return (
                f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
                f"OUTSIDE the active workspace ({str(root)!r}). The edit will land "
                "in a different directory than the terminal's cwd. If this is not "
                "intended (e.g. a git-worktree session writing into the main "
                "checkout), pass an absolute path under the workspace instead."
            )
    except Exception:
        return None


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd paths that can hang reads."""
    normalized = os.path.normpath(path)
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/self/fd/0-2 and /proc/<pid>/fd/0-2 are Linux aliases for stdio
    if normalized.startswith("/proc/") and normalized.endswith(
        ("/fd/0", "/fd/1", "/fd/2")
    ):
        return True
    # /proc/*/environ, /proc/*/cmdline, /proc/*/maps (and the maps variants
    # smaps, smaps_rollup, numa_maps) can leak secrets, command-line args, and
    # memory layout (ASLR bypass) from the host process (issue #4427).
    # /proc/*/mem exposes raw process memory; block it as defense-in-depth even
    # though it requires address knowledge to exploit usefully.
    # /proc/*/auxv leaks AT_RANDOM (stack canary seed) plus AT_BASE/AT_PHDR
    # load addresses — an ASLR oracle on par with maps. /proc/*/pagemap exposes
    # virtual->physical translation. Both are blocked alongside the maps family.
    # endswith matches both /proc/<pid>/X and /proc/<pid>/task/<tid>/X.
    if normalized.startswith("/proc/") and normalized.endswith(
        (
            "/environ",
            "/cmdline",
            "/maps",
            "/smaps",
            "/smaps_rollup",
            "/numa_maps",
            "/mem",
            "/auxv",
            "/pagemap",
        )
    ):
        return True
    return False


async def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """Return True if the path would hang the process (infinite output or blocking input).

    Check the literal path first so aliases like /dev/stdin are caught before
    they resolve to terminal-specific paths. Then check each symlink hop before
    the final resolved path so aliases to devices cannot bypass the guard.
    """
    expanded = await _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)  # noqa: ASYNC240 - lexical only
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = await aiofiles.os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)  # noqa: ASYNC240 - lexical only
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        realpath = aiofiles.os.wrap(os.path.realpath)
        resolved = os.path.normpath(  # noqa: ASYNC240 - lexical only
            await realpath(normalized)
        )
    except (OSError, ValueError):
        return False
    if _is_blocked_device_path(resolved):
        return True
    return False






# Paths that file tools should refuse to write to without going through the
# terminal tool's approval system.  These match prefixes after os.path.realpath.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/db/", "/private/var/root/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False


async def _get_hermes_config_path() -> str | None:
    """Return the absolute Hermes config path without filesystem access."""
    global _hermes_config_resolved, _hermes_config_resolved_loaded
    if _hermes_config_resolved_loaded:
        return _hermes_config_resolved
    _hermes_config_resolved_loaded = True
    try:
        from hermes_cli.config import get_config_path
        _hermes_config_resolved = await aiofiles.os.path.abspath(get_config_path())
    except Exception:
        try:
            _hermes_config_resolved = await aiofiles.os.path.abspath(
                await _expand_tilde("~/.hermes/config.yaml")
            )
        except Exception:
            _hermes_config_resolved = None
    return _hermes_config_resolved


async def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
    try:
        resolved = str(await _resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        resolved = filepath
    normalized = os.path.normpath(  # noqa: ASYNC240 - lexical only
        await _expand_tilde(filepath)
    )
    _err = (
        f"Refusing to write to sensitive system path: {filepath}\n"
        "Use the terminal tool with sudo if you need to modify system files."
    )
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix) or normalized.startswith(prefix):
            return _err
    if resolved in _SENSITIVE_EXACT_PATHS or normalized in _SENSITIVE_EXACT_PATHS:
        return _err
    # Prevent agents from modifying the Hermes config file directly.
    # approvals.mode and other security settings live here; a malicious or
    # prompt-injected agent could silently disable exec approval by writing to
    # this file.
    hermes_config = await _get_hermes_config_path()
    # macOS resolves ``/tmp`` through the ``/private`` symlink.  Canonicalize
    # both values so the config guard is stable regardless of the spelling the
    # caller used.
    realpath = aiofiles.os.wrap(os.path.realpath)
    canonical_resolved = await realpath(resolved)
    canonical_normalized = await realpath(normalized)
    canonical_config = await realpath(hermes_config) if hermes_config else None
    if canonical_config and (
        canonical_resolved == canonical_config
        or canonical_normalized == canonical_config
    ):
        return (
            f"Refusing to write to Hermes config file: {filepath}\n"
            "Agent cannot modify security-sensitive configuration. "
            "Edit ~/.hermes/config.yaml directly or use 'hermes config' instead."
        )
    return None


async def _check_cross_profile_path(
    filepath: str,
    task_id: str = "default",
) -> str | None:
    """Apply the cross-profile and host-side sandbox-mirror soft guards."""
    try:
        resolved = str(await _resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        resolved = filepath

    warning = await get_cross_profile_warning(resolved)
    if warning is not None:
        return warning
    return await get_sandbox_mirror_warning(resolved)






def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """Return True for content dominated by read_file's ``LINE_NUM|CONTENT`` display.

    ``read_file`` intentionally returns line-numbered text to the model. If
    that display format is echoed into ``write_file``, config/source files are
    silently corrupted with prefixes like `` 1|``.  We reject writes where the
    non-empty lines are mostly consecutive read_file-style numbered lines, while
    allowing sparse literal pipe content such as a single ``1|value`` line.
    """
    if not isinstance(content, str):
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numbered: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        prefix, sep, _rest = stripped.partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))

    if len(numbered) < 2:
        return False
    if len(numbered) / len(lines) < 0.6:
        return False

    consecutive_pairs = sum(
        1 for prev, current in zip(numbered, numbered[1:])
        if current == prev + 1
    )
    return consecutive_pairs >= len(numbered) - 1


def _is_internal_file_status_text(content: str) -> bool:
    """Return True when *content* is the read dedup status, not file data."""
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped == _READ_DEDUP_STATUS_MESSAGE:
        return True
    return (
        _READ_DEDUP_STATUS_MESSAGE in stripped
        and len(stripped) <= 2 * len(_READ_DEDUP_STATUS_MESSAGE)
    )


def _is_internal_file_tool_content(content: str) -> bool:
    """Return True when content is file-tool display text, not intended file bytes."""
    return (
        _is_internal_file_status_text(content)
        or _looks_like_read_file_line_numbered_content(content)
    )


def reset_file_dedup(task_id: str | None = None) -> None:
    """Clear cached unchanged-read results after context compression."""
    with _read_tracker_lock:
        states = [_read_tracker.get(task_id)] if task_id else list(_read_tracker.values())
        for state in states:
            if state:
                state.setdefault("dedup", {}).clear()
                state.setdefault("dedup_hits", {}).clear()


def clear_file_ops_cache(task_id: str | None = None) -> None:
    """Release per-task file-tool state when a terminal environment closes.

    The async fork no longer keeps the upstream shell-backed ``FileOperations``
    object cache, but it still owns equivalent per-task read-loop and patch
    failure state.  Keep the upstream cleanup entry point so lifecycle code
    and integrations can release that state without knowing the backend
    implementation.
    """
    with _read_tracker_lock:
        if task_id is None:
            _read_tracker.clear()
        else:
            _read_tracker.pop(task_id, None)
    with _patch_failure_lock:
        if task_id is None:
            _patch_failure_tracker.clear()
        else:
            _patch_failure_tracker.pop(task_id, None)
    if task_id is not None:
        file_state.get_registry().clear_task(task_id)


def notify_other_tool_call(task_id: str = "default") -> None:
    """Break a consecutive read/stub loop after another tool runs."""
    with _read_tracker_lock:
        state = _read_tracker.get(task_id)
        if state:
            state["last_key"] = None
            state["consecutive"] = 0
            state.setdefault("dedup_hits", {}).clear()
            state.setdefault("not_found", {}).clear()


async def _invalidate_dedup_for_path(filepath: str, task_id: str) -> None:
    try:
        resolved = str(await _resolve_path(filepath, task_id))
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        state = _read_tracker.get(task_id)
        if state:
            state["dedup"] = {
                key: value
                for key, value in state["dedup"].items()
                if key[0] != resolved
            }
            not_found = state.get("not_found")
            if not_found:
                for key in list(not_found):
                    if key[1] == resolved:
                        not_found.pop(key, None)


async def _search_result_read_block_error(
    path: str,
    task_id: str = "default",
) -> str | None:
    """Apply the credential read guard to a search result path."""
    try:
        resolved = await _resolve_path_for_task(path, task_id)
    except (OSError, ValueError, RuntimeError):
        return await get_read_block_error(path)
    return await get_read_block_error(str(resolved))


async def _filter_search_output_lines(
    lines: list[str],
    task_id: str,
) -> tuple[list[str], int]:
    """Remove credential-bearing ``rg`` rows before returning them to the model."""
    safe: list[str] = []
    omitted = 0
    for line in lines:
        candidate = line.split(":", 1)[0] if ":" in line else line
        if await _search_result_read_block_error(candidate, task_id):
            omitted += 1
            continue
        safe.append(line)
    return safe, omitted






























# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error


def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. Jupyter notebooks (.ipynb), Word documents (.docx), and Excel workbooks (.xlsx) are auto-extracted to readable text. NOTE: Cannot read images or other binary files — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 2000, max: 2000). Reads are additionally capped at a ~100K-character budget with a next_offset continuation.", "default": 2000, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out). The result's verified:true means the on-disk content hash was confirmed — do NOT re-read the file to check the write landed.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hermes profile's skills/plugins/memories — by default these writes are blocked with a warning because they affect a different profile than the one this session is running under.",
                "default": False,
            },
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
        "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
        "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
        "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
        "REQUIRED PARAMETERS: mode, patch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["replace", "patch"],
                "description": "Edit mode. 'replace' (default): requires path + old_string + new_string. 'patch': requires patch content only.",
                "default": "replace",
            },
            "path": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Replacement text. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            "patch": {
                "type": "string",
                "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
            },
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hermes profile's skills/plugins/memories.",
                "default": False,
            },
        },
        "required": ["mode"],
    },
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0}
        },
        "required": ["pattern"]
    }
}










# Native async handlers -------------------------------------------------------
#
# These handlers own filesystem I/O directly.  There is no synchronous
# compatibility backend: a caller either awaits the native handler or receives
# the explicit unsupported-backend error from ``_native_file_path``.

async def _native_file_path(path: str, task_id: str) -> Path | str:
    """Resolve a host path or return the user-facing reason it cannot run."""
    try:
        resolved = await _resolve_path_for_task(path, task_id)
    except (OSError, ValueError) as exc:
        return tool_error(f"Invalid file path {path!r}: {exc}")
    if not isinstance(resolved, Path):
        return tool_error(
            "The configured terminal backend does not expose a native async "
            "filesystem. Use a backend with an async file implementation."
        )
    return resolved


async def _file_mtime(path: Path) -> float | None:
    try:
        stat_result = await aiofiles.os.stat(path)
    except OSError:
        return None
    return stat_result.st_mtime


async def _check_file_staleness(path: str, task_id: str) -> str | None:
    try:
        resolved = str(await _resolve_path(path, task_id))
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        state = _read_tracker.get(task_id) or {}
        read_mtime = state.get("read_timestamps", {}).get(resolved)
    if read_mtime is None:
        return None
    current_mtime = await _file_mtime(Path(resolved))
    if current_mtime is None or current_mtime == read_mtime:
        return None
    return (
        f"Warning: {path} was modified since you last read it "
        "(external edit or concurrent agent). The content you read may be "
        "stale. Consider re-reading the file to verify before writing."
    )


async def _refresh_read_timestamp(path: str, task_id: str) -> None:
    """Invalidate cached ranges and record the post-write mtime asynchronously."""
    await _invalidate_dedup_for_path(path, task_id)
    try:
        resolved = str(await _resolve_path(path, task_id))
    except (OSError, ValueError):
        return
    mtime = await _file_mtime(Path(resolved))
    if mtime is None:
        return
    with _read_tracker_lock:
        state = _read_tracker.get(task_id)
        if state:
            state.setdefault("read_timestamps", {})[resolved] = mtime
            _cap_read_tracker_data(state)


def _record_read_metadata(
    task_id: str,
    *,
    path: str,
    offset: int,
    limit: int,
    mtime: float | None,
    truncated: bool,
) -> int:
    """Update read-loop state after a successful native read."""
    state = _read_state(task_id)
    with _read_tracker_lock:
        dedup_key = (path, offset, limit)
        state.setdefault("dedup", {}).pop(dedup_key, None)
        state.setdefault("dedup_hits", {}).pop(dedup_key, None)
        state.setdefault("read_history", set()).add((path, offset, limit))
        read_key = ("read", path, offset, limit)
        if state.get("last_key") == read_key:
            state["consecutive"] += 1
        else:
            state["last_key"] = read_key
            state["consecutive"] = 1
        if mtime is not None:
            state["dedup"][dedup_key] = mtime
            state.setdefault("read_timestamps", {})[path] = mtime
        _cap_read_tracker_data(state)
        return state["consecutive"]


async def _handle_read_file(args, **kw):
    """Read a local text file with native async I/O and stable line gutters."""
    from tools.tool_output_limits import _refresh_tool_output_limits

    await _refresh_tool_output_limits()
    task_id = kw.get("task_id") or "default"
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        return tool_error("read_file: missing required field 'path'.")

    offset, limit = normalize_read_pagination(args.get("offset", 1), args.get("limit", 500))
    device_base = (
        None
        if Path(await _expand_tilde(path)).is_absolute()
        else await _resolve_base_dir(task_id)
    )
    if await _is_blocked_device(path, base_dir=device_base):
        return tool_error(
            f"Cannot read '{path}': this is a device file that would block or produce infinite output."
        )

    resolved = await _native_file_path(path, task_id)
    if isinstance(resolved, str):
        return resolved

    # Structured documents are binary containers but intentionally render as
    # text in Hermes. Keep this before the binary-extension guard, matching the
    # upstream read_file contract.
    from tools.read_extract import (
        ExtractionError,
        extract_document_text,
        is_extractable_document,
    )

    if is_extractable_document(str(resolved)):
        try:
            extracted_text = await extract_document_text(str(resolved))
        except ExtractionError:
            logger.debug("document extraction failed for %s", path, exc_info=True)
        else:
            lines = extracted_text.splitlines()
            page = lines[offset - 1:offset - 1 + limit]
            numbered = _add_line_numbers("\n".join(page), offset) if page else ""
            numbered, lines_kept, char_truncated = _truncate_to_char_budget(
                numbered, await _get_max_read_chars()
            )
            next_offset = offset + lines_kept
            truncated = len(lines) >= next_offset or char_truncated
            result = {
                "content": (
                    redact_sensitive_text(numbered, file_read=True)
                    if numbered
                    else ""
                ),
                "total_lines": len(lines),
                "file_size": (await aiofiles.os.stat(resolved)).st_size,
                "truncated": truncated,
                "extracted_document": True,
            }
            if char_truncated:
                max_chars = await _get_max_read_chars()
                shown_end = offset + lines_kept - 1
                result["truncated_by"] = "bytes"
                result["next_offset"] = next_offset
                result["hint"] = (
                    f"Output truncated at the {max_chars:,}-char read budget "
                    f"after {lines_kept} line(s) (showing lines {offset}-"
                    f"{shown_end} of {len(lines)}). Use offset={next_offset} "
                    "to continue."
                )
                if len(numbered.split("\n", 1)[0]) >= max_chars:
                    result["hint"] += (
                        " Note: the first line alone exceeded the budget and "
                        "was clamped mid-line; its remainder is not retrievable "
                        "via offset."
                    )
            elif truncated:
                result["next_offset"] = next_offset
                result["hint"] = (
                    f"Use offset={next_offset} to continue reading "
                    f"(showing {offset}-{max(offset, next_offset - 1)} "
                    f"of {len(lines)} lines)"
                )
            return json.dumps(result, ensure_ascii=False)

    if has_binary_extension(str(resolved)):
        return tool_error(
            f"Cannot read binary file '{path}' ({resolved.suffix.lower()}). "
            "Use vision_analyze for images, or terminal to inspect binary files."
        )
    block_error = await get_read_block_error(str(resolved))
    if block_error:
        return tool_error(block_error)

    resolved_str = str(resolved)
    cached_not_found = await _check_not_found_cache(
        "read", resolved_str, task_id
    )
    if cached_not_found is not None:
        return cached_not_found
    dedup_key = (resolved_str, offset, limit)
    state = _read_state(task_id)
    mtime = await _file_mtime(resolved)
    with _read_tracker_lock:
        cached_mtime = state.setdefault("dedup", {}).get(dedup_key)
    if mtime is not None and cached_mtime == mtime:
        with _read_tracker_lock:
            hits = state.setdefault("dedup_hits", {}).get(dedup_key, 0) + 1
            state["dedup_hits"][dedup_key] = hits
            _cap_read_tracker_data(state)
        if hits >= 2:
            return tool_error(
                f"BLOCKED: You have called read_file on this exact region "
                f"{hits + 1} times and the file has NOT changed. STOP calling "
                "read_file for this path — the content from your earlier "
                "read_file result in this conversation is still current. "
                "Proceed with your task using the information you already have.",
                path=path,
                already_read=hits + 1,
            )
        return json.dumps(
            {
                "status": "unchanged",
                "message": _READ_DEDUP_STATUS_MESSAGE,
                "path": path,
                "dedup": True,
                "content_returned": False,
            },
            ensure_ascii=False,
        )

    try:
        async with aiofiles.open(resolved, "rb") as handle:
            data = await handle.read()
    except FileNotFoundError:
        directory = resolved.parent
        display_directory = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        basename_no_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1].lower()
        lower_name = filename.lower()
        scored: list[tuple[int, str]] = []
        try:
            candidates = (await aiofiles.os.listdir(directory))[:50]
        except OSError:
            candidates = []
        for candidate in candidates:
            lower_candidate = candidate.lower()
            score = 0
            if lower_candidate == lower_name:
                score = 100
            elif Path(candidate).stem.lower() == basename_no_ext.lower():
                score = 90
            elif (
                lower_candidate.startswith(lower_name)
                or lower_name.startswith(lower_candidate)
            ):
                score = 70
            elif lower_name in lower_candidate:
                score = 60
            elif lower_candidate in lower_name and len(lower_candidate) > 2:
                score = 40
            elif extension and Path(candidate).suffix.lower() == extension:
                common = set(lower_name) & set(lower_candidate)
                if len(common) >= max(len(lower_name), len(lower_candidate)) * 0.4:
                    score = 30
            if score:
                scored.append((score, os.path.join(display_directory, candidate)))
        scored.sort(key=lambda item: -item[0])
        error_json = tool_error(
            f"File not found: {path}",
            similar_files=[candidate for _, candidate in scored[:5]],
        )
        _record_not_found("read", resolved_str, task_id, error_json)
        return error_json
    except OSError as exc:
        return tool_error(f"Failed to read {path}: {exc}")

    sample = data[:1000].decode("utf-8", errors="replace")
    non_printable = sum(
        1 for character in sample if ord(character) < 32 and character not in "\n\r\t"
    )
    if (
        "\ufffd" in sample
        or (sample and non_printable / len(sample) > 0.30)
    ):
        return json.dumps(
            ReadResult(
                file_size=len(data),
                is_binary=True,
                error=(
                    "Binary file - cannot display as text. Use appropriate "
                    "tools to handle this file type."
                ),
            ).to_dict(),
            ensure_ascii=False,
        )

    text = data.decode("utf-8", errors="replace")
    if offset == 1:
        text, _ = _strip_bom(text)
    lines = text.splitlines()
    total_lines = data.count(b"\n")
    page = lines[offset - 1:offset - 1 + limit]
    numbered = _add_line_numbers("\n".join(page), offset) if page else ""
    max_chars = await _get_max_read_chars()
    numbered, lines_kept, char_truncated = _truncate_to_char_budget(
        numbered, max_chars
    )
    next_offset = offset + lines_kept
    truncated = total_lines > offset + limit - 1 or char_truncated
    result = {
        "content": redact_sensitive_text(numbered, file_read=True) if numbered else "",
        "total_lines": total_lines,
        "file_size": len(data),
        "truncated": truncated,
        "is_binary": False,
        "is_image": False,
    }
    if char_truncated:
        shown_end = offset + lines_kept - 1
        result["truncated_by"] = "bytes"
        result["next_offset"] = next_offset
        result["hint"] = (
            f"Output truncated at the {max_chars:,}-char read budget after "
            f"{lines_kept} line(s) (showing lines {offset}-{shown_end} of "
            f"{total_lines}). Use offset={next_offset} to continue."
        )
        if len(numbered.split("\n", 1)[0]) >= max_chars:
            result["hint"] += (
                " Note: the first line alone exceeded the budget and was "
                "clamped mid-line; its remainder is not retrievable via offset."
            )
    elif truncated:
        result["next_offset"] = next_offset
        result["hint"] = (
            f"Use offset={next_offset} to continue reading "
            f"(showing {offset}-{max(offset, next_offset - 1)} of {total_lines} lines)"
        )
    if len(data) > _LARGE_FILE_HINT_BYTES and limit > 200 and truncated:
        result["_hint"] = (
            f"This file is large ({len(data):,} bytes). Consider reading only "
            "the section you need with offset and limit to keep context usage "
            "efficient."
        )
    count = _record_read_metadata(
        task_id,
        path=resolved_str,
        offset=offset,
        limit=limit,
        mtime=mtime if mtime is not None else await _file_mtime(resolved),
        truncated=truncated,
    )
    try:
        await file_state.record_read(
            task_id,
            resolved_str,
            partial=offset > 1 or truncated,
        )
    except Exception:
        logger.debug("file_state.record_read failed", exc_info=True)
    if count >= 4:
        return tool_error(
            f"BLOCKED: You have read this exact file region {count} times in a row. "
            "The content has NOT changed. You already have this information. "
            "STOP re-reading and proceed with your task.",
            path=path,
            already_read=count,
        )
    if count >= 3:
        result["_warning"] = (
            f"You have read this exact file region {count} times consecutively. "
            "The content has not changed since your last read. Use the "
            "information you already have."
        )
    return json.dumps(result, ensure_ascii=False)


async def _write_native_file(path: Path, content: str) -> None:
    """Atomically replace *path* without leaving a partial write on cancel."""
    write_path = path
    is_link = aiofiles.os.wrap(os.path.islink)
    if await is_link(path):
        realpath = aiofiles.os.wrap(os.path.realpath)
        resolved_target = await realpath(path)
        if resolved_target:
            write_path = Path(resolved_target)

    original_mode: int | None = None
    try:
        original_mode = stat.S_IMODE((await aiofiles.os.stat(write_path)).st_mode)
    except FileNotFoundError:
        pass
    try:
        await aiofiles.os.makedirs(write_path.parent, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Unable to create parent directory {write_path.parent}: {exc}"
        ) from exc

    temporary = (
        write_path.parent / f".{write_path.name}.hermes-{uuid.uuid4().hex}.tmp"
    )
    try:
        async with aiofiles.open(temporary, "w", encoding="utf-8", newline="") as handle:
            await handle.write(content)
            await handle.flush()
        if original_mode is not None:
            await aiofiles.os.wrap(os.chmod)(temporary, original_mode)
        await aiofiles.os.replace(temporary, write_path)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass


async def _verify_native_file(path: Path, expected: str) -> bool:
    """Read back an atomic write and compare the exact persisted text."""
    async with aiofiles.open(
        path, "r", encoding="utf-8", errors="strict", newline=""
    ) as handle:
        return await handle.read() == expected


async def _finish_subprocess_communicate(
    process: asyncio.subprocess.Process,
    communicate: asyncio.Task[tuple[bytes, bytes | None]],
) -> tuple[bytes, bytes | None]:
    """Finish one owned communicate task before propagating cancellation."""
    async def drain_or_wait() -> tuple[bytes, bytes | None]:
        try:
            return await communicate
        except BaseException:
            await process.wait()
            raise

    cleanup_task = asyncio.create_task(drain_or_wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if cleanup_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def _check_lint(path: Path, content: str | None = None) -> LintResult:
    """Run the v2026.8.3 syntax check without blocking the event loop."""
    extension = path.suffix.lower()
    in_process = LINTERS_INPROC.get(extension)
    if in_process is not None:
        if content is None:
            try:
                content = await _read_native_patch_content(path)
            except OSError:
                return LintResult(
                    skipped=True,
                    message=f"Failed to read {path} for lint",
                )
        ok, error = in_process(content)
        if error == "__SKIP__":
            return LintResult(
                skipped=True,
                message=f"No linter available for {extension} (missing dependency)",
            )
        return LintResult(success=ok, output="" if ok else error)

    linter_command = LINTERS.get(extension)
    if linter_command is None:
        return LintResult(
            skipped=True,
            message=f"No linter for {extension} files",
        )
    if extension in _SHELL_LINTER_LSP_REDUNDANT and await _lsp_will_handle(path):
        return LintResult(
            skipped=True,
            message=f"{extension} diagnostics handled by LSP",
        )
    argv = [
        str(path) if part == "{file}" else part
        for part in linter_command.split()
        if part != "2>&1"
    ]
    base_command = argv[0]
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return LintResult(skipped=True, message=f"{base_command} not available")

    communicate = asyncio.create_task(process.communicate())
    communication_error: Exception | None = None
    try:
        async with asyncio.timeout(30):
            stdout, _ = await asyncio.shield(communicate)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        # The communicate task is deliberately kept alive after the timeout
        # so its pipe readers can drain and reap the child.  Shield cleanup
        # from a second caller cancellation; otherwise the killed process can
        # still leave a detached communicate task behind.
        stdout, _ = await _finish_subprocess_communicate(process, communicate)
        output = stdout.decode(errors="replace").strip()
        return LintResult(
            success=False,
            output=output or "Lint command timed out after 30s",
        )
    except asyncio.CancelledError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await _finish_subprocess_communicate(process, communicate)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Lint cleanup after cancellation failed", exc_info=True)
        raise
    except Exception as exc:
        communication_error = exc
    if communication_error is not None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await _finish_subprocess_communicate(process, communicate)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        raise communication_error

    output = stdout.decode(errors="replace").strip()
    if process.returncode and _looks_like_linter_unusable(base_command, output):
        from tools.ansi_strip import strip_ansi

        cleaned = strip_ansi(output).strip()
        first_line = next(
            (line.strip() for line in cleaned.splitlines() if line.strip()),
            cleaned[:120],
        )
        return LintResult(
            skipped=True,
            message=f"{base_command} not usable: {first_line[:200]}",
        )
    return LintResult(success=process.returncode == 0, output=output)


async def _check_lint_delta(
    path: Path,
    pre_content: str | None,
    post_content: str | None = None,
) -> LintResult:
    """Report syntax errors introduced by an edit, preserving upstream semantics."""
    post = await _check_lint(path, post_content)
    if post.success or post.skipped or pre_content is None:
        return post

    pre = await _check_lint(path, pre_content)
    if pre.success or pre.skipped or not pre.output:
        return post

    pre_lines = {line.strip() for line in pre.output.splitlines() if line.strip()}
    post_lines = [
        line
        for line in post.output.splitlines()
        if line.strip() and line.strip() not in pre_lines
    ]
    if not post_lines:
        return LintResult(
            success=False,
            output=post.output,
            message=(
                "Pre-existing lint errors — this edit didn't introduce new ones "
                "but the file is still broken."
            ),
        )
    return LintResult(
        success=False,
        output=(
            "New lint errors introduced by this edit "
            "(pre-existing errors filtered out):\n" + "\n".join(post_lines)
        ),
    )


async def _lsp_will_handle(path: Path) -> bool:
    """Return whether the native LSP service will inspect ``path``."""
    try:
        from agent.lsp import get_service

        service = await get_service()
        return bool(service and await service.enabled_for(str(path)))
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        return False


async def _snapshot_lsp_baseline(path: Path) -> None:
    """Capture pre-write semantic diagnostics without making writes fragile."""
    try:
        from agent.lsp import get_service

        service = await get_service()
        if service is not None:
            await service.snapshot_baseline(str(path))
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.debug("LSP baseline snapshot failed", exc_info=True)


async def _maybe_lsp_diagnostics(
    path: Path,
    *,
    pre_content: str | None,
    post_content: str,
) -> str:
    """Return formatted semantic diagnostics introduced by the write."""
    try:
        from agent.lsp import get_service

        service = await get_service()
        if service is None or not await service.enabled_for(str(path)):
            return ""
        line_shift = None
        if pre_content is not None and pre_content != post_content:
            from agent.lsp.range_shift import build_line_shift

            line_shift = build_line_shift(pre_content, post_content)
        diagnostics = await service.get_diagnostics_sync(
            str(path), delta=True, line_shift=line_shift
        )
        if not diagnostics:
            return ""
        from agent.lsp.reporter import report_for_file, truncate

        block = report_for_file(str(path), diagnostics)
        return (
            truncate("LSP diagnostics introduced by this edit:\n" + block)
            if block
            else ""
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.debug("LSP diagnostics failed", exc_info=True)
        return ""


async def _write_native_result(path: Path, content: str) -> WriteResult:
    """Native-async equivalent of v2026.8.3 ``write_file`` mechanics."""
    extension = path.suffix.lower()
    in_process_linter = (
        LINTERS_INPROC.get(extension)
        if extension in _FAIL_CLOSED_INPROC_EXTS
        else None
    )
    if in_process_linter is not None:
        valid, lint_error = in_process_linter(content)
        if not valid and lint_error != "__SKIP__":
            return WriteResult(
                error=(
                    f"Refusing to write '{path}': candidate content fails "
                    f"{extension} syntax validation ({lint_error}). The file was "
                    "NOT created or modified. Fix the content and retry."
                )
            )

    try:
        existing = await _read_native_patch_content(path)
    except FileNotFoundError:
        existing = None
    except OSError:
        existing = None

    if existing is not None:
        from tools.file_operations import _detect_line_ending, _normalize_line_endings

        original_ending = _detect_line_ending(existing)
        if original_ending == "\r\n":
            content = _normalize_line_endings(content, original_ending)
        if _has_bom(existing) and not _has_bom(content):
            content = "\ufeff" + content

    await _snapshot_lsp_baseline(path)

    try:
        await _write_native_file(path, content)
    except OSError as exc:
        return WriteResult(error=f"Failed to write file: {exc}")

    try:
        verified = await _verify_native_file(path, content)
    except (OSError, UnicodeError):
        verified = None
    if verified is False:
        return WriteResult(
            error=(
                f"Post-write verification failed for {path}: on-disk content "
                "hash differs from the intended write. The write did not persist "
                "correctly — re-read the file and retry."
            )
        )

    lint_result = await _check_lint_delta(
        path,
        pre_content=(existing if extension in LINTERS_INPROC else None),
        post_content=content,
    )
    lsp_diagnostics = ""
    if lint_result.success or lint_result.skipped:
        lsp_diagnostics = await _maybe_lsp_diagnostics(
            path,
            pre_content=existing,
            post_content=content,
        )
    return WriteResult(
        bytes_written=len(content.encode("utf-8")),
        dirs_created=True,
        verified=verified,
        lint=lint_result.to_dict(),
        lsp_diagnostics=lsp_diagnostics or None,
    )


async def _handle_write_file(args, **kw):
    """Write a file atomically through ``aiofiles``."""
    task_id = kw.get("task_id") or "default"
    path = args.get("path")
    content = args.get("content")
    if not isinstance(path, str) or not path:
        return tool_error("write_file: missing required field 'path'.")
    if not isinstance(content, str):
        return tool_error("write_file: missing required string field 'content'.")
    sensitive_error = await _check_sensitive_path(path, task_id)
    if sensitive_error:
        return tool_error(sensitive_error)
    if not args.get("cross_profile", False):
        profile_warning = await _check_cross_profile_path(path, task_id)
        if profile_warning:
            return tool_error(profile_warning)
    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file content."
        )

    resolved = await _native_file_path(path, task_id)
    if isinstance(resolved, str):
        return resolved
    denied_error = await get_write_denied_error(str(resolved))
    if denied_error:
        return tool_error(denied_error)
    if not args.get("cross_profile", False):
        profile_error = await _check_cross_profile_path(str(resolved), task_id)
        if profile_error:
            return tool_error(profile_error)
    cross_warning: str | None = None
    stale_warning: str | None = None
    cwd_warning: str | None = None
    write_result: WriteResult
    try:
        async with file_state.lock_path(resolved):
            cross_warning = await file_state.check_stale(task_id, str(resolved))
            stale_warning = await _check_file_staleness(path, task_id)
            cwd_warning = await _path_resolution_warning(path, resolved, task_id)
            write_result = await _write_native_result(resolved, content)
            if not write_result.error:
                await _refresh_read_timestamp(path, task_id)
                await file_state.note_write(task_id, str(resolved))
    except OSError as exc:
        return tool_error(f"Failed to write {path}: {exc}")

    result = write_result.to_dict()
    result["resolved_path"] = str(resolved)
    if not write_result.error:
        result["files_modified"] = [str(resolved)]
        await _mark_verification_stale(
            task_id,
            [str(resolved)],
            session_id=kw.get("session_id"),
        )
    if cross_warning or stale_warning or cwd_warning:
        result["_warning"] = cross_warning or stale_warning or cwd_warning
    return json.dumps(
        result,
        ensure_ascii=False,
    )


async def _read_native_patch_content(path: Path) -> str:
    """Read a complete text file for an atomic native patch transaction."""
    async with aiofiles.open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        return await handle.read()


class _V4AFileOperations:
    """Async file-operations interface consumed by the upstream V4A applier."""

    async def read_file_raw(self, path: str) -> ReadResult:
        target = Path(path)
        try:
            raw_content = await _read_native_patch_content(target)
        except OSError as exc:
            return ReadResult(error=f"File not found: {path}" if isinstance(exc, FileNotFoundError) else str(exc))
        content, _ = _strip_bom(raw_content)
        return ReadResult(
            content=content,
            total_lines=len(content.splitlines()),
            file_size=len(raw_content.encode("utf-8")),
        )

    async def write_file(self, path: str, content: str) -> WriteResult:
        return await _write_native_result(Path(path), content)

    async def delete_file(self, path: str) -> WriteResult:
        try:
            await aiofiles.os.remove(path)
        except OSError as exc:
            return WriteResult(error=f"Failed to delete {path}: {exc}")
        return WriteResult()

    async def move_file(self, src: str, dst: str) -> WriteResult:
        try:
            await aiofiles.os.rename(src, dst)
        except OSError as exc:
            return WriteResult(error=f"Failed to move {src} -> {dst}: {exc}")
        return WriteResult()

    async def _check_lint(self, path: str) -> LintResult:
        return await _check_lint(Path(path))


async def _handle_v4a_patch(
    args: dict, task_id: str, session_id: str | None = None
) -> str:
    """Apply a V4A patch through the directly async-converted upstream applier."""
    patch_content = args.get("patch")
    if not isinstance(patch_content, str) or not patch_content.strip():
        return tool_error("patch content required")

    from tools.patch_parser import OperationType, apply_v4a_operations, parse_v4a_patch
    from tools.path_security import has_traversal_component

    operations, parse_error = parse_v4a_patch(patch_content)
    if parse_error:
        return json.dumps(
            PatchResult(error=f"Failed to parse patch: {parse_error}").to_dict(),
            ensure_ascii=False,
        )

    raw_paths: list[str] = []
    for operation in operations:
        raw_paths.append(operation.file_path)
        if operation.operation is OperationType.MOVE and operation.new_path:
            raw_paths.append(operation.new_path)

    for raw_path in raw_paths:
        if has_traversal_component(raw_path):
            return tool_error(
                f"V4A patch header contains '..' traversal: {raw_path!r}. "
                "Use the agent's cwd-relative path (no '..') or an absolute "
                "path in '*** Update File:' / '*** Add File:' / "
                "'*** Delete File:' / '*** Move File:' headers."
            )
        sensitive_error = await _check_sensitive_path(raw_path, task_id)
        if sensitive_error:
            return tool_error(sensitive_error)
        if not args.get("cross_profile", False):
            profile_error = await _check_cross_profile_path(raw_path, task_id)
            if profile_error:
                return tool_error(profile_error)

    resolved_by_raw: dict[str, Path] = {}
    for raw_path in raw_paths:
        resolved = await _native_file_path(raw_path, task_id)
        if isinstance(resolved, str):
            return resolved
        denied_error = await get_write_denied_error(str(resolved))
        if denied_error:
            return tool_error(denied_error)
        resolved_by_raw[raw_path] = resolved

    for operation in operations:
        operation.file_path = str(resolved_by_raw[operation.file_path])
        if operation.operation is OperationType.MOVE and operation.new_path:
            operation.new_path = str(resolved_by_raw[operation.new_path])

    resolved_paths = sorted(set(resolved_by_raw.values()), key=str)
    stale_warnings: list[str] = []
    async with contextlib.AsyncExitStack() as stack:
        for resolved in resolved_paths:
            await stack.enter_async_context(file_state.lock_path(resolved))

        for raw_path in raw_paths:
            resolved = resolved_by_raw[raw_path]
            warning = await file_state.check_stale(task_id, str(resolved))
            warning = warning or await _check_file_staleness(raw_path, task_id)
            warning = warning or await _path_resolution_warning(
                raw_path,
                resolved,
                task_id,
            )
            if warning:
                stale_warnings.append(warning)

        patch_result = await apply_v4a_operations(
            operations,
            _V4AFileOperations(),
        )
        result = patch_result.to_dict()
        if stale_warnings:
            result["_warning"] = (
                stale_warnings[0]
                if len(stale_warnings) == 1
                else " | ".join(stale_warnings)
            )
        if not patch_result.error:
            resolved_modified = [str(resolved_by_raw[path]) for path in raw_paths]
            result["files_modified"] = resolved_modified
            if len(resolved_modified) == 1:
                result["resolved_path"] = resolved_modified[0]
            for raw_path in raw_paths:
                resolved = resolved_by_raw[raw_path]
                await _refresh_read_timestamp(raw_path, task_id)
                await file_state.note_write(task_id, str(resolved))
            _reset_patch_failures(task_id, resolved_modified)
            await _mark_verification_stale(
                task_id,
                resolved_modified,
                session_id=session_id,
            )

    return json.dumps(result, ensure_ascii=False)


async def _handle_patch(args, **kw):
    """Apply a native async replace or V4A patch."""
    task_id = kw.get("task_id") or "default"
    mode = args.get("mode", "replace")
    if mode == "patch":
        return await _handle_v4a_patch(
            args, task_id, session_id=kw.get("session_id")
        )
    if mode != "replace":
        return tool_error(f"patch: unknown mode {mode!r}.")
    path = args.get("path")
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    if not isinstance(path, str) or not path:
        return tool_error("patch: mode='replace' requires 'path'.")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return tool_error("patch: mode='replace' requires old_string and new_string.")
    sensitive_error = await _check_sensitive_path(path, task_id)
    if sensitive_error:
        return tool_error(sensitive_error)

    resolved = await _native_file_path(path, task_id)
    if isinstance(resolved, str):
        return resolved
    denied_error = await get_write_denied_error(str(resolved))
    if denied_error:
        return tool_error(denied_error)
    if not args.get("cross_profile", False):
        profile_error = await _check_cross_profile_path(str(resolved), task_id)
        if profile_error:
            return tool_error(profile_error)
    cross_warning: str | None = None
    stale_warning: str | None = None
    cwd_warning: str | None = None
    try:
        async with file_state.lock_path(resolved):
            cross_warning = await file_state.check_stale(task_id, str(resolved))
            stale_warning = await _check_file_staleness(path, task_id)
            cwd_warning = await _path_resolution_warning(path, resolved, task_id)
            raw_content = await _read_native_patch_content(resolved)
            content, had_bom = _strip_bom(raw_content)
            from tools.file_operations import (
                _detect_line_ending,
                _normalize_line_endings,
            )

            from tools.fuzzy_match import (
                format_no_match_hint,
                fuzzy_find_and_replace,
                is_already_applied,
            )

            updated, replacement_count, _strategy, match_error = (
                fuzzy_find_and_replace(
                    content,
                    old_string,
                    new_string,
                    bool(args.get("replace_all", False)),
                )
            )
            if match_error or replacement_count == 0:
                if not is_already_applied(content, old_string, new_string):
                    error_message = match_error or (
                        f"Could not find match for old_string in {resolved}"
                    )
                    error_message += format_no_match_hint(
                        error_message,
                        replacement_count,
                        old_string,
                        content,
                    )
                    failure_count = _record_patch_failure(task_id, str(resolved))
                    extra = {}
                    if failure_count >= 3:
                        extra["_hint"] = (
                            f"This is failure #{failure_count} patching {path!r}. "
                            "Stop retrying with variations of the same old_string. "
                            "Either: (1) re-read the file fresh to verify current "
                            "content, (2) use a longer / more unique old_string with "
                            "surrounding context lines, or (3) use write_file to "
                            "replace the entire file if the targeted region is hard "
                            "to anchor."
                        )
                    return tool_error(error_message, **extra)

                _reset_patch_failures(task_id, [str(resolved)])
                result = {
                    "success": True,
                    "no_change": True,
                    "note": (
                        "File already contains the target text — the edit appears "
                        f"to be already applied to {resolved}. No write performed; "
                        "do not re-send this patch."
                    ),
                    "resolved_path": str(resolved),
                    "files_modified": [str(resolved)],
                }
                if cross_warning or stale_warning or cwd_warning:
                    result["_warning"] = (
                        cross_warning or stale_warning or cwd_warning
                    )
                return json.dumps(result)

            line_ending = _detect_line_ending(content)
            if line_ending:
                updated = _normalize_line_endings(updated, line_ending)
            extension = resolved.suffix.lower()
            in_process_linter = (
                LINTERS_INPROC.get(extension)
                if extension in _FAIL_CLOSED_INPROC_EXTS
                else None
            )
            if in_process_linter is not None:
                valid, lint_error = in_process_linter(updated)
                if not valid and lint_error != "__SKIP__":
                    return tool_error(
                        f"Failed to write changes: Refusing to write '{resolved}': "
                        f"candidate content fails {extension} syntax validation "
                        f"({lint_error}). The file was NOT created or modified. "
                        "Fix the content and retry."
                    )

            persisted = "\ufeff" + updated if had_bom and not _has_bom(updated) else updated
            await _snapshot_lsp_baseline(resolved)
            await _write_native_file(resolved, persisted)
            verified_content = await _read_native_patch_content(resolved)
            verified_content, _ = _strip_bom(verified_content)
            if (
                verified_content.replace("\r\n", "\n").replace("\r", "\n")
                != updated.replace("\r\n", "\n").replace("\r", "\n")
            ):
                return tool_error(
                    f"Post-write verification failed for {resolved}: on-disk "
                    "content differs from intended write. The patch did not "
                    "persist. Re-read the file and try again."
                )
            lint_result = await _check_lint_delta(
                resolved,
                pre_content=content,
                post_content=updated,
            )
            lsp_diagnostics = ""
            if lint_result.success or lint_result.skipped:
                lsp_diagnostics = await _maybe_lsp_diagnostics(
                    resolved,
                    pre_content=content,
                    post_content=updated,
                )
            await _refresh_read_timestamp(path, task_id)
            await file_state.note_write(task_id, str(resolved))
    except FileNotFoundError:
        return tool_error(f"File not found: {path}")
    except OSError as exc:
        _record_patch_failure(task_id, str(resolved))
        return tool_error(f"Failed to patch {path}: {exc}")

    _reset_patch_failures(task_id, [str(resolved)])
    diff = "".join(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{resolved}",
            tofile=f"b/{resolved}",
        )
    )
    result = {
        "success": True,
        "diff": diff,
        "resolved_path": str(resolved),
        "files_modified": [str(resolved)],
        "lint": lint_result.to_dict(),
    }
    if lsp_diagnostics:
        result["lsp_diagnostics"] = lsp_diagnostics
    if cross_warning or stale_warning or cwd_warning:
        result["_warning"] = cross_warning or stale_warning or cwd_warning
    await _mark_verification_stale(
        task_id,
        [str(resolved)],
        session_id=kw.get("session_id"),
    )
    return json.dumps(
        result,
        ensure_ascii=False,
    )


async def _mark_verification_stale(
    task_id: str,
    resolved_paths: list[str],
    session_id: str | None = None,
) -> None:
    """Best-effort note that successful edits made prior verification stale."""
    paths = [path for path in resolved_paths if path]
    if not paths:
        return
    try:
        from agent.verification_evidence import mark_workspace_edited

        cwd = str(Path(paths[0]).parent)
        await mark_workspace_edited(
            session_id=session_id or task_id,
            cwd=cwd,
            paths=paths,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("verification stale marker failed", exc_info=True)


_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")


def _pattern_has_regex_newline(pattern: str) -> bool:
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


async def _run_rg(arguments: list[str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        "rg",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    communicate = asyncio.create_task(process.communicate())
    communication_error: Exception | None = None
    try:
        async with asyncio.timeout(60):
            stdout, stderr = await asyncio.shield(communicate)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = await _finish_subprocess_communicate(
            process, communicate
        )
        return 124, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    except asyncio.CancelledError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await _finish_subprocess_communicate(process, communicate)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Search cleanup after cancellation failed", exc_info=True)
        raise
    except Exception as exc:
        communication_error = exc
    if communication_error is not None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await _finish_subprocess_communicate(process, communicate)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        raise communication_error
    return (
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _count_rg_matches(output: str) -> tuple[int, int]:
    total = 0
    files = 0
    for line in output.splitlines():
        _path, separator, count = line.rpartition(":")
        if separator and count.isdigit():
            total += int(count)
            files += 1
    return total, files


async def _zero_match_hint(
    pattern: str,
    paths: list[Path],
    file_glob: str | None,
) -> str | None:
    common = ["--count-matches"]
    if file_glob:
        common.extend(["--glob", file_glob])
    path_args = [str(path) for path in paths]

    _code, output, _error = await _run_rg(["-i", *common, pattern, *path_args])
    total, files = _count_rg_matches(output)
    if total:
        return (
            f"0 exact matches, but {total} case-insensitive match(es) in "
            f"{files} file(s) — the pattern's casing may be wrong."
        )

    _code, output, _error = await _run_rg(
        ["--hidden", "--no-ignore", *common, pattern, *path_args]
    )
    total, files = _count_rg_matches(output)
    if total:
        return (
            f"0 matches in visible files, but {total} match(es) in {files} "
            "hidden or gitignored file(s) — these are excluded by default."
        )

    if re.search(r"[.\[\](){}?*+^$\\|]", pattern):
        _code, output, _error = await _run_rg(
            ["-F", *common, pattern, *path_args]
        )
        total, _files = _count_rg_matches(output)
        if total:
            return (
                f"0 regex matches, but {total} literal match(es) — the pattern "
                "contains regex metacharacters that likely need escaping."
            )
    return None


async def _handle_search_files(args, **kw):
    """Search files using a native subprocess and preserve async cancellation."""
    from tools.tool_output_limits import _refresh_tool_output_limits

    await _refresh_tool_output_limits()
    task_id = kw.get("task_id") or "default"
    pattern = args.get("pattern", "")
    if not isinstance(pattern, str) or not pattern:
        return tool_error("search_files: missing required field 'pattern'.")
    target = {"grep": "content", "find": "files"}.get(
        args.get("target", "content"), args.get("target", "content")
    )
    offset, limit = normalize_search_pagination(
        args.get("offset", 0),
        args.get("limit", 50),
    )
    search_key = (
        "search", pattern, str(args.get("path", ".")), target,
        args.get("file_glob"), offset, limit, args.get("output_mode", "content"),
        int(args.get("context", 0) or 0),
    )
    state = _read_state(task_id)
    with _read_tracker_lock:
        if state.get("last_key") == search_key:
            state["consecutive"] += 1
        else:
            state["last_key"] = search_key
            state["consecutive"] = 1
        count = state["consecutive"]
    if count >= 4:
        return tool_error(
            f"BLOCKED: You have run this exact search {count} times in a row. "
            "The results have NOT changed. STOP re-searching and proceed with "
            "your task.",
            pattern=pattern,
            already_searched=count,
        )
    raw_path = str(args.get("path", "."))
    resolved = await _native_file_path(raw_path, task_id)
    if isinstance(resolved, str):
        return resolved
    resolved_str = str(resolved)
    cached_not_found = await _check_not_found_cache(
        "search", resolved_str, task_id
    )
    if cached_not_found is not None:
        return cached_not_found
    if await aiofiles.os.path.exists(resolved):
        requested_paths = [raw_path]
    else:
        requested_paths = [
            part for part in re.split(r"[\s,]+", raw_path.strip()) if part
        ]
    existing_paths: list[Path] = []
    missing_paths: list[str] = []
    for requested_path in requested_paths:
        candidate = await _native_file_path(requested_path, task_id)
        if isinstance(candidate, str):
            return candidate
        if await aiofiles.os.path.exists(candidate):
            block_error = await get_read_block_error(str(candidate))
            if block_error:
                return tool_error(block_error)
            existing_paths.append(candidate)
        else:
            missing_paths.append(str(candidate))
    if not existing_paths:
        error_json = tool_error(f"Path not found: {raw_path}")
        _record_not_found("search", resolved_str, task_id, error_json)
        return error_json

    output_mode = args.get("output_mode", "content")
    context = max(0, int(args.get("context", 0) or 0))
    if target == "files":
        glob_pattern = f"*{pattern}" if "/" not in pattern and not pattern.startswith("*") else pattern
        command = [
            "--files",
            "--sortr=modified",
            "--glob",
            glob_pattern,
            *map(str, existing_paths),
        ]
    else:
        command = [
            "--line-number",
            "--no-heading",
            "--with-filename",
            "--color",
            "never",
        ]
        multiline = _pattern_has_regex_newline(pattern)
        if multiline:
            command.append("--multiline")
        if context:
            command.extend(["-C", str(context)])
        file_glob = args.get("file_glob")
        if isinstance(file_glob, str) and file_glob:
            command.extend(["--glob", file_glob])
        if output_mode == "files_only":
            command.append("-l")
        elif output_mode == "count":
            command.append("-c")
        command.extend(["--regexp", pattern, *map(str, existing_paths)])

    try:
        returncode, stdout, stderr = await _run_rg(command)
    except FileNotFoundError:
        return tool_error("search_files requires ripgrep (rg), which is not installed.")

    if target == "files" and returncode not in {0, 1, 124} and not stdout.strip():
        try:
            returncode, stdout, stderr = await _run_rg(
                ["--files", "--glob", glob_pattern, *map(str, existing_paths)]
            )
        except FileNotFoundError:
            return tool_error("search_files requires ripgrep (rg), which is not installed.")

    if returncode not in {0, 1, 124} and not stdout.strip():
        message = stderr.strip() or stdout.strip() or "Search error"
        return json.dumps(
            SearchResult(error=f"Search failed: {message}").to_dict(),
            ensure_ascii=False,
        )

    limit_reason = "search_timeout" if returncode == 124 else None
    all_lines = stdout.splitlines()
    all_lines, omitted = await _filter_search_output_lines(all_lines, task_id)
    fetch_limit = offset + limit + (200 if context else 0)
    fetched_lines = all_lines[:fetch_limit]
    truncated_by_limit = len(all_lines) >= fetch_limit
    warning: str | None = None
    if len(requested_paths) > 1:
        warning = (
            f"path contained {len(requested_paths)} entries; searched "
            f"{len(existing_paths)} that exist"
        )
        if missing_paths:
            warning += "; skipped missing: " + ", ".join(missing_paths[:3])
            if len(missing_paths) > 3:
                warning += f" (+{len(missing_paths) - 3} more)"
    if target != "files" and multiline:
        multiline_note = (
            "Pattern contains \\n — multiline mode (-U) was enabled automatically "
            "so the regex can match across line boundaries."
        )
        warning = f"{warning} {multiline_note}" if warning else multiline_note
    if target != "files" and not fetched_lines:
        try:
            hint = await _zero_match_hint(
                pattern,
                existing_paths,
                file_glob if isinstance(file_glob, str) else None,
            )
        except (FileNotFoundError, TimeoutError):
            hint = None
        if hint:
            warning = f"{warning} {hint}" if warning else hint

    if target == "files" or output_mode == "files_only":
        page = fetched_lines[offset:offset + limit]
        search_result = SearchResult(
            files=page,
            total_count=len(fetched_lines),
            truncated=(
                truncated_by_limit or bool(limit_reason)
                if target == "files"
                else bool(limit_reason)
            ),
            limit_reason=limit_reason,
            warning=warning,
        )
    elif output_mode == "count":
        counts: dict[str, int] = {}
        for line in fetched_lines:
            path_part, separator, count_text = line.rpartition(":")
            if separator and count_text.isdigit():
                counts[path_part] = int(count_text)
        search_result = SearchResult(
            counts=counts,
            total_count=sum(counts.values()),
            truncated=bool(limit_reason),
            limit_reason=limit_reason,
            warning=warning,
        )
    else:
        match_pattern = re.compile(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$")
        matches: list[SearchMatch] = []
        for line in fetched_lines:
            if not line or line == "--":
                continue
            match = match_pattern.match(line)
            if match:
                matches.append(
                    SearchMatch(
                        path=(match.group(1) or "") + match.group(2),
                        line_number=int(match.group(3)),
                        content=redact_sensitive_text(
                            match.group(4)[:500],
                            file_read=True,
                        ),
                    )
                )
                continue
            if context:
                parsed = _parse_search_context_line(line)
                if parsed:
                    matches.append(
                        SearchMatch(
                            path=parsed[0],
                            line_number=parsed[1],
                            content=redact_sensitive_text(
                                parsed[2][:500],
                                file_read=True,
                            ),
                        )
                    )
        total = len(matches)
        search_result = SearchResult(
            matches=matches[offset:offset + limit],
            total_count=total,
            truncated=(
                total > offset + limit
                or bool(limit_reason)
            ),
            limit_reason=limit_reason,
            warning=warning,
        )

    result = search_result.to_dict(densify=True)
    if omitted:
        result["_omitted"] = (
            f"{omitted} result(s) omitted because they target credential, "
            "token, cache, or secret-bearing environment files."
        )
    if count >= 3:
        result["_warning"] = (
            f"You have run this exact search {count} times consecutively. "
            "The results have not changed. Use the information you already have."
        )
    result_json = json.dumps(result, ensure_ascii=False)
    if result.get("truncated"):
        next_offset = offset + limit
        result_json += (
            f"\n\n[Hint: Results truncated. Use offset={next_offset} to see "
            "more, or narrow with a more specific pattern or file_glob.]"
        )
    return result_json


async def read_file_tool(
    path: str,
    offset: int = 1,
    limit: int = 2000,
    task_id: str = "default",
) -> str:
    """Read a file through the native async implementation."""
    return await _handle_read_file(
        {"path": path, "offset": offset, "limit": limit},
        task_id=task_id,
    )


async def write_file_tool(
    path: str,
    content: str,
    task_id: str = "default",
    cross_profile: bool = False,
    session_id: str | None = None,
) -> str:
    """Write a file through the native async implementation."""
    return await _handle_write_file(
        {
            "path": path,
            "content": content,
            "cross_profile": cross_profile,
        },
        task_id=task_id,
        session_id=session_id,
    )


async def patch_tool(
    mode: str = "replace",
    path: str | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
    replace_all: bool = False,
    patch: str | None = None,
    task_id: str = "default",
    cross_profile: bool = False,
    session_id: str | None = None,
) -> str:
    """Patch files through the native async implementation."""
    return await _handle_patch(
        {
            "mode": mode,
            "path": path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
            "patch": patch,
            "cross_profile": cross_profile,
        },
        task_id=task_id,
        session_id=session_id,
    )


async def search_tool(
    pattern: str,
    target: str = "content",
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
    offset: int = 0,
    output_mode: str = "content",
    context: int = 0,
    task_id: str = "default",
) -> str:
    """Search files through the native async implementation."""
    return await _handle_search_files(
        {
            "pattern": pattern,
            "target": target,
            "path": path,
            "file_glob": file_glob,
            "limit": limit,
            "offset": offset,
            "output_mode": output_mode,
            "context": context,
        },
        task_id=task_id,
    )


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs, emoji="📖", max_result_size_chars=100_000)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs, emoji="✍️", max_result_size_chars=100_000)
registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs, emoji="🔧", max_result_size_chars=100_000)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs, emoji="🔎", max_result_size_chars=100_000)
