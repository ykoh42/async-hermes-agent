#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import asyncio
import contextlib
import difflib
import json
import logging
import os
import posixpath
import sys
import threading
import uuid
from pathlib import Path, PurePosixPath

import aiofiles
import aiofiles.os

from agent.file_safety import get_read_block_error
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    MAX_LINE_LENGTH,
    _strip_bom,
    normalize_read_pagination,
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
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from the earlier read_file "
    "result in this conversation is still current — refer to that instead "
    "of re-reading."
)


def _cap_read_tracker_data(task_data: dict) -> None:
    """Bound per-task read metadata so long sessions cannot grow unbounded."""
    history = task_data.get("read_history")
    if history is not None:
        while len(history) > _READ_HISTORY_CAP:
            history.pop()
    for key, cap in (("dedup", _DEDUP_CAP), ("read_timestamps", _READ_TIMESTAMPS_CAP)):
        values = task_data.get(key)
        if values is not None:
            while len(values) > cap:
                values.pop(next(iter(values)))


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


def _expand_tilde(path: str) -> str:
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

        home = get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    return os.path.expanduser(path)


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


def _get_max_read_chars() -> int:
    """Return the configured max characters per file read.

    Reads ``file_read_max_chars`` from config.yaml on first call, caches
    the result for the lifetime of the process.  Falls back to the
    built-in default if the config is missing or invalid.
    """
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
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


def _resolve_path(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve a path relative to TERMINAL_CWD (the worktree base directory)
    instead of the main repository root.
    """
    return _resolve_path_for_task(filepath, task_id)


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
    """Best-effort terminal backend type for path-resolution decisions."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        try:
            container_key = _resolve_container_task_id(task_id)
        except Exception:
            container_key = task_id
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)
        if env is not None:
            name = env.__class__.__name__.lower()
            if "local" in name:
                return "local"
            if "ssh" in name:
                return "ssh"
            if "docker" in name:
                return "docker"
            if "singularity" in name:
                return "singularity"
            if "modal" in name:
                return "modal"
            if "daytona" in name:
                return "daytona"
        cfg = _get_env_config()
        return str(cfg.get("env_type") or os.getenv("TERMINAL_ENV") or "local").lower()
    except Exception:
        return str(os.getenv("TERMINAL_ENV") or "local").lower()


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


def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Normalize a cwd candidate to an absolute, sentinel-free anchor.

    Returns the expanded path only when *raw* is non-empty, not a sentinel (see
    ``_TERMINAL_CWD_SENTINELS``), and absolute. A relative anchor is meaningless
    without knowing which cwd it is relative to — exactly the ambiguity that
    misroutes worktree edits — so relative/sentinel/empty values yield ``None``.
    """
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = _expand_tilde(raw)
    if not os.path.isabs(expanded):
        return None
    return expanded


def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real directory anchor.

    Sentinel values (see ``_TERMINAL_CWD_SENTINELS``) and relative paths are
    rejected — a relative anchor is meaningless without knowing which cwd it is
    relative to, which is exactly the ambiguity that misroutes worktree edits.
    Only an absolute, sentinel-free value is honored.
    """
    return _sentinel_free_abs_cwd(os.environ.get("TERMINAL_CWD"))


def _registered_task_cwd_override(task_id: str = "default") -> str | None:
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

    return _sentinel_free_abs_cwd(overrides.get("cwd"))


def _authoritative_workspace_root(task_id: str = "default") -> str | None:
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

        recorded = get_session_cwd(task_id)
    except Exception:
        recorded = None
    if recorded:
        return recorded
    registered = _registered_task_cwd_override(task_id)
    if registered:
        return registered
    return _configured_terminal_cwd()


def _resolve_base_dir(
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
    root = _authoritative_workspace_root(task_id)
    if container_paths is None:
        container_paths = _uses_container_paths(task_id)
    if root:
        base_text = _expand_tilde(root)
    else:
        base_text = os.getcwd()
    if container_paths:
        if not posixpath.isabs(base_text):
            base_text = posixpath.join(os.getcwd(), base_text)
        return _normalize_without_host_deref(base_text)
    # Git Bash ``pwd -P`` reports ``/c/Users/...``; translate before Path so
    # relative file-tool paths don't anchor under a nonexistent ``\\c\\Users``.
    from tools.environments.local import _msys_to_windows_path

    base_text = _msys_to_windows_path(base_text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(base_text):
            base_text = ntpath.join(os.getcwd(), base_text)
        return Path(ntpath.normpath(base_text))
    base = Path(base_text)
    if not base.is_absolute():
        # Last-resort anchoring: a live cwd should already be absolute, but if a
        # terminal backend ever reports a relative cwd, anchor it to the process
        # cwd once, here, so the result no longer depends on cwd at resolve().
        base = Path(os.getcwd()) / base
    return base.resolve()


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory.

    See :func:`_resolve_base_dir` for how the base is chosen. Absolute input
    paths are returned resolved-but-unanchored.

    On native Windows, Git Bash / MSYS drive paths (``/c/Users/...``) are
    translated to ``C:\\Users\\...`` before resolution so file tools don't
    treat them as relative ``\\c\\Users\\...`` under the process cwd.
    """
    container_paths = _uses_container_paths(task_id)
    if container_paths:
        expanded = _expand_tilde(filepath)
        if posixpath.isabs(expanded):
            return _normalize_without_host_deref(expanded)
        resolved = _resolve_base_dir(task_id, container_paths=True) / expanded
        return _normalize_without_host_deref(resolved)

    # Host paths only — never rewrite Linux paths inside a container/WSL env.
    from tools.environments.local import _msys_to_windows_path

    expanded = _expand_tilde(_msys_to_windows_path(filepath))
    if sys.platform == "win32":
        import ntpath

        if ntpath.isabs(expanded):
            return Path(ntpath.normpath(expanded))
        joined = ntpath.join(str(_resolve_base_dir(task_id, container_paths=False)), expanded)
        return Path(ntpath.normpath(joined))

    p = Path(expanded)
    if p.is_absolute():
        return p.resolve()
    resolved = _resolve_base_dir(task_id, container_paths=False) / p
    return resolved.resolve()


def _path_resolution_warning(filepath: str, resolved: Path, task_id: str = "default") -> str | None:
    """Warn when a relative path resolved OUTSIDE the task's workspace root.

    Surfaces the worktree-cwd divergence the moment it would matter: if the
    agent passes a relative path but it resolves under a directory that is not
    the workspace root (i.e. the edit is about to land in a different checkout
    than the one the agent is working in), return a message naming the absolute
    target. ``None`` when the path is absolute, the base is unknown, or the
    resolved path is correctly under the workspace root.

    The workspace root is the live terminal cwd when known, else a registered
    task/session cwd override, else a sentinel-free absolute ``$TERMINAL_CWD``
    — so a worktree or Desktop session whose terminal registry is still empty
    (no ``cd`` run yet) is warned on the very first write.
    """
    try:
        if Path(_expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = _authoritative_workspace_root(task_id)
        if not workspace_root:
            return None  # No authoritative workspace root to compare against.
        if _uses_container_paths(task_id):
            root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
        else:
            root = Path(_expand_tilde(workspace_root)).resolve()
        # Is `resolved` inside `root`?
        try:
            resolved.relative_to(root)
            return None  # Inside the workspace — expected.
        except ValueError:
            return (
                f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
                f"OUTSIDE the active workspace ({str(root)!r}). The edit will land in "
                f"a different directory than the terminal's cwd. If this is not "
                f"intended (e.g. a git-worktree session writing into the main "
                f"checkout), pass an absolute path under the workspace instead."
            )
    except Exception:
        return None


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd paths that can hang reads."""
    normalized = os.path.normpath(_expand_tilde(path))
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


def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """Return True if the path would hang the process (infinite output or blocking input).

    Check the literal path first so aliases like /dev/stdin are caught before
    they resolve to terminal-specific paths. Then check each symlink hop before
    the final resolved path so aliases to devices cannot bypass the guard.
    """
    expanded = _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        resolved = os.path.normpath(os.path.realpath(normalized))
    except (OSError, ValueError):
        return False
    if _is_blocked_device_path(resolved):
        return True
    return False






# Paths that file tools should refuse to write to without going through the
# terminal tool's approval system.  These match prefixes after os.path.realpath.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False


def _get_hermes_config_resolved() -> str | None:
    """Return the resolved absolute path of the Hermes config file (cached)."""
    global _hermes_config_resolved, _hermes_config_resolved_loaded
    if _hermes_config_resolved_loaded:
        return _hermes_config_resolved
    _hermes_config_resolved_loaded = True
    try:
        from hermes_cli.config import get_config_path
        _hermes_config_resolved = str(get_config_path().resolve())
    except Exception:
        try:
            _hermes_config_resolved = str(Path(_expand_tilde("~/.hermes/config.yaml")).resolve())
        except Exception:
            _hermes_config_resolved = None
    return _hermes_config_resolved


def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        resolved = filepath
    normalized = os.path.normpath(_expand_tilde(filepath))
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
    hermes_config = _get_hermes_config_resolved()
    # macOS resolves ``/tmp`` through the ``/private`` symlink.  Canonicalize
    # both values so the config guard is stable regardless of the spelling the
    # caller used.
    canonical_resolved = os.path.realpath(resolved)
    canonical_normalized = os.path.realpath(normalized)
    canonical_config = os.path.realpath(hermes_config) if hermes_config else None
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


def _get_container_mirror_prefix_for_task(task_id: str = "default") -> str | None:
    """Return the container-side Hermes mirror prefix for Docker file tools."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        container_key = _resolve_container_task_id(task_id)
    except Exception:
        return None

    try:
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)

        if env is not None:
            if env.__class__.__name__ == "DockerEnvironment" and bool(
                getattr(env, "_persistent", False)
            ):
                return "/root/.hermes"
            return None

        config = _get_env_config()
    except Exception:
        return None

    if config.get("env_type") == "docker" and config.get("container_persistent", True):
        return "/root/.hermes"
    return None


def _check_cross_profile_path(filepath: str, task_id: str = "default") -> str | None:
    """Return a soft-guard warning when ``filepath`` lands in another Hermes
    profile's scoped area, a host-side sandbox-mirror of authoritative profile
    state, or the Docker container's sandbox mirror of Hermes state.

    Three detectors run in order:

    * cross-profile — writes that hit another profile's
      ``skills/plugins/cron/memories`` directory.
    * sandbox-mirror (#32049) — writes that hit the
      ``…/sandboxes/<backend>/<task>/home/.hermes/…`` mirror created by a
      non-local terminal backend (Docker, Daytona, etc.), where the host
      Hermes process never reads the mirror and the authoritative file is
      left untouched.
    * container-mirror (#32049 follow-up) — writes from inside a Docker
      container whose bind-mounted home strips the ``sandboxes/`` prefix, so
      the agent sees a plain ``/root/.hermes/…`` path.

    Returns ``None`` when the write is in-scope or outside Hermes scope.
    All detectors are soft guards — the agent can override any by
    passing ``cross_profile=True`` to its write tool after explicit user
    direction. Defense-in-depth, NOT a security boundary — the terminal
    tool runs as the same OS user and can write any of these paths
    directly. See ``agent/file_safety.classify_cross_profile_target``,
    ``classify_sandbox_mirror_target`` and ``classify_container_mirror_target``
    for the detection rules.
    """
    try:
        from agent.file_safety import (
            get_container_mirror_warning,
            get_cross_profile_warning,
            get_sandbox_mirror_warning,
        )
    except Exception:
        # Fail open on import error — the existing sensitive-path guard
        # plus the write_denied list still apply.
        return None

    # Resolve via the task's cwd so a relative ``skills/foo/SKILL.md``
    # in a session that cd'd into ``~/.hermes/profiles/other/`` is
    # classified against the right base.
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        resolved = filepath

    warning = get_cross_profile_warning(resolved)
    if warning is not None:
        return warning

    warning = get_sandbox_mirror_warning(resolved)
    if warning is not None:
        return warning

    return get_container_mirror_warning(
        resolved,
        mirror_prefix=_get_container_mirror_prefix_for_task(task_id),
    )






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


def notify_other_tool_call(task_id: str = "default") -> None:
    """Break a consecutive read/stub loop after another tool runs."""
    with _read_tracker_lock:
        state = _read_tracker.get(task_id)
        if state:
            state["last_key"] = None
            state["consecutive"] = 0
            state.setdefault("dedup_hits", {}).clear()


def _invalidate_dedup_for_path(filepath: str, task_id: str) -> None:
    try:
        resolved = str(_resolve_path(filepath, task_id))
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


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """Refresh mtime/dedup state after a successful write or patch."""
    _invalidate_dedup_for_path(filepath, task_id)
    try:
        resolved = str(_resolve_path(filepath, task_id))
        mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        state = _read_tracker.get(task_id)
        if state:
            state.setdefault("read_timestamps", {})[resolved] = mtime
            _cap_read_tracker_data(state)




def _search_result_read_block_error(
    path: str,
    task_id: str = "default",
) -> str | None:
    """Apply the credential read guard to a search result path."""
    try:
        resolved = _resolve_path_for_task(path, task_id)
    except (OSError, ValueError, RuntimeError):
        return get_read_block_error(path)
    return get_read_block_error(str(resolved))


def _filter_search_output_lines(
    lines: list[str],
    task_id: str,
) -> tuple[list[str], int]:
    """Remove credential-bearing ``rg`` rows before returning them to the model."""
    safe: list[str] = []
    omitted = 0
    for line in lines:
        candidate = line.split(":", 1)[0] if ":" in line else line
        if _search_result_read_block_error(candidate, task_id):
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
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 500, max: 2000)", "default": 500, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hermes profile's skills/plugins/cron/memories — by default these writes are blocked with a warning because they affect a different profile than the one this session is running under.",
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
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hermes profile's skills/plugins/cron/memories.",
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

_async_file_locks: dict[str, asyncio.Lock] = {}
_async_file_locks_guard = asyncio.Lock()


async def _async_file_lock(path: Path) -> asyncio.Lock:
    key = str(path)
    async with _async_file_locks_guard:
        return _async_file_locks.setdefault(key, asyncio.Lock())


def _native_file_path(path: str, task_id: str) -> Path | str:
    """Resolve a host path or return the user-facing reason it cannot run."""
    try:
        resolved = _resolve_path_for_task(path, task_id)
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
        resolved = str(_resolve_path(path, task_id))
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
    _invalidate_dedup_for_path(path, task_id)
    try:
        resolved = str(_resolve_path(path, task_id))
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
    task_id = kw.get("task_id") or "default"
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        return tool_error("read_file: missing required field 'path'.")

    offset, limit = normalize_read_pagination(args.get("offset", 1), args.get("limit", 500))
    device_base = None if Path(_expand_tilde(path)).is_absolute() else _resolve_base_dir(task_id)
    if _is_blocked_device(path, base_dir=device_base):
        return tool_error(
            f"Cannot read '{path}': this is a device file that would block or produce infinite output."
        )

    resolved = _native_file_path(path, task_id)
    if isinstance(resolved, str):
        return resolved
    if has_binary_extension(str(resolved)):
        return tool_error(
            f"Cannot read binary file '{path}' ({resolved.suffix.lower()}). "
            "Use vision_analyze for images, or terminal to inspect binary files."
        )
    block_error = get_read_block_error(str(resolved))
    if block_error:
        return tool_error(block_error)

    resolved_str = str(resolved)
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
        return tool_error(f"File not found: {path}")
    except OSError as exc:
        return tool_error(f"Failed to read {path}: {exc}")

    if b"\x00" in data[:1000]:
        return tool_error(
            f"Cannot read binary file '{path}'. Use an appropriate binary tool instead."
        )

    text = data.decode("utf-8", errors="replace")
    if offset == 1:
        text, _ = _strip_bom(text)
    lines = text.splitlines()
    page = lines[offset - 1:offset - 1 + limit]
    numbered = "\n".join(
        f"{line_number}|{line[:MAX_LINE_LENGTH]}"
        for line_number, line in enumerate(page, start=offset)
    )
    numbered, lines_kept, char_truncated = _truncate_to_char_budget(
        numbered, _DEFAULT_MAX_READ_CHARS
    )
    next_offset = offset + lines_kept
    truncated = len(lines) >= next_offset or char_truncated
    result = {
        "content": redact_sensitive_text(numbered, file_read=True) if numbered else "",
        "total_lines": len(lines),
        "file_size": len(data),
        "truncated": truncated,
    }
    if truncated:
        result["next_offset"] = next_offset
        result["hint"] = (
            f"Use offset={next_offset} to continue reading "
            f"(showing {offset}-{max(offset, next_offset - 1)} of {len(lines)} lines)."
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
        file_state.record_read(task_id, resolved_str, partial=offset > 1 or truncated)
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
    try:
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Unable to create parent directory {path.parent}: {exc}") from exc

    temporary = path.parent / f".{path.name}.hermes-{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(temporary, "w", encoding="utf-8", newline="") as handle:
            await handle.write(content)
            await handle.flush()
        await aiofiles.os.replace(temporary, path)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass


async def _handle_write_file(args, **kw):
    """Write a file atomically through ``aiofiles``."""
    task_id = kw.get("task_id") or "default"
    path = args.get("path")
    content = args.get("content")
    if not isinstance(path, str) or not path:
        return tool_error("write_file: missing required field 'path'.")
    if not isinstance(content, str):
        return tool_error("write_file: missing required string field 'content'.")
    sensitive_error = _check_sensitive_path(path, task_id)
    if sensitive_error:
        return tool_error(sensitive_error)
    if not args.get("cross_profile", False):
        profile_warning = _check_cross_profile_path(path, task_id)
        if profile_warning:
            return tool_error(profile_warning)
    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file content."
        )

    resolved = _native_file_path(path, task_id)
    if isinstance(resolved, str):
        return resolved
    cross_warning = file_state.check_stale(task_id, str(resolved))
    stale_warning = await _check_file_staleness(path, task_id)
    lock = await _async_file_lock(resolved)
    try:
        async with lock:
            await _write_native_file(resolved, content)
    except OSError as exc:
        return tool_error(f"Failed to write {path}: {exc}")

    await _refresh_read_timestamp(path, task_id)
    file_state.note_write(task_id, str(resolved))
    result = {
        "bytes_written": len(content.encode("utf-8")),
        "resolved_path": str(resolved),
        "files_modified": [str(resolved)],
    }
    if cross_warning or stale_warning:
        result["_warning"] = cross_warning or stale_warning
    return json.dumps(
        result,
        ensure_ascii=False,
    )


async def _read_native_patch_content(path: Path) -> str:
    """Read a complete text file for an atomic native patch transaction."""
    async with aiofiles.open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        return await handle.read()


def _native_v4a_error(message: str) -> str:
    return tool_error(f"V4A patch validation failed (no files were modified): {message}")


async def _apply_native_v4a_update(content: str, operation) -> tuple[str, str | None]:
    """Apply one parsed V4A update in memory.

    Parsing and fuzzy matching are CPU-only.  Keeping them in this helper lets
    the surrounding transaction validate every operation before the first
    filesystem mutation, while all reads and writes remain native async.
    """
    from tools.fuzzy_match import format_no_match_hint, fuzzy_find_and_replace

    updated = content
    changed = False
    for hunk_index, hunk in enumerate(operation.hunks, start=1):
        search_lines = [line.content for line in hunk.lines if line.prefix in {" ", "-"}]
        replacement_lines = [line.content for line in hunk.lines if line.prefix in {" ", "+"}]
        if search_lines == replacement_lines:
            continue
        if search_lines:
            search = "\n".join(search_lines)
            replacement = "\n".join(replacement_lines)
            next_content, count, _strategy, error = fuzzy_find_and_replace(
                updated, search, replacement, replace_all=False
            )
            if count == 0 and hunk.context_hint:
                hint_position = updated.find(hunk.context_hint)
                if hint_position >= 0:
                    start = max(0, hint_position - 500)
                    end = min(len(updated), hint_position + 2000)
                    window, count, _strategy, error = fuzzy_find_and_replace(
                        updated[start:end], search, replacement, replace_all=False
                    )
                    if count:
                        next_content = updated[:start] + window + updated[end:]
            if not count:
                detail = error or "could not find a unique match"
                return content, (
                    f"{operation.file_path}: hunk {hunk_index} could not be applied: "
                    f"{detail}{format_no_match_hint(detail, 0, search, updated)}"
                )
            updated = next_content
            changed = True
            continue

        insert = "\n".join(replacement_lines)
        if not insert:
            continue
        hint = hunk.context_hint
        if hint:
            occurrences = updated.count(hint)
            if occurrences > 1:
                return content, (
                    f"{operation.file_path}: addition-only hunk context hint "
                    f"{hint!r} is ambiguous ({occurrences} occurrences)"
                )
            if occurrences == 1:
                position = updated.find(hint)
                line_end = updated.find("\n", position)
                if line_end >= 0:
                    updated = updated[:line_end + 1] + insert + "\n" + updated[line_end + 1:]
                else:
                    updated = updated + "\n" + insert
            else:
                updated = updated.rstrip("\n") + "\n" + insert + "\n"
        else:
            updated = updated.rstrip("\n") + "\n" + insert + "\n"
        changed = True
    if not changed:
        return content, f"{operation.file_path}: update contains no changes"
    return updated, None


async def _handle_v4a_patch(args: dict, task_id: str) -> str:
    """Validate then apply a V4A patch using only async local-file I/O.

    The original V4A parser is retained as the canonical syntax parser.  Its
    synchronous *backend adapter* is intentionally not used: this transaction
    resolves paths once, locks every target in lexical order, computes the full
    final file state in memory, and only then publishes atomic replacements.
    """
    patch_content = args.get("patch")
    if not isinstance(patch_content, str) or not patch_content.strip():
        return tool_error("patch: mode='patch' requires non-empty 'patch' content.")

    from tools.patch_parser import OperationType, parse_v4a_patch
    from tools.path_security import has_traversal_component

    operations, parse_error = parse_v4a_patch(patch_content)
    if parse_error:
        return _native_v4a_error(parse_error)
    if not operations:
        return _native_v4a_error("patch contains no operations")

    raw_paths: list[str] = []
    for operation in operations:
        raw_paths.append(operation.file_path)
        if operation.operation is OperationType.MOVE and operation.new_path:
            raw_paths.append(operation.new_path)
    for raw_path in raw_paths:
        if has_traversal_component(raw_path):
            return tool_error(
                f"V4A patch header contains '..' traversal: {raw_path!r}. "
                "Use an absolute or cwd-relative path without '..'."
            )
        sensitive_error = _check_sensitive_path(raw_path, task_id)
        if sensitive_error:
            return tool_error(sensitive_error)
        if not args.get("cross_profile", False):
            profile_error = _check_cross_profile_path(raw_path, task_id)
            if profile_error:
                return tool_error(profile_error)

    resolved_by_raw: dict[str, Path] = {}
    for raw_path in raw_paths:
        resolved = _native_file_path(raw_path, task_id)
        if isinstance(resolved, str):
            return resolved
        resolved_by_raw[raw_path] = resolved

    paths = sorted(set(resolved_by_raw.values()), key=str)
    locks = [await _async_file_lock(path) for path in paths]
    async with contextlib.AsyncExitStack() as stack:
        for lock in locks:
            await stack.enter_async_context(lock)

        original: dict[Path, str | None] = {}
        state: dict[Path, str | None] = {}

        async def load(path: Path) -> str | None:
            if path not in state:
                try:
                    content = await _read_native_patch_content(path)
                except FileNotFoundError:
                    content = None
                state[path] = content
                original[path] = content
            return state[path]

        for operation in operations:
            source = resolved_by_raw[operation.file_path]
            if operation.operation is OperationType.ADD:
                if await load(source) is not None:
                    return _native_v4a_error(f"{operation.file_path}: destination already exists")
                state[source] = "\n".join(
                    line.content
                    for hunk in operation.hunks
                    for line in hunk.lines
                    if line.prefix == "+"
                )
                continue
            if operation.operation is OperationType.DELETE:
                if await load(source) is None:
                    return _native_v4a_error(f"{operation.file_path}: file not found for deletion")
                state[source] = None
                continue
            if operation.operation is OperationType.MOVE:
                destination = resolved_by_raw[operation.new_path]
                source_content = await load(source)
                if source_content is None:
                    return _native_v4a_error(f"{operation.file_path}: source file not found for move")
                if await load(destination) is not None:
                    return _native_v4a_error(
                        f"{operation.new_path}: destination already exists — move would overwrite"
                    )
                state[destination] = source_content
                state[source] = None
                continue
            current = await load(source)
            if current is None:
                return _native_v4a_error(f"{operation.file_path}: file not found for update")
            replacement, error = await _apply_native_v4a_update(current, operation)
            if error:
                return _native_v4a_error(error)
            state[source] = replacement

        changed_paths = [path for path in paths if state.get(path) != original.get(path)]
        if not changed_paths:
            return _native_v4a_error("patch contains no changes")
        try:
            for path in changed_paths:
                content = state[path]
                if content is None:
                    await aiofiles.os.remove(path)
                else:
                    await _write_native_file(path, content)
        except OSError as exc:
            return tool_error(
                f"V4A apply failed after validation: {exc}. Run git diff to inspect state."
            )

    for path in changed_paths:
        await _refresh_read_timestamp(str(path), task_id)
        file_state.note_write(task_id, str(path))
    _reset_patch_failures(task_id, [str(path) for path in changed_paths])
    diffs: list[str] = []
    for path in changed_paths:
        before = original.get(path) or ""
        after = state.get(path) or ""
        fromfile = f"a/{path}" if original.get(path) is not None else "/dev/null"
        tofile = f"b/{path}" if state.get(path) is not None else "/dev/null"
        diffs.append("".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=fromfile, tofile=tofile,
        )))
    return json.dumps(
        {
            "success": True,
            "diff": "".join(diffs),
            "files_modified": [str(path) for path in changed_paths],
        },
        ensure_ascii=False,
    )


async def _handle_patch(args, **kw):
    """Apply a native async replace or V4A patch."""
    task_id = kw.get("task_id") or "default"
    mode = args.get("mode", "replace")
    if mode == "patch":
        return await _handle_v4a_patch(args, task_id)
    if mode != "replace":
        return tool_error(f"patch: unknown mode {mode!r}.")
    path = args.get("path")
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    if not isinstance(path, str) or not path:
        return tool_error("patch: mode='replace' requires 'path'.")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return tool_error("patch: mode='replace' requires old_string and new_string.")
    sensitive_error = _check_sensitive_path(path, task_id)
    if sensitive_error:
        return tool_error(sensitive_error)

    resolved = _native_file_path(path, task_id)
    if isinstance(resolved, str):
        return resolved
    cross_warning = file_state.check_stale(task_id, str(resolved))
    stale_warning = await _check_file_staleness(path, task_id)
    lock = await _async_file_lock(resolved)
    try:
        async with lock:
            async with aiofiles.open(resolved, "r", encoding="utf-8", errors="replace", newline="") as handle:
                content = await handle.read()
            occurrences = content.count(old_string)
            if occurrences == 0:
                return tool_error(
                    "old_string not found. Use read_file to verify the current content."
                )
            if occurrences > 1 and not args.get("replace_all", False):
                return tool_error(
                    "old_string appears more than once. Add unique context or set replace_all=true."
                )
            replacement_count = occurrences if args.get("replace_all", False) else 1
            updated = content.replace(old_string, new_string, replacement_count)
            await _write_native_file(resolved, updated)
    except FileNotFoundError:
        return tool_error(f"File not found: {path}")
    except OSError as exc:
        _record_patch_failure(task_id, str(resolved))
        return tool_error(f"Failed to patch {path}: {exc}")

    _reset_patch_failures(task_id, [str(resolved)])
    await _refresh_read_timestamp(path, task_id)
    file_state.note_write(task_id, str(resolved))
    result = {
        "success": True,
        "replacements": replacement_count,
        "resolved_path": str(resolved),
        "files_modified": [str(resolved)],
    }
    if cross_warning or stale_warning:
        result["_warning"] = cross_warning or stale_warning
    return json.dumps(
        result,
        ensure_ascii=False,
    )


async def _handle_search_files(args, **kw):
    """Search files using a native subprocess and preserve async cancellation."""
    task_id = kw.get("task_id") or "default"
    pattern = args.get("pattern", "")
    if not isinstance(pattern, str) or not pattern:
        return tool_error("search_files: missing required field 'pattern'.")
    target = {"grep": "content", "find": "files"}.get(
        args.get("target", "content"), args.get("target", "content")
    )
    limit = max(1, min(int(args.get("limit", 50)), 500))
    offset = max(0, int(args.get("offset", 0)))
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
    resolved = _native_file_path(args.get("path", "."), task_id)
    if isinstance(resolved, str):
        return resolved
    block_error = get_read_block_error(str(resolved))
    if block_error:
        return tool_error(block_error)

    if target == "files":
        command = ["rg", "--files", "--glob", pattern, str(resolved)]
    else:
        command = ["rg", "--line-number", "--no-heading", "--color", "never"]
        file_glob = args.get("file_glob")
        if isinstance(file_glob, str) and file_glob:
            command.extend(["--glob", file_glob])
        command.extend(["--regexp", pattern, str(resolved)])

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except FileNotFoundError:
        return tool_error("search_files requires ripgrep (rg), which is not installed.")
    except TimeoutError:
        return tool_error("search_files timed out after 30 seconds.")

    if process.returncode not in {0, 1}:
        return tool_error(stderr.decode(errors="replace") or "search_files failed")
    all_lines = stdout.decode(errors="replace").splitlines()
    all_lines, omitted = _filter_search_output_lines(all_lines, task_id)
    page = all_lines[offset:offset + limit]
    result = {
        "matches": page,
        "total_count": len(all_lines),
        "truncated": len(all_lines) > offset + len(page),
        "next_offset": offset + len(page) if len(all_lines) > offset + len(page) else None,
    }
    if omitted:
        result["omitted_sensitive_results"] = omitted
    if count >= 3:
        result["_warning"] = (
            f"You have run this exact search {count} times consecutively. "
            "The results have not changed. Use the information you already have."
        )
    return json.dumps(result, ensure_ascii=False)


async def read_file_tool(
    path: str,
    offset: int = 1,
    limit: int = 500,
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
