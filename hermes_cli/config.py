"""
Configuration management for Hermes Agent.

Config files are stored in ~/.hermes/ for easy access:
- ~/.hermes/config.yaml  - All settings (model, toolsets, terminal, etc.)
- ~/.hermes/.env         - API keys and secrets

This module provides:
- hermes config          - Show current configuration
- hermes config get      - Print a resolved configuration value
- hermes config set      - Set a specific value
- hermes config unset    - Remove a user configuration value
- hermes config wizard   - Re-run setup wizard
"""

import asyncio
import copy
import json
import logging
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

from hermes_cli.route_identity import normalize_route_base_url
from hermes_cli.secret_prompt import masked_secret_prompt

logger = logging.getLogger(__name__)

# Track which (config_path, mtime_ns, size) tuples we've already warned about
# so concurrent CLI/gateway loads of a broken config.yaml don't spam stderr
# every time. Cleared automatically when the file changes (different mtime).
_CONFIG_PARSE_WARNED: set = set()


def _backup_corrupt_config(config_path: Path) -> Optional[Path]:
    """Preserve a corrupted ``config.yaml`` by copying it to a timestamped ``.bak``.

    When the YAML can't be parsed, ``load_config()`` silently falls back to
    ``DEFAULT_CONFIG`` and the user's broken file stays on disk untouched.
    That file is still the user's only copy of their intended overrides — if
    they re-run the setup wizard or ``hermes config set`` (which rewrites
    ``config.yaml``), the broken-but-recoverable content is gone for good.

    This snapshots the corrupted file to ``config.yaml.corrupt.<ts>.bak`` so
    the user can diff/repair it. Unlike Gemini CLI's policy-file recovery
    (which resets the live file to a clean state), we deliberately leave
    ``config.yaml`` in place: hermes never silently mutates the user's config,
    and leaving it means a hand-fixed file is re-read on the next load. The
    backup is best-effort — any failure (permissions, symlink, disk full) is
    swallowed so config loading is never blocked by backup problems.

    Returns the backup path on success, else ``None``. Symlinks are not
    followed/copied (mirrors the Gemini #21541 lstat guard) to avoid
    clobbering whatever a malicious/misconfigured symlink points at.
    """
    try:
        if config_path.is_symlink():
            return None
        st = config_path.stat()
        if st.st_size == 0:
            # Empty file isn't worth preserving and yaml.safe_load returns {}
            # for it anyway (so it wouldn't reach here), but guard regardless.
            return None
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_path = config_path.with_name(f"{config_path.name}.corrupt.{ts}.bak")
        # Don't clobber an existing backup from the same second; if there's
        # already a corrupt backup for this exact mtime, assume we've snapshotted
        # this corruption already and skip (the dedup cache normally prevents a
        # second call, but a process restart can clear it).
        sibling_baks = list(
            config_path.parent.glob(f"{config_path.name}.corrupt.*.bak")
        )
        for existing in sibling_baks:
            try:
                if existing.stat().st_size == st.st_size:
                    # Same size as the current broken file — likely the same
                    # corruption already preserved. Avoid backup churn.
                    return None
            except OSError:
                continue
        if backup_path.exists():
            return None
        shutil.copy2(config_path, backup_path)
        return backup_path
    except Exception:
        return None


def _warn_config_parse_failure(
    config_path: Path, exc: Exception, *, fallback: str = "defaults"
) -> None:
    """Surface a config.yaml parse failure to user, log, and stderr.

    A YAML parse error in ``~/.hermes/config.yaml`` causes ``load_config()``
    to silently fall back to ``DEFAULT_CONFIG``, which means every user
    override (auxiliary providers, fallback chain, model overrides, etc.)
    is dropped. Before this helper that was a one-line ``print(...)`` that
    scrolled off-screen on the first invocation and was never seen again.

    Now: warn once per (path, mtime_ns, size) on stderr **and** in
    ``agent.log`` / ``errors.log`` at WARNING level so ``hermes logs``
    surfaces it. Re-warns automatically if the file changes (different
    mtime/size), so users editing the config see the next failure. On the
    first warning for a given broken file we also snapshot it to a
    timestamped ``.bak`` (best-effort) so the user's recoverable content
    survives any later rewrite of ``config.yaml`` by the setup wizard or
    ``hermes config set``.

    ``fallback`` selects the message wording: ``"defaults"`` (fresh process,
    nothing else to serve) or ``"last-known-good"`` (in-process retention of
    the previously loaded config — see the codex#31188 port in
    ``_load_config_impl``).
    """
    try:
        st = config_path.stat()
        key = (str(config_path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(config_path), 0, 0)
    if key in _CONFIG_PARSE_WARNED:
        return
    _CONFIG_PARSE_WARNED.add(key)

    backup_path = _backup_corrupt_config(config_path)

    if fallback == "last-known-good":
        msg = (
            f"Failed to parse {config_path}: {exc}. "
            f"Keeping the previously loaded config for this process — "
            f"edits to config.yaml are being IGNORED until the YAML is fixed."
        )
    else:
        msg = (
            f"Failed to parse {config_path}: {exc}. "
            f"Falling back to default config — every user override "
            f"(auxiliary providers, fallback chain, model settings) is being IGNORED. "
            f"Fix the YAML and restart."
        )
    if backup_path is not None:
        msg += f" A copy of the corrupted file was saved to {backup_path}."
    logger.warning(msg)
    try:
        sys.stderr.write(f"⚠️  hermes config: {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


async def _warn_config_parse_failure_from_event_loop(
    config_path: Path, exc: Exception, *, fallback: str = "defaults"
) -> None:
    """Native-async form of the upstream warning and backup path."""
    import aiofiles
    import aiofiles.os

    try:
        stat_result = await aiofiles.os.stat(config_path)
        key = (str(config_path), stat_result.st_mtime_ns, stat_result.st_size)
    except OSError:
        stat_result = None
        key = (str(config_path), 0, 0)
    if key in _CONFIG_PARSE_WARNED:
        return
    _CONFIG_PARSE_WARNED.add(key)

    backup_path: Optional[Path] = None
    candidate: Optional[Path] = None
    candidate_created = False
    backup_complete = False
    try:
        if not await aiofiles.os.path.islink(config_path) and (
            stat_result is not None and stat_result.st_size > 0
        ):
            already_backed_up = False
            prefix = f"{config_path.name}.corrupt."
            for sibling_name in await aiofiles.os.listdir(config_path.parent):
                if not sibling_name.startswith(prefix) or not sibling_name.endswith(".bak"):
                    continue
                try:
                    sibling_stat = await aiofiles.os.stat(
                        config_path.parent / sibling_name
                    )
                except OSError:
                    continue
                if sibling_stat.st_size == stat_result.st_size:
                    already_backed_up = True
                    break
            if not already_backed_up:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                candidate = config_path.with_name(
                    f"{config_path.name}.corrupt.{timestamp}.bak"
                )
                async with aiofiles.open(config_path, "rb") as source:
                    payload = await source.read()
                try:
                    async with aiofiles.open(candidate, "xb") as destination:
                        candidate_created = True
                        await destination.write(payload)
                except FileExistsError:
                    candidate = None
                if candidate_created:
                    try:
                        set_mode = aiofiles.os.wrap(os.chmod)
                        await set_mode(candidate, stat.S_IMODE(stat_result.st_mode))
                        set_times = aiofiles.os.wrap(os.utime)
                        await set_times(
                            candidate,
                            ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns),
                        )
                    except OSError:
                        pass
                    backup_path = candidate
                    backup_complete = True
    except asyncio.CancelledError:
        if candidate_created and not backup_complete and candidate is not None:
            try:
                await aiofiles.os.remove(candidate)
            except OSError:
                pass
        raise
    except Exception:
        if candidate_created and not backup_complete and candidate is not None:
            try:
                await aiofiles.os.remove(candidate)
            except OSError:
                pass
        backup_path = None

    if fallback == "last-known-good":
        message = (
            f"Failed to parse {config_path}: {exc}. "
            f"Failed to parse {config_path}: {exc}. "
            "Keeping the previously loaded config for this process — "
            "edits to config.yaml are being IGNORED until the YAML is fixed."
        )
    else:
        message = (
            f"Failed to parse {config_path}: {exc}. "
            "Falling back to default config — every user override "
            "(auxiliary providers, fallback chain, model settings) is being "
            "IGNORED. Fix the YAML and restart."
        )
    if backup_path is not None:
        message += f" A copy of the corrupted file was saved to {backup_path}."
    logger.warning(message)
    try:
        sys.stderr.write(f"⚠️  hermes config: {message}\n")
        sys.stderr.flush()
    except Exception:
        pass

_IS_WINDOWS = platform.system() == "Windows"
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Env var names that influence how the next subprocess executes —
# never writable through ``save_env_value``. Anything that controls
# the loader, interpreter, shell, or replacement editor counts:
#
# * ``LD_PRELOAD`` / ``LD_LIBRARY_PATH`` / ``LD_AUDIT`` — Linux dynamic
#   loader. ``DYLD_*`` — macOS equivalent. Planting a path here means
#   the next ``subprocess.run([...])`` Hermes makes loads attacker code
#   before main().
# * ``PYTHONPATH`` / ``PYTHONHOME`` / ``PYTHONSTARTUP`` /
#   ``PYTHONUSERBASE`` — Python interpreter init. Hermes itself starts
#   from one of these on every restart.
# * ``NODE_OPTIONS`` / ``NODE_PATH`` — Node interpreter; affects npm,
#   ``hermes update``, the TUI build.
# * ``PATH`` — too broad to allow. The dashboard never needs to rewrite
#   the operator's PATH; if a tool can't be found, the fix is to add an
#   absolute path in the integration config, not to mutate PATH globally.
# * ``GIT_SSH_COMMAND`` / ``GIT_EXEC_PATH`` — git rewrites that fire
#   on every plugin install / ``hermes update``.
# * ``BROWSER`` / ``EDITOR`` / ``VISUAL`` / ``PAGER`` — commands the
#   shell or CLI invokes implicitly. Wrong values here = RCE on next
#   ``$EDITOR``.
# * ``SHELL`` — what subprocess uses with ``shell=True`` (we try to
#   avoid that, but defense in depth).
# * ``HERMES_HOME`` / ``HERMES_PROFILE`` / ``HERMES_CONFIG`` /
#   ``HERMES_ENV`` — Hermes runtime location flags. Writing these into
#   ``.env`` would relocate state in ways the user did not request from
#   the dashboard. ``config.yaml`` is the supported surface for these.
#
# IMPORTANT: ``HERMES_*`` overall is NOT blocked. Many legitimate provider
# credentials and endpoint overrides follow that prefix. The
# denylist is name-by-name on purpose so the gate stays narrow and
# doesn't accidentally break provider setup wizards.
#
# This is enforced on *write* only — values already in ``.env`` (set
# by the operator out-of-band, or pre-existing) keep working. The
# point is that the dashboard's writable surface cannot escalate by
# planting them.
_ENV_VAR_NAME_DENYLIST: frozenset[str] = frozenset({
    # Loader / linker
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_FALLBACK_FRAMEWORK_PATH",
    # Python
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
    "PYTHONEXECUTABLE", "PYTHONNOUSERSITE",
    # Node
    "NODE_OPTIONS", "NODE_PATH",
    # General
    "PATH", "SHELL", "BROWSER", "EDITOR", "VISUAL", "PAGER",
    # Git
    "GIT_SSH_COMMAND", "GIT_EXEC_PATH", "GIT_SHELL",
    # Hermes runtime location — never via dashboard env writer.
    # NOT a HERMES_* blanket: provider credentials and endpoint overrides are
    # allowed.
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV",
})


def _reject_denylisted_env_var(key: str) -> None:
    """Raise if ``key`` is in :data:`_ENV_VAR_NAME_DENYLIST`.

    Centralised so both the regular and "secure" env writers share the
    same gate, and so the message is consistent for callers.
    """
    if key in _ENV_VAR_NAME_DENYLIST:
        raise ValueError(
            f"Environment variable {key!r} is on the writer denylist. "
            "Names that influence subprocess execution (LD_PRELOAD, "
            "PYTHONPATH, PATH, EDITOR, ...) or Hermes runtime location "
            "(HERMES_HOME, HERMES_PROFILE, ...) cannot be persisted via "
            "the env writer. If you really need this, edit "
            "~/.hermes/.env directly."
        )

_LAST_EXPANDED_CONFIG_BY_PATH: Dict[str, Any] = {}
# (path, mtime_ns, size) -> cached expanded config dict.
# load_config() returns a deepcopy of the cached value when the file
# hasn't changed since the last load, skipping yaml.safe_load +
# _deep_merge + _normalize_* + _expand_env_vars (~13 ms/call).
# save_config() + migrate_config() write via atomic_yaml_write which
# produces a fresh inode, so stat() sees a new mtime_ns and the next
# load repopulates automatically — no explicit invalidation hook.
# Cached tuple is (user_mtime_ns, user_size, managed_mtime_ns, managed_size,
# merged_value, env_ref_snapshot) — the managed-file signature is folded in so
# editing the managed-scope config.yaml invalidates the cache (see
# managed_scope), and the env snapshot invalidates it when a referenced ${VAR}
# changes value (late .env load, in-process rotation — #58514).
_LOAD_CONFIG_CACHE: Dict[str, Tuple[int, int, int, int, Dict[str, Any], Dict[str, Optional[str]]]] = {}
# (path, mtime_ns, size) -> cached raw yaml dict. Same pattern as
# _LOAD_CONFIG_CACHE but for read_raw_config() — used when callers want
# the user's on-disk values without defaults merged in.
_RAW_CONFIG_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
# Serializes all config read/write paths. libyaml's C extension is not
# thread-safe for concurrent safe_load() on the same file, and multiple
# tool threads (approval.py, browser_tool.py, setup flows) hit
# load_config / read_raw_config / save_config from different threads
# during long agent runs. RLock (not Lock) because save_config internally
# calls read_raw_config. Also covers mutation of the module-level cache
# dicts above.
_CONFIG_LOCK = threading.RLock()
_ASYNC_CONFIG_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.ReferenceType[asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _get_async_config_lock() -> asyncio.Lock:
    """Return the config lock owned by the active event loop."""
    loop = asyncio.get_running_loop()
    lock_ref = _ASYNC_CONFIG_LOCKS.get(loop)
    lock = lock_ref() if lock_ref is not None else None
    if lock is None:
        lock = asyncio.Lock()
        _ASYNC_CONFIG_LOCKS[loop] = weakref.ref(lock)
    return lock


# Env var names written to .env that aren't in OPTIONAL_ENV_VARS
# (managed by setup/provider flows directly).
_EXTRA_ENV_KEYS = frozenset({
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
})
import yaml

from hermes_cli.colors import Colors, color
from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul


# =============================================================================
# Managed mode (NixOS declarative config)
# =============================================================================

_MANAGED_TRUE_VALUES = ("true", "1", "yes")
_MANAGED_SYSTEM_NAMES = {
    "nix": "NixOS",
    "nixos": "NixOS",
}
# Values that used to signal a Homebrew-managed install. Homebrew is no
# longer a supported distribution method, so these are explicitly ignored
# rather than treated as a managed system — they fall through to git/unknown
# detection instead of blocking config writes.
_IGNORED_MANAGED_VALUES = frozenset({"brew", "homebrew"})


def get_managed_system() -> Optional[str]:
    """Return the package manager owning this install, if any."""
    raw = os.getenv("HERMES_MANAGED", "").strip()
    if raw:
        normalized = raw.lower()
        if normalized in _IGNORED_MANAGED_VALUES:
            return None
        if normalized in _MANAGED_TRUE_VALUES:
            return "NixOS"
        return _MANAGED_SYSTEM_NAMES.get(normalized, raw)

    managed_marker = get_hermes_home() / ".managed"
    if managed_marker.exists():
        return "NixOS"
    return None


def is_managed() -> bool:
    """Check if Hermes is running in package-manager-managed mode.

    Two signals: the HERMES_MANAGED env var (set by the systemd service),
    or a .managed marker file in HERMES_HOME (set by the NixOS activation
    script, so interactive shells also see it).
    """
    return get_managed_system() is not None


def format_managed_message(action: str = "modify this Hermes installation") -> str:
    """Build a user-facing error for managed installs."""
    managed_system = get_managed_system() or "a package manager"
    raw = os.getenv("HERMES_MANAGED", "").strip().lower()

    if managed_system == "NixOS":
        env_hint = "true" if raw in _MANAGED_TRUE_VALUES else raw or "true"
        return (
            f"Cannot {action}: this Hermes installation is managed by NixOS "
            f"(HERMES_MANAGED={env_hint}).\n"
            "Edit services.hermes-agent.settings in your configuration.nix and run:\n"
            "  sudo nixos-rebuild switch"
        )

    return (
        f"Cannot {action}: this Hermes installation is managed by {managed_system}.\n"
        "Use your package manager to upgrade or reinstall Hermes."
    )

def managed_error(action: str = "modify configuration"):
    """Print user-friendly error for managed mode."""
    print(format_managed_message(action), file=sys.stderr)


# =============================================================================
# Config paths
# =============================================================================

# Re-export from hermes_constants — canonical definition lives there.
from hermes_constants import get_hermes_home, get_process_hermes_home  # noqa: F811,E402
from utils import atomic_replace, fast_safe_load

def get_config_path() -> Path:
    """Get the main config file path."""
    return get_hermes_home() / "config.yaml"

def get_env_path() -> Path:
    """Get the .env file path (for API keys)."""
    return get_hermes_home() / ".env"

def _resolve_hermes_uid_gid() -> tuple[Optional[int], Optional[int]]:
    """Read the HERMES_UID / HERMES_GID env vars set by Docker deployments.

    Docker containers running Hermes commonly set these to map the in-container
    user to a host user so volume-mounted state files end up with the right
    ownership. The entrypoint chowns the top-level HERMES_HOME once, but
    subdirectories created at runtime by ``ensure_hermes_home()`` (especially
    for profile namespaces under ``profiles/<name>/``) need the same chown
    or they land as ``root:root`` and block subsequent uid-mapped workers
    with ``PermissionError [Errno 13]``. See #34107.

    Returns ``(uid, gid)`` parsed from the env vars, or ``(None, None)``
    when either is missing/invalid. Returns ``(None, None)`` on Windows
    too (where chown is a no-op anyway).
    """
    if sys.platform == "win32":
        return None, None
    uid_str = os.environ.get("HERMES_UID", "").strip()
    gid_str = os.environ.get("HERMES_GID", "").strip()
    try:
        uid = int(uid_str) if uid_str else None
    except ValueError:
        uid = None
    try:
        gid = int(gid_str) if gid_str else None
    except ValueError:
        gid = None
    return uid, gid


def _chown_to_hermes_uid(path) -> None:
    """Chown ``path`` to ``HERMES_UID:HERMES_GID`` if those env vars are set.

    No-op when:
      - Either env var is unset/invalid
      - The current process isn't root (chown will EPERM — silently ignored)
      - On Windows (chown semantics don't apply)

    Used by :func:`_secure_dir` to keep ownership consistent across all
    directories created by :func:`ensure_hermes_home` on Docker deployments.
    See #34107.
    """
    uid, gid = _resolve_hermes_uid_gid()
    if uid is None and gid is None:
        return
    try:
        # os.chown with -1 means "don't change" for that field.
        os.chown(
            path,
            uid if uid is not None else -1,
            gid if gid is not None else -1,
        )
    except (OSError, AttributeError, NotImplementedError):
        # OSError covers EPERM (not running as root) and ENOENT (race),
        # both of which are non-fatal — the dir is still created and
        # the entrypoint's startup chown -R will fix it on next restart.
        pass


def _secure_dir(path):
    """Set directory to owner-only access (0700 by default). No-op on Windows.

    Skipped in managed mode — the NixOS module sets group-readable
    permissions (0750) so interactive users in the hermes group can
    share state with the gateway service.

    The mode can be overridden via the HERMES_HOME_MODE environment variable
    (e.g. HERMES_HOME_MODE=0701) for deployments where a web server (nginx,
    caddy, etc.) needs to traverse HERMES_HOME to reach a served subdirectory.
    The execute-only bit on a directory permits cd-through without exposing
    directory listings.

    Also applies ``HERMES_UID``/``HERMES_GID``-based ownership when those env
    vars are set (#34107 — Docker deployments need this so profile subdirs
    created at runtime by kanban workers don't land as root:root and block
    subsequent uid-mapped workers).
    """
    if is_managed():
        return
    try:
        mode_str = os.environ.get("HERMES_HOME_MODE", "").strip()
        mode = int(mode_str, 8) if mode_str else 0o700
    except ValueError:
        mode = 0o700
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass
    _chown_to_hermes_uid(path)


def _is_container() -> bool:
    """Detect if we're running inside a Docker/Podman/LXC container.

    When Hermes runs in a container with volume-mounted config files, forcing
    0o600 permissions breaks multi-process setups where the gateway and
    dashboard run as different UIDs or the volume mount requires broader
    permissions.
    """
    # Explicit opt-out
    if os.environ.get("HERMES_CONTAINER") or os.environ.get("HERMES_SKIP_CHMOD"):
        return True
    # Docker / Podman marker file
    if os.path.exists("/.dockerenv"):
        return True
    # LXC / cgroup-based detection
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            cgroup_content = f.read()
        if "docker" in cgroup_content or "lxc" in cgroup_content or "kubepods" in cgroup_content:
            return True
    except (OSError, IOError):
        pass
    return False


def _secure_file(path):
    """Set file to owner-only read/write (0600). No-op on Windows.

    Skipped in managed mode — the NixOS activation script sets
    group-readable permissions (0640) on config files.

    Skipped in containers — Docker/Podman volume mounts often need broader
    permissions.  Set HERMES_SKIP_CHMOD=1 to force-skip on other systems.
    """
    if is_managed() or _is_container():
        return
    try:
        if os.path.exists(str(path)):
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _ensure_default_soul_md(home: Path) -> None:
    """Seed a default SOUL.md into HERMES_HOME, upgrading legacy empty templates.

    First run: write DEFAULT_SOUL_MD. Existing installs whose SOUL.md is still
    the old comment-only scaffold (seeded by older install.sh / install.ps1 /
    docker images, which shadowed the runtime default) get upgraded in place to
    DEFAULT_SOUL_MD. A SOUL.md the user actually customized is never touched.
    """
    soul_path = home / "SOUL.md"
    if soul_path.exists():
        try:
            existing = soul_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if not is_legacy_template_soul(existing):
            return
        # Legacy empty template -> upgrade to the real default in place.
    soul_path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    _secure_file(soul_path)


# Home paths whose directory skeleton has been created this process — see
# ensure_hermes_home(). Only successful passes are recorded, so a raised
# managed-mode/missing-profile error keeps re-checking on later loads.
_HERMES_HOME_ENSURED: set = set()


def ensure_hermes_home():
    """Ensure ~/.hermes directory structure exists with secure permissions.

    In managed mode (NixOS), dirs are created by the activation script with
    setgid + group-writable (2770). We skip mkdir and set umask(0o007) so
    any files created (e.g. SOUL.md) are group-writable (0660).

    Memoized per home path: this runs on EVERY ``load_config()`` (inside the
    config lock), and the ~14 mkdir/chmod syscalls per call made repeated
    config loads the dominant cost of hot read paths like ``model.options``.
    After the first successful pass for a given ``HERMES_HOME`` we only re-run
    the full walk if the home directory itself has vanished (a deleted home is
    recreated on the next load, as before). Profile switches change
    ``get_hermes_home()`` and therefore re-run for the new path.
    """
    home = get_hermes_home()
    key = str(home)

    if key in _HERMES_HOME_ENSURED and home.is_dir():
        return
    # Named profiles must be created explicitly (e.g. ``hermes profile create``).
    # If a stale process keeps running after the profile was renamed/deleted,
    # silently mkdir-ing the old HERMES_HOME would resurrect an empty skeleton
    # and make the deleted profile reappear in Desktop/profile lists.
    if home.parent.name == "profiles" and not home.exists():
        raise FileNotFoundError(
            f"Named profile home does not exist: {home}. "
            "Create the profile explicitly before using it."
        )
    if is_managed():
        old_umask = os.umask(0o007)
        try:
            _ensure_hermes_home_managed(home)
        finally:
            os.umask(old_umask)
    else:
        home.mkdir(parents=True, exist_ok=True)
        _secure_dir(home)
        for subdir in (
            "sessions", "logs", "memories", "hooks", "image_cache", "skills",
        ):
            d = home / subdir
            d.mkdir(parents=True, exist_ok=True)
            _secure_dir(d)
        _ensure_default_soul_md(home)

    _HERMES_HOME_ENSURED.add(key)


def _ensure_hermes_home_managed(home: Path):
    """Managed-mode variant: verify dirs exist (activation creates them), seed SOUL.md."""
    if not home.is_dir():
        raise RuntimeError(
            f"HERMES_HOME {home} does not exist. "
            "Run 'sudo nixos-rebuild switch' first."
        )
    for subdir in ("sessions", "logs", "memories"):
        d = home / subdir
        if not d.is_dir():
            raise RuntimeError(
                f"{d} does not exist. "
                "Run 'sudo nixos-rebuild switch' first."
            )
    # Inside umask(0o007) scope — SOUL.md will be created as 0660
    _ensure_default_soul_md(home)


# =============================================================================
# Config loading/saving
# =============================================================================

from hermes_cli.config_defaults import DEFAULT_CONFIG, OPTIONAL_ENV_VARS  # noqa: F401

# =============================================================================
# Config Migration System
# =============================================================================

# Track which env vars were introduced in each config version.
# Migration only mentions vars new since the user's previous version.
ENV_VARS_BY_VERSION: Dict[int, List[str]] = {
    3: ["FIRECRAWL_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "FAL_KEY"],
    10: ["TAVILY_API_KEY"],
}

# Required environment variables with metadata for migration prompts.
# LLM provider is required but handled in the setup wizard's provider
# selection step (Nous Portal / OpenRouter / Custom endpoint), so this
# dict is intentionally empty — no single env var is universally required.
REQUIRED_ENV_VARS = {}

def get_missing_env_vars(required_only: bool = False) -> List[Dict[str, Any]]:
    """
    Check which environment variables are missing.
    
    Returns list of dicts with var info for missing variables.
    """
    missing = []
    
    # Check required vars
    for var_name, info in REQUIRED_ENV_VARS.items():
        if not get_env_value(var_name):
            missing.append({"name": var_name, **info, "is_required": True})
    
    # Check optional vars (if not required_only)
    if not required_only:
        for var_name, info in OPTIONAL_ENV_VARS.items():
            if not get_env_value(var_name):
                missing.append({"name": var_name, **info, "is_required": False})
    
    return missing


def _set_nested(config, dotted_key: str, value):
    """Set a value at an arbitrarily nested dotted key path.

    Supports both dict and list navigation:
      _set_nested(c, "a.b.c", 1)     → c["a"]["b"]["c"] = 1
      _set_nested(c, "a.0.b", 1)     → c["a"][0]["b"] = 1
      _set_nested(c, "providers.1", "x") → c["providers"][1] = "x"

    Intermediate dicts are created on demand.  List indices are parsed
    from numeric path segments; the referenced index must already exist
    (we do not grow lists — the user is navigating into structure they
    wrote themselves).  If a segment targets a non-container leaf
    (scalar), the leaf is replaced with a fresh dict so the write can
    proceed — this preserves the pre-existing behavior for bare scalar
    overrides (e.g. setting ``a.b.c`` where ``a.b`` was previously a
    string).

    Guards against #17876: before this fix the code unconditionally
    replaced any non-dict value (including lists) with ``{}``, silently
    destroying list-typed config like ``custom_providers`` whenever a
    caller used an indexed path.
    """
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if isinstance(current, list):
            try:
                idx = int(part)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Cannot navigate into list at key {dotted_key!r}: "
                    f"segment {part!r} is not a numeric index"
                )
            current = current[idx]
        elif isinstance(current, dict):
            existing = current.get(part)
            # Preserve dicts and lists; replace missing/scalar with a fresh dict.
            if part not in current or not isinstance(existing, (dict, list)):
                current[part] = {}
            current = current[part]
        else:
            raise TypeError(
                f"Cannot navigate into {type(current).__name__} at key {dotted_key!r}"
            )
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def clear_model_endpoint_credentials(
    model_cfg: Dict[str, Any],
    *,
    clear_api_key: bool = True,
    clear_api_mode: bool = True,
    clear_base_url: bool = False,
) -> Dict[str, Any]:
    """Remove stale inline endpoint credentials from a model config.

    ``model.api_key`` is valid only for explicit custom endpoint assignments.
    Built-in providers resolve credentials from env vars, auth.json, or the
    credential pool. When switching away from a custom endpoint, leaving these
    fields behind keeps secrets in config.yaml and can contaminate later custom
    resolution paths.
    """
    if not isinstance(model_cfg, dict):
        return model_cfg
    if clear_api_key:
        model_cfg.pop("api_key", None)
        model_cfg.pop("api", None)
    if clear_api_mode:
        model_cfg.pop("api_mode", None)
    if clear_base_url:
        model_cfg.pop("base_url", None)
    return model_cfg


_MISSING = object()


def _get_nested(config, dotted_key: str):
    """Return a dotted-path value from nested dict/list config data."""
    current = config
    for part in dotted_key.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (TypeError, ValueError, IndexError):
                return _MISSING
        elif isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        else:
            return _MISSING
    return current


def _unset_nested(config, dotted_key: str) -> bool:
    """Remove a dotted-path value from nested dict/list config data."""
    parts = dotted_key.split(".")
    if not parts:
        return False

    parents = []
    current = config
    for part in parts[:-1]:
        parents.append((current, part))
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (TypeError, ValueError, IndexError):
                return False
        elif isinstance(current, dict):
            if part not in current:
                return False
            current = current[part]
        else:
            return False

    last = parts[-1]
    removed = False
    if isinstance(current, list):
        try:
            current.pop(int(last))
            removed = True
        except (TypeError, ValueError, IndexError):
            return False
    elif isinstance(current, dict):
        if last not in current:
            return False
        del current[last]
        removed = True
    else:
        return False

    # Drop empty dict containers left behind by the deletion while preserving
    # user-authored empty lists and non-empty sibling branches.
    for parent, part in reversed(parents):
        if current != {}:
            break
        if isinstance(parent, list):
            try:
                idx = int(part)
            except (TypeError, ValueError):
                break
            if 0 <= idx < len(parent) and parent[idx] == {}:
                parent.pop(idx)
                current = parent
                continue
        elif isinstance(parent, dict) and parent.get(part) == {}:
            del parent[part]
            current = parent
            continue
        break

    return removed


def _is_env_config_key(key: str) -> bool:
    """Return whether `hermes config set` routes this key to .env."""
    if "." in key:
        return False
    key_upper = key.upper()
    api_keys = [
        'OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
        'EXA_API_KEY', 'PARALLEL_API_KEY', 'FIRECRAWL_API_KEY', 'FIRECRAWL_API_URL',
        'TAVILY_API_KEY',
        'BROWSERBASE_API_KEY', 'BROWSERBASE_PROJECT_ID', 'BROWSER_USE_API_KEY',
        'FAL_KEY', 'SUDO_PASSWORD',
        'GITHUB_TOKEN',
    ]
    return (
        key_upper in api_keys
        or key_upper.endswith(('_API_KEY', '_TOKEN'))
    )


def _format_config_get_value(value, *, as_json: bool) -> str:
    """Format a config value for command-line output."""
    if as_json:
        import json
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, sort_keys=False).rstrip()
    return str(value)


def get_missing_config_fields() -> List[Dict[str, Any]]:
    """
    Check which config fields are missing or outdated (recursive).
    
    Walks the DEFAULT_CONFIG tree at arbitrary depth and reports any keys
    present in defaults but absent from the user's loaded config.
    """
    config = load_config()
    missing = []

    def _check(defaults: dict, current: dict, prefix: str = ""):
        for key, default_value in defaults.items():
            if key.startswith('_'):
                continue
            full_key = key if not prefix else f"{prefix}.{key}"
            if key not in current:
                missing.append({
                    "key": full_key,
                    "default": default_value,
                    "description": f"New config option: {full_key}",
                })
            elif isinstance(default_value, dict) and isinstance(current.get(key), dict):
                _check(default_value, current[key], full_key)

    _check(DEFAULT_CONFIG, config)
    return missing


async def get_missing_skill_config_vars() -> List[Dict[str, Any]]:
    """Return skill-declared config vars that are missing or empty in config.yaml.

    Scans all enabled skills for ``metadata.hermes.config`` entries, then checks
    which ones are absent or empty under ``skills.config.<key>`` in the user's
    config.yaml.  Returns a list of dicts suitable for prompting.
    """
    try:
        from agent.skill_utils import discover_all_skill_config_vars, SKILL_CONFIG_PREFIX
    except Exception:
        return []

    try:
        all_vars = await discover_all_skill_config_vars()
    except Exception as e:
        # A malformed SKILL.md, unreadable external skill dir, or similar
        # should never break `hermes update`.  Skill-config prompting is a
        # post-migration nicety, not a blocker.
        import logging
        logging.getLogger(__name__).debug(
            "discover_all_skill_config_vars failed: %s", e
        )
        return []
    if not all_vars:
        return []

    config = await load_config_readonly()
    missing: List[Dict[str, Any]] = []
    for var in all_vars:
        # Skill config is stored under skills.config.<logical_key>
        storage_key = f"{SKILL_CONFIG_PREFIX}.{var['key']}"
        parts = storage_key.split(".")
        current = config
        value = None
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
                value = current
            else:
                value = None
                break
        # Missing = key doesn't exist or is empty string
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(var)
    return missing


# ``_normalize_custom_provider_entry`` runs on every ``load_picker_context()``
# call (i.e. per interactive picker/inventory request), so any warning it emits
# fires repeatedly for the same static config. Deduplicate per (provider,
# signature): on Windows a repeated-warning storm contends on
# ``concurrent-log-handler``'s cross-process rotation lock and can peg a core /
# stall the gateway/serve event loop. The cache lives for the process lifetime.
_PROVIDER_NORMALIZE_WARNED: set = set()


def _warn_once_per_provider(
    provider_key: str, signature: str, msg: str, *args: Any
) -> None:
    """Emit ``logger.warning(msg, *args)`` at most once per (provider, signature)."""
    dedup_key = (provider_key or "?", signature)
    if dedup_key in _PROVIDER_NORMALIZE_WARNED:
        return
    _PROVIDER_NORMALIZE_WARNED.add(dedup_key)
    logger.warning(msg, *args)


def _normalize_custom_provider_entry(
    entry: Any,
    *,
    provider_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Return a runtime-compatible custom provider entry or ``None``."""
    if not isinstance(entry, dict):
        return None

    # Shallow-copy before the alias normalization below writes into the
    # entry: callers (get_compatible_custom_providers,
    # providers_dict_to_custom_providers) pass live sub-dicts from
    # load_config_readonly()'s shared cache, and mutating those both
    # violates the cache's no-mutation contract and leaks duplicated
    # alias keys back into config.yaml through any later
    # save_config(load_config()) round-trip.
    entry = dict(entry)

    # Accept camelCase aliases commonly used in hand-written configs.
    _CAMEL_ALIASES: Dict[str, str] = {
        "apiKey": "api_key",
        "baseUrl": "base_url",
        "apiMode": "api_mode",
        "keyEnv": "key_env",
        "apiKeyEnv": "key_env",  # alias — OpenClaw-compatible + docs variant
        "defaultModel": "default_model",
        "contextLength": "context_length",
        "rateLimitDelay": "rate_limit_delay",
    }
    # api_key_env is a documented snake_case alias for key_env (see
    # website/docs/guides/azure-foundry.md).  Normalize it up front so the
    # rest of the normalizer treats it as the canonical field.
    if "api_key_env" in entry and "key_env" not in entry:
        entry["key_env"] = entry["api_key_env"]
    _KNOWN_KEYS = {
        # ``provider`` duplicates the ``providers.<name>`` mapping key and is
        # unused here, but Hermes' own config writer has historically emitted it
        # into provider entries. Accept it silently so those (self-written)
        # configs don't warn on every load.
        "provider",
        "name", "api", "url", "base_url", "api_key", "key_env", "api_key_env",
        "api_mode", "transport", "model", "default_model", "models",
        "context_length", "rate_limit_delay",
        "request_timeout_seconds", "stale_timeout_seconds",
        "discover_models", "extra_body", "extra_headers",
        "ssl_ca_cert", "ssl_verify",
    }
    for camel, snake in _CAMEL_ALIASES.items():
        if camel in entry and snake not in entry:
            _warn_once_per_provider(
                provider_key, f"camel:{camel}",
                "providers.%s: camelCase key '%s' auto-mapped to '%s' "
                "(use snake_case to avoid this warning)",
                provider_key or "?", camel, snake,
            )
            entry[snake] = entry[camel]
    unknown = set(entry.keys()) - _KNOWN_KEYS - set(_CAMEL_ALIASES.keys())
    if unknown:
        _warn_once_per_provider(
            provider_key, "unknown:" + ",".join(sorted(unknown)),
            "providers.%s: unknown config keys ignored: %s",
            provider_key or "?", ", ".join(sorted(unknown)),
        )

    from urllib.parse import urlparse

    base_url = ""
    for url_key in ("base_url", "url", "api"):
        raw_url = entry.get(url_key)
        if isinstance(raw_url, str) and raw_url.strip():
            candidate = raw_url.strip()
            # Accept URLs containing unresolved placeholder tokens — both
            # ``${ENV_VAR}`` env-refs and bare ``{region}``-style templates —
            # without URL validation. They are expanded at runtime, so a
            # caller reaching this normalizer with raw (un-expanded) config
            # would otherwise see the provider silently dropped (#14457).
            if re.search(r"\{[^}]+\}", candidate):
                base_url = candidate
                break
            parsed = urlparse(candidate)
            if parsed.scheme and parsed.netloc:
                base_url = candidate
                break
            else:
                logger.warning(
                    "providers.%s: '%s' value '%s' is not a valid URL "
                    "(no scheme or host) — skipped",
                    provider_key or "?", url_key, candidate,
                )
    if not base_url:
        return None

    name = ""
    raw_name = entry.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        name = raw_name.strip()
    elif provider_key.strip():
        name = provider_key.strip()
    if not name:
        return None

    normalized: Dict[str, Any] = {
        "name": name,
        "base_url": base_url,
    }

    provider_key = provider_key.strip()
    if provider_key:
        normalized["provider_key"] = provider_key

    api_key = entry.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        normalized["api_key"] = api_key.strip()

    key_env = entry.get("key_env")
    if isinstance(key_env, str) and key_env.strip():
        normalized["key_env"] = key_env.strip()

    api_mode = entry.get("api_mode") or entry.get("transport")
    if isinstance(api_mode, str) and api_mode.strip():
        normalized["api_mode"] = api_mode.strip()

    model_name = entry.get("model") or entry.get("default_model")
    if isinstance(model_name, str) and model_name.strip():
        normalized["model"] = model_name.strip()

    models = entry.get("models")
    if isinstance(models, dict) and models:
        # Shallow-copy: `entry` may alias a cached config sub-dict, and the
        # normalized entry escapes into long-lived runtime state
        # (agent._custom_providers) — don't share the cached models mapping.
        normalized["models"] = dict(models)
    elif isinstance(models, list) and models:
        # Hand-edited configs (and older Hermes versions) may write
        # ``models`` as a plain list of ids or as ``[{id: ...}]`` rows.
        # Preserve both by converting to the dict shape downstream code
        # expects; otherwise normalize silently drops the list and /model
        # shows the provider with (0) models.
        normalized_models: Dict[str, Any] = {}
        for item in models:
            if isinstance(item, str) and item.strip():
                normalized_models[item.strip()] = {}
                continue
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                model_id = item.get("name")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_meta = {
                k: v for k, v in item.items() if k not in {"id", "name"}
            }
            normalized_models[model_id.strip()] = model_meta
        if normalized_models:
            normalized["models"] = normalized_models

    context_length = entry.get("context_length")
    if isinstance(context_length, int) and context_length > 0:
        normalized["context_length"] = context_length

    rate_limit_delay = entry.get("rate_limit_delay")
    if isinstance(rate_limit_delay, (int, float)) and rate_limit_delay >= 0:
        normalized["rate_limit_delay"] = rate_limit_delay

    discover_models = entry.get("discover_models")
    if isinstance(discover_models, bool):
        normalized["discover_models"] = discover_models

    extra_body = entry.get("extra_body")
    if isinstance(extra_body, dict):
        normalized["extra_body"] = dict(extra_body)

    # Per-provider extra HTTP headers (proxies, gateways, custom auth).
    # Values may carry credentials (e.g. CF-Access-Client-Secret) — never
    # log them anywhere downstream.
    normalized_headers = normalize_extra_headers(entry.get("extra_headers"))
    if normalized_headers:
        normalized["extra_headers"] = normalized_headers

    ssl_ca_cert = entry.get("ssl_ca_cert")
    if isinstance(ssl_ca_cert, str) and ssl_ca_cert.strip():
        normalized["ssl_ca_cert"] = ssl_ca_cert.strip()

    ssl_verify = entry.get("ssl_verify")
    if isinstance(ssl_verify, bool):
        normalized["ssl_verify"] = ssl_verify
    elif isinstance(ssl_verify, str) and ssl_verify.strip():
        normalized["ssl_verify"] = ssl_verify.strip()

    return normalized


def _custom_provider_entry_to_provider_config(
    entry: Any,
    *,
    provider_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Translate a legacy custom provider entry to the v12 providers shape."""
    normalized = _normalize_custom_provider_entry(
        dict(entry) if isinstance(entry, dict) else entry,
        provider_key=provider_key,
    )
    if normalized is None:
        return None

    provider_entry: Dict[str, Any] = {"api": normalized["base_url"]}

    for field in (
        "name",
        "api_key",
        "key_env",
        "models",
        "context_length",
        "rate_limit_delay",
        "discover_models",
        "extra_body",
        "extra_headers",
        "ssl_ca_cert",
        "ssl_verify",
    ):
        if field in normalized:
            provider_entry[field] = normalized[field]

    if "model" in normalized:
        provider_entry["default_model"] = normalized["model"]
    if "api_mode" in normalized:
        provider_entry["transport"] = normalized["api_mode"]

    return provider_entry


def providers_dict_to_custom_providers(providers_dict: Any) -> List[Dict[str, Any]]:
    """Normalize ``providers`` config entries into the legacy custom-provider shape."""
    if not isinstance(providers_dict, dict):
        return []

    custom_providers: List[Dict[str, Any]] = []
    for key, entry in providers_dict.items():
        if isinstance(entry, dict) and not is_provider_enabled(entry):
            continue
        normalized = _normalize_custom_provider_entry(entry, provider_key=str(key))
        if normalized is not None:
            custom_providers.append(normalized)

    return custom_providers


def get_compatible_custom_providers(
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return a deduplicated custom-provider view across legacy and v12+ config.

    ``custom_providers`` remains the on-disk legacy format, while ``providers``
    is the newer keyed schema.  Runtime and picker flows still need a single
    list-shaped view, but we should not materialise that compatibility layer
    back into config.yaml because it duplicates entries in UIs.
    """
    if config is None:
        config = load_config()

    compatible: List[Dict[str, Any]] = []
    seen_provider_keys: set = set()
    seen_name_url_pairs: set = set()

    def _append_if_new(entry: Optional[Dict[str, Any]]) -> None:
        if entry is None:
            return
        provider_key = str(entry.get("provider_key", "") or "").strip().lower()
        name = str(entry.get("name", "") or "").strip().lower()
        base_url = str(entry.get("base_url", "") or "").strip().rstrip("/").lower()
        model = str(entry.get("model", "") or "").strip().lower()
        pair = (name, base_url, model)

        if provider_key and provider_key in seen_provider_keys:
            return
        if name and base_url and pair in seen_name_url_pairs:
            return

        compatible.append(entry)
        if provider_key:
            seen_provider_keys.add(provider_key)
        if name and base_url:
            seen_name_url_pairs.add(pair)

    custom_providers = config.get("custom_providers")
    if custom_providers is not None:
        if not isinstance(custom_providers, list):
            return []
        for entry in custom_providers:
            _append_if_new(_normalize_custom_provider_entry(entry))

    for entry in providers_dict_to_custom_providers(config.get("providers")):
        _append_if_new(entry)

    return compatible


def _coerce_ssl_verify(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return None


def get_custom_provider_tls_settings(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return TLS settings from a matching ``custom_providers`` / ``providers`` entry."""
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            custom_providers = []
    if not base_url or not isinstance(custom_providers, list):
        return {}

    target_url = normalize_route_base_url(base_url)
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        entry_url = normalize_route_base_url(entry.get("base_url"))
        if not entry_url or entry_url != target_url:
            continue
        out: Dict[str, Any] = {}
        ca = entry.get("ssl_ca_cert")
        if isinstance(ca, str) and ca.strip():
            out["ssl_ca_cert"] = ca.strip()
        verify = _coerce_ssl_verify(entry.get("ssl_verify"))
        if verify is not None:
            out["ssl_verify"] = verify
        return out
    return {}


def apply_custom_provider_tls_to_client_kwargs(
    client_kwargs: Dict[str, Any],
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Attach per-provider TLS knobs to OpenAI client kwargs when matched."""
    tls = get_custom_provider_tls_settings(base_url, custom_providers, config)
    if tls.get("ssl_ca_cert"):
        client_kwargs["ssl_ca_cert"] = tls["ssl_ca_cert"]
    if "ssl_verify" in tls:
        client_kwargs["ssl_verify"] = tls["ssl_verify"]


def normalize_extra_headers(extra_headers: Any) -> Dict[str, str]:
    """Normalize a raw ``extra_headers`` value into a ``dict[str, str]``.

    Stringifies keys and values and drops entries whose value is ``None``.
    Returns ``{}`` for non-dict or empty inputs. This is the single shared
    normalizer for per-provider ``extra_headers`` across config normalization,
    runtime resolution, client construction, and live ``/models`` discovery.

    SECURITY: header values routinely carry credentials (Cloudflare Access
    service tokens, proxy auth, custom bearer schemes). Callers must never
    log the returned values.
    """
    if not isinstance(extra_headers, dict) or not extra_headers:
        return {}
    return {str(k): str(v) for k, v in extra_headers.items() if v is not None}


def get_custom_provider_extra_headers(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return ``extra_headers`` from a matching ``providers`` / ``custom_providers`` entry.

    Matches the entry whose normalized route identity equals *base_url*,
    mirroring :func:`get_custom_provider_tls_settings`, and returns its
    ``extra_headers`` dict, or ``{}`` when no entry matches or declares none.

    SECURITY: header values routinely carry credentials (Cloudflare Access
    service tokens, proxy auth, custom bearer schemes). Callers must never
    log the returned values.
    """
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            custom_providers = []
    if not base_url or not isinstance(custom_providers, list):
        return {}

    target_url = normalize_route_base_url(base_url)
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        entry_url = normalize_route_base_url(entry.get("base_url"))
        if not entry_url or entry_url != target_url:
            continue
        return normalize_extra_headers(entry.get("extra_headers"))
    return {}


def apply_custom_provider_extra_headers_to_client_kwargs(
    client_kwargs: Dict[str, Any],
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge per-provider ``extra_headers`` onto OpenAI client ``default_headers``.

    Provider-specific headers win over provider/SDK defaults already present in
    ``client_kwargs`` — they are the most specific configuration level. No-op
    when the base_url matches no ``providers`` / ``custom_providers`` entry or
    the entry declares no headers.

    SECURITY: values may carry credentials — never log them.
    """
    extra_headers = get_custom_provider_extra_headers(base_url, custom_providers, config)
    if not extra_headers:
        return
    merged = dict(client_kwargs.get("default_headers") or {})
    merged.update(extra_headers)
    client_kwargs["default_headers"] = merged


def get_custom_provider_context_length(
    model: str,
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Look up a per-model ``context_length`` override from ``custom_providers``.

    Matches any entry whose normalized route identity equals ``base_url`` and
    returns ``custom_providers[i].models.<model>.context_length`` if present and
    valid.  Returns ``None`` when no override applies.

    This is the single source of truth for custom-provider context overrides,
    used by:
      * ``AIAgent.__init__`` (startup resolution)
      * ``agent.model_metadata.get_model_context_length`` (when custom_providers is threaded through)

    Before this helper existed, the lookup was duplicated in ``run_agent.py``'s
    startup path only; every other path (notably ``/model`` switch) fell back
    to the 128K default.  See #15779.
    """
    if not model or not base_url:
        return None
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            if config is None:
                return None
            raw = config.get("custom_providers")
            custom_providers = raw if isinstance(raw, list) else []
    if not isinstance(custom_providers, list):
        return None

    target_url = normalize_route_base_url(base_url)
    if not target_url:
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        entry_url = normalize_route_base_url(entry.get("base_url"))
        if not entry_url or entry_url != target_url:
            continue
        models = entry.get("models")
        if not isinstance(models, dict):
            continue
        model_cfg = models.get(model)
        if not isinstance(model_cfg, dict):
            continue
        raw_ctx = model_cfg.get("context_length")
        if raw_ctx is None:
            continue
        try:
            ctx = int(raw_ctx)
        except (TypeError, ValueError):
            continue
        if ctx > 0:
            return ctx
    return None


def _coerce_config_version(value: Any) -> int:
    """Return a safe integer config version, treating invalid values as legacy."""
    if isinstance(value, bool):
        return 0
    try:
        version = int(value)
    except (TypeError, ValueError):
        return 0
    return max(version, 0)


def _raw_config_has_explicit_version() -> bool:
    """True when config.yaml exists, parses, and carries a ``_config_version`` key.

    Distinguishes an ANCIENT config (explicit old version → refused by the
    v12 support floor) from a fresh minimal/hand-written/cloned config with
    no version key at all (→ migrated + stamped normally). Missing or
    unparseable files return False so they never trip the floor gate.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return False
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = fast_safe_load(f) or {}
    except Exception:
        return False
    return isinstance(raw, dict) and "_config_version" in raw


def check_config_version() -> Tuple[int, int]:
    """
    Check the raw on-disk config schema version.

    ``load_config()`` deliberately starts from ``DEFAULT_CONFIG`` and deep-merges
    the user's file, which is correct for runtime reads but wrong for deciding
    whether the user's persisted schema has been migrated. A config file with no
    raw ``_config_version`` must remain visible as legacy instead of inheriting
    the latest default version in memory.

    Returns (current_version, latest_version).
    """
    latest = _coerce_config_version(DEFAULT_CONFIG.get("_config_version", 1)) or 1
    config_path = get_config_path()
    if not config_path.exists():
        return latest, latest

    try:
        with open(config_path, encoding="utf-8") as f:
            config = fast_safe_load(f) or {}
    except Exception as e:
        # Invalid YAML needs a parse warning, not an automatic schema rewrite
        # that could replace the user's broken file with defaults.
        _warn_config_parse_failure(config_path, e)
        return latest, latest

    if not isinstance(config, dict):
        config = {}
    current = _coerce_config_version(config.get("_config_version"))
    return current, latest


# =============================================================================
# Config structure validation
# =============================================================================

# Fields that are valid at root level of config.yaml.
# DEFAULT_CONFIG is the single source of truth for documented roots; keep this
# set derived so new defaults (skills, security, browser, …) are accepted
# automatically. A few optional/legacy roots are valid on disk but intentionally
# absent from DEFAULT_CONFIG (omitted when unused / alternate schema forms).
_EXTRA_KNOWN_ROOT_KEYS = {
    "custom_providers",  # legacy list form; modern equivalent is providers: {}
    "fallback_model",    # optional single dict or chain list; omitted when disabled
    "mcp_servers",       # MCP server definitions written by setup/tools flows
    # Roots read from the raw user YAML (or written by our own flows) that are
    # intentionally absent from DEFAULT_CONFIG:
    "image_gen",         # image-generation provider config (agent/image_gen_registry.py)
    "video_gen",         # video-generation provider config (agent/video_gen_registry.py)
    "plugins",           # plugin enable/disable lists (hermes_cli/plugins_cmd.py)
}
_KNOWN_ROOT_KEYS = frozenset(DEFAULT_CONFIG.keys()) | _EXTRA_KNOWN_ROOT_KEYS

# Valid fields inside a custom_providers list entry
_VALID_CUSTOM_PROVIDER_FIELDS = {
    "name", "base_url", "api_key", "api_mode", "model", "models",
    "context_length", "rate_limit_delay", "extra_body",
    "ssl_ca_cert", "ssl_verify",
    # key_env is read at runtime by runtime_provider.py and auxiliary_client.py
    # — include it here so the set accurately describes the supported schema.
    "key_env",
}

# Fields that look like they should be inside custom_providers, not at root
_CUSTOM_PROVIDER_LIKE_FIELDS = {"base_url", "api_key", "rate_limit_delay", "api_mode"}


@dataclass
class ConfigIssue:
    """A detected config structure problem."""

    severity: str  # "error", "warning"
    message: str
    hint: str


def validate_config_structure(config: Optional[Dict[str, Any]] = None) -> List["ConfigIssue"]:
    """Validate config.yaml structure and return a list of detected issues.

    Catches common YAML formatting mistakes that produce confusing runtime
    errors (like "Unknown provider") instead of clear diagnostics.

    Can be called with a pre-loaded config dict, or will load from disk.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            return [ConfigIssue("error", "Could not load config.yaml", "Run 'hermes setup' to create a valid config")]

    issues: List[ConfigIssue] = []

    # ── custom_providers must be a list, not a dict ──────────────────────
    cp = config.get("custom_providers")
    if cp is not None:
        if isinstance(cp, dict):
            issues.append(ConfigIssue(
                "error",
                "custom_providers is a dict — it must be a YAML list (items prefixed with '-')",
                "Change to:\n"
                "  custom_providers:\n"
                "    - name: my-provider\n"
                "      base_url: https://...\n"
                "      api_key: ...",
            ))
            # Check if dict keys look like they should be list-entry fields
            cp_keys = set(cp.keys()) if isinstance(cp, dict) else set()
            suspicious = cp_keys & _CUSTOM_PROVIDER_LIKE_FIELDS
            if suspicious:
                issues.append(ConfigIssue(
                    "warning",
                    f"Root-level keys {sorted(suspicious)} look like custom_providers entry fields",
                    "These should be indented under a '- name: ...' list entry, not at root level",
                ))
        elif isinstance(cp, list):
            # Validate each entry in the list
            for i, entry in enumerate(cp):
                if not isinstance(entry, dict):
                    issues.append(ConfigIssue(
                        "warning",
                        f"custom_providers[{i}] is not a dict (got {type(entry).__name__})",
                        "Each entry should have at minimum: name, base_url",
                    ))
                    continue
                if not entry.get("name"):
                    issues.append(ConfigIssue(
                        "warning",
                        f"custom_providers[{i}] is missing 'name' field",
                        "Add a name, e.g.: name: my-provider",
                    ))
                if not entry.get("base_url"):
                    issues.append(ConfigIssue(
                        "warning",
                        f"custom_providers[{i}] is missing 'base_url' field",
                        "Add the API endpoint URL, e.g.: base_url: https://api.example.com/v1",
                    ))

    # ── fallback_model: single dict OR list of dicts (chain) ─────────────
    fb = config.get("fallback_model")
    if fb is not None:
        if isinstance(fb, list):
            # Chain fallback — validate each entry
            for i, entry in enumerate(fb):
                if not isinstance(entry, dict):
                    issues.append(ConfigIssue(
                        "error",
                        f"fallback_model[{i}] should be a dict, got {type(entry).__name__}",
                        "Each entry needs provider + model",
                    ))
                else:
                    if not entry.get("provider"):
                        issues.append(ConfigIssue(
                            "warning",
                            f"fallback_model[{i}] is missing 'provider' field",
                            "Add: provider: openrouter (or another provider)",
                        ))
                    if not entry.get("model"):
                        issues.append(ConfigIssue(
                            "warning",
                            f"fallback_model[{i}] is missing 'model' field",
                            "Add: model: <model-name>",
                        ))
        elif not isinstance(fb, dict):
            issues.append(ConfigIssue(
                "error",
                f"fallback_model should be a dict with 'provider' and 'model', got {type(fb).__name__}",
                "Change to:\n"
                "  fallback_model:\n"
                "    provider: openrouter\n"
                "    model: anthropic/claude-sonnet-4",
            ))
        elif fb:
            if not fb.get("provider"):
                issues.append(ConfigIssue(
                    "warning",
                    "fallback_model is missing 'provider' field — fallback will be disabled",
                    "Add: provider: openrouter (or another provider)",
                ))
            if not fb.get("model"):
                issues.append(ConfigIssue(
                    "warning",
                    "fallback_model is missing 'model' field — fallback will be disabled",
                    "Add: model: anthropic/claude-sonnet-4 (or another model)",
                ))

    # ── Check for fallback_model accidentally nested inside custom_providers ──
    if isinstance(cp, dict) and "fallback_model" not in config and "fallback_model" in (cp or {}):
        issues.append(ConfigIssue(
            "error",
            "fallback_model appears inside custom_providers instead of at root level",
            "Move fallback_model to the top level of config.yaml (no indentation)",
        ))

    # ── model section: should exist when custom_providers is configured ──
    model_cfg = config.get("model")
    if cp and not model_cfg:
        issues.append(ConfigIssue(
            "warning",
            "custom_providers defined but no 'model' section — Hermes won't know which provider to use",
            "Add a model section:\n"
            "  model:\n"
            "    provider: custom\n"
            "    default: your-model-name\n"
            "    base_url: https://...",
        ))

    # ── Root-level keys that look misplaced ──────────────────────────────
    # Only provider-like fields (base_url, api_key, …) are flagged. Arbitrary
    # unknown top-level keys are deliberately NOT warned about: top-level
    # scalars are bridged into os.environ (gateway/run.py, hermes send) so
    # users can feed skills and external apps env-style keys from config.yaml
    # — a closed-world allowlist can never enumerate those.
    for key in config:
        if key.startswith("_"):
            continue
        if key not in _KNOWN_ROOT_KEYS and key in _CUSTOM_PROVIDER_LIKE_FIELDS:
            issues.append(ConfigIssue(
                "warning",
                f"Root-level key '{key}' looks misplaced — should it be under 'model:' or inside a 'custom_providers' entry?",
                f"Move '{key}' under the appropriate section",
            ))

    return issues


def print_config_warnings(config: Optional[Dict[str, Any]] = None) -> None:
    """Print config structure warnings to stderr at startup.

    Called early in CLI and gateway init so users see problems before
    they hit cryptic "Unknown provider" errors.  Prints nothing if
    config is healthy.
    """
    try:
        issues = validate_config_structure(config)
    except Exception:
        return
    if not issues:
        return

    lines = ["\033[33m⚠ Config issues detected in config.yaml:\033[0m"]
    for ci in issues:
        marker = "\033[31m✗\033[0m" if ci.severity == "error" else "\033[33m⚠\033[0m"
        lines.append(f"  {marker} {ci.message}")
    lines.append("  \033[2mRun 'hermes doctor' for fix suggestions.\033[0m")
    sys.stderr.write("\n".join(lines) + "\n\n")


def warn_deprecated_cwd_env_vars(config: Optional[Dict[str, Any]] = None) -> None:
    """Warn if MESSAGING_CWD or TERMINAL_CWD is set in .env instead of config.yaml.

    These env vars are deprecated — the canonical setting is terminal.cwd
    in config.yaml.  Prints a migration hint to stderr.
    """
    messaging_cwd = os.environ.get("MESSAGING_CWD")
    terminal_cwd_env = os.environ.get("TERMINAL_CWD")

    if config is None:
        try:
            config = load_config()
        except Exception:
            return

    terminal_cfg = config.get("terminal", {})
    config_cwd = terminal_cfg.get("cwd", ".") if isinstance(terminal_cfg, dict) else "."
    # Only warn if config.yaml doesn't have an explicit path
    config_has_explicit_cwd = config_cwd not in {".", "auto", "cwd", ""}

    lines: list[str] = []
    if messaging_cwd:
        lines.append(
            f"  \033[33m⚠\033[0m MESSAGING_CWD={messaging_cwd} found in .env — "
            f"this is deprecated."
        )
    if terminal_cwd_env and not config_has_explicit_cwd:
        # TERMINAL_CWD in env but not from config bridge — likely from .env
        lines.append(
            f"  \033[33m⚠\033[0m TERMINAL_CWD={terminal_cwd_env} found in .env — "
            f"this is deprecated."
        )
    if lines:
        from hermes_constants import display_hermes_home

        hint_path = display_hermes_home()
        lines.insert(0, "\033[33m⚠ Deprecated .env settings detected:\033[0m")
        lines.append(
            "  \033[2mMove to config.yaml instead:  "
            "terminal:\\n    cwd: /your/project/path\033[0m"
        )
        lines.append(
            f"  \033[2mThen remove the old entries from {hint_path}/.env\033[0m"
        )
        sys.stderr.write("\n".join(lines) + "\n\n")


def _persist_migration(config: Dict[str, Any]) -> None:
    """Persist a migrated config under the migration write invariant.

    THE INVARIANT (single source of truth for the whole migration pipeline):
    a migration may only persist values that DIFFER from the current schema
    default, plus explicit removals/renames of user data. Pure schema defaults
    are never materialised to disk — ``load_config()``'s deep-merge supplies
    them at read time, so writing them adds nothing and actively shadows future
    default changes (see ``save_config``'s docstring). Materialising defaults on
    every version bump is what rewrote hand-curated configs into full
    DEFAULT_CONFIG dumps (the "hermes update / hermes -p blows up my config"
    reports).

    Every migration step MUST route its write through this helper instead of
    calling ``save_config`` directly. It is a thin wrapper over
    ``save_config(config)`` (default-stripping ON, no ``merge_existing``);
    centralising the call makes the invariant impossible to regress one
    migration at a time. Callers must pass the full raw config returned by
    ``read_raw_config()`` after in-place mutations (including key removals);
    deep-merging the on-disk file back in would resurrect keys the migration
    just deleted. Partial-save preservation for unrelated top-level sections
    belongs on ``save_config(..., merge_existing=True)``, not here.
    """
    save_config(config)


def migrate_config(interactive: bool = True, quiet: bool = False) -> Dict[str, Any]:
    """
    Migrate config to latest version, prompting for new required fields.
    
    Args:
        interactive: If True, prompt user for missing values
        quiet: If True, suppress output
        
    Returns:
        Dict with migration results: {"env_added": [...], "config_added": [...], "warnings": [...]}
    """
    results = {"env_added": [], "config_added": [], "warnings": []}

    # ── Always: normalize safe .env line formatting ──
    try:
        fixes = sanitize_env_file()
        if fixes and not quiet:
            print(f"  ✓ Normalized .env line formatting ({fixes} line(s) changed)")
    except Exception:
        pass  # best-effort; don't block migration on sanitize failure

    # Check config version
    current_ver, latest_ver = check_config_version()

    # ── Auto-migration support floor (policy: v12, July 2026) ──
    # A config with an EXPLICIT on-disk ``_config_version`` below the floor is
    # NOT auto-migrated and NOT rewritten: we surface a clear, actionable
    # message and leave the file byte-for-byte untouched. This matches the
    # fail-safe posture for unparseable configs (warn on stderr, continue —
    # load_config() deep-merges defaults at read time), so the CLI never
    # crashes on an ancient config. The floor gate lives here in the wrapper
    # (not in run_migrations) so the registry driver stays a pure mechanism
    # that tests can exercise directly.
    #
    # A config with NO ``_config_version`` key at all is NOT floor-refused:
    # that shape is a fresh minimal config (profile clones write bare keys;
    # users hand-write two-line configs), not an ancient install. Those get
    # the normal ladder (the retired <12 steps were no-ops for configs
    # lacking the legacy keys they migrated) and a fresh version stamp —
    # the historical behavior.
    from hermes_cli.config_migrations import (
        SUPPORT_FLOOR_VERSION,
        run_migrations,
        support_floor_message,
    )

    _explicit_version = _raw_config_has_explicit_version()
    floor_refused = (
        _explicit_version
        and current_ver < SUPPORT_FLOOR_VERSION
        and current_ver < latest_ver
    )
    if floor_refused:
        msg = support_floor_message()
        results["warnings"].append(msg)
        # stderr so it is visible even on quiet startup paths, matching the
        # corrupt-config warning posture in _warn_config_parse_failure().
        sys.stderr.write(f"⚠ hermes config: {msg}\n")
        if not quiet:
            print(f"  ⚠ {msg}")
    else:
        # ── Versioned migration ladder (table-driven) ──
        # The per-version steps live in hermes_cli.config_migrations as a
        # (target_version, fn) registry; the driver applies every step whose
        # target exceeds current_ver, in strict ascending order, preserving the
        # original sequential if-block semantics byte-for-byte. Imported lazily
        # to avoid a module-level import cycle (the steps call back into this
        # module for read_raw_config/_persist_migration/etc. at call time).
        run_migrations(current_ver, results, quiet)

    # ── Post-migration: disable exfiltration-shaped MCP stdio entries ──
    # Users can hand-edit mcp_servers, and older installs may already contain a
    # malicious entry. Preserve the stanza for auditability but mark it
    # disabled so the next startup will not spawn it. (#45620)
    config = read_raw_config()
    raw_mcp_servers = config.get("mcp_servers")
    if isinstance(raw_mcp_servers, dict):
        try:
            from hermes_cli.mcp_security import validate_mcp_server_entry as _validate_mcp_server_entry
        except Exception:
            _validate_mcp_server_entry = None
        if _validate_mcp_server_entry:
            mcp_touched = False
            for server_name, entry in raw_mcp_servers.items():
                if not isinstance(entry, dict):
                    continue
                issues = _validate_mcp_server_entry(server_name, entry)
                if not issues:
                    continue
                entry["enabled"] = False
                mcp_touched = True
                results["warnings"].append(
                    f"Disabled suspicious MCP server '{server_name}'"
                )
                if not quiet:
                    for issue in issues:
                        print(f"  ⚠ {issue}")
                    print(f"  ⚠ Disabled MCP server '{server_name}' pending review")
            if mcp_touched:
                config["mcp_servers"] = raw_mcp_servers
                _persist_migration(config)

    # ── Always: validate platform_toolsets after migration ──
    # A migration (or hand-edit) that leaves an invalid toolset name in
    # platform_toolsets silently disables the affected tools — resolve_toolset()
    # returns [] for an unknown name, so the agent quietly loses tools with no
    # error or warning. Surface it loudly instead. See #38798.
    try:
        from toolsets import validate_toolset
        from hermes_cli.toolset_validation import validate_platform_toolsets

        ts_warnings = validate_platform_toolsets(
            read_raw_config().get("platform_toolsets"), validate_toolset
        )
        for w in ts_warnings:
            results["warnings"].append(w)
            if not quiet:
                print(f"  ⚠ {w}")
    except Exception as _ts_val_err:
        # best-effort; never block migration on validation
        logger.debug("platform_toolsets validation skipped: %s", _ts_val_err)

    if current_ver < latest_ver and not quiet and not floor_refused:
        print(f"Config version: {current_ver} → {latest_ver}")

    # Check for missing required env vars
    missing_env = get_missing_env_vars(required_only=True)
    
    if missing_env and not quiet:
        print("\n⚠️  Missing required environment variables:")
        for var in missing_env:
            print(f"   • {var['name']}: {var['description']}")
    
    if interactive and missing_env:
        print("\nLet's configure them now:\n")
        for var in missing_env:
            if var.get("url"):
                print(f"  Get your key at: {var['url']}")
            
            if var.get("password"):
                value = masked_secret_prompt(f"  {var['prompt']}: ")
            else:
                value = input(f"  {var['prompt']}: ").strip()
            
            if value:
                save_env_value(var["name"], value)
                results["env_added"].append(var["name"])
                print(f"  ✓ Saved {var['name']}")
            else:
                results["warnings"].append(f"Skipped {var['name']} - some features may not work")
            print()
    
    # Check for missing optional env vars and offer to configure interactively
    # Skip "advanced" vars (like OPENAI_BASE_URL) -- those are for power users
    missing_optional = get_missing_env_vars(required_only=False)
    required_names = {v["name"] for v in missing_env} if missing_env else set()
    missing_optional = [
        v for v in missing_optional
        if v["name"] not in required_names and not v.get("advanced")
    ]
    
    # Only offer to configure env vars that are NEW since the user's previous version
    new_var_names = set()
    for ver in range(current_ver + 1, latest_ver + 1):
        new_var_names.update(ENV_VARS_BY_VERSION.get(ver, []))

    if new_var_names and interactive and not quiet:
        new_and_unset = [
            (name, OPTIONAL_ENV_VARS[name])
            for name in sorted(new_var_names)
            if not get_env_value(name) and name in OPTIONAL_ENV_VARS
        ]
        if new_and_unset:
            print(f"\n  {len(new_and_unset)} new optional key(s) in this update:")
            for name, info in new_and_unset:
                print(f"    • {name} — {info.get('description', '')}")
            print()
            try:
                answer = input("  Configure new keys? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer in {"y", "yes"}:
                print()
                for name, info in new_and_unset:
                    if info.get("url"):
                        print(f"  {info.get('description', name)}")
                        print(f"  Get your key at: {info['url']}")
                    else:
                        print(f"  {info.get('description', name)}")
                    if info.get("password"):
                        value = masked_secret_prompt(
                            f"  {info.get('prompt', name)} (Enter to skip): "
                        )
                    else:
                        value = input(f"  {info.get('prompt', name)} (Enter to skip): ").strip()
                    if value:
                        save_env_value(name, value)
                        results["env_added"].append(name)
                        print(f"  ✓ Saved {name}")
                    print()
            else:
                print("  Set later with: hermes config set <key> <value>")
    
    # Check for missing config fields.
    #
    # New default keys are NOT materialised to disk: load_config() deep-merges
    # DEFAULT_CONFIG at read time, so a missing key already takes effect with
    # its default (see _persist_migration's invariant). We surface the list for
    # the informational "N new config option(s) available" display in
    # `hermes update`, but only the version bump is persisted.
    missing_config = get_missing_config_fields()
    if missing_config:
        results["config_added"].extend(field["key"] for field in missing_config)

    if current_ver < latest_ver and not floor_refused:
        config = read_raw_config()
        config["_config_version"] = latest_ver
        _persist_migration(config)

    return results


def _merge_partial_save(raw: dict, override: dict) -> dict:
    """Merge *override* over *raw* for partial ``save_config`` writes.

    Top-level sections omitted from *override* are preserved from *raw*.
    Shared top-level dict sections are deep-merged so a caller can update one
    nested key without dropping sibling keys from disk. Intentional key
    removals within a section are not supported here — migration writes must
    route through ``_persist_migration`` with a full ``read_raw_config()`` dict
    instead.
    """
    result = copy.deepcopy(override)
    for key, value in raw.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
        elif isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(value, result[key])
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, preserving nested defaults.

    Keys in *override* take precedence. If both values are dicts the merge
    recurses, so a user who overrides one provider field keeps its sibling
    defaults intact.

    An empty section key in config.yaml (``terminal:`` with no value) parses
    as YAML ``None``; treating that as an override would replace the entire
    default dict with ``None`` and crash every downstream consumer that
    expects a mapping (#58277). A ``None`` override of a dict default is
    ignored — same as the key being absent.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], dict) and value is None:
            continue
        else:
            result[key] = value
    return result


def _strip_dotted_keys(cfg: dict, dotted_keys: set) -> Tuple[dict, set]:
    """Remove the given dotted leaf keys from a nested config dict.

    Returns ``(pruned_cfg, set_of_stripped_keys_that_were_present)``. Used by
    ``save_config`` to drop managed-scope leaves before persisting, so a bulk
    write never writes a user value that would lose to the managed layer on the
    next load. Only keys actually present in ``cfg`` are reported as stripped.
    """
    stripped: set = set()
    for dotted in dotted_keys:
        parts = dotted.split(".")
        node = cfg
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                node = None
                break
            node = node[p]
        if isinstance(node, dict) and parts[-1] in node:
            del node[parts[-1]]
            stripped.add(dotted)
    return cfg, stripped


def _env_expand_match(m: re.Match) -> str:
    """Expand one ``${...}`` config reference.

    Two accepted shapes, matching what MCP server config already resolves
    (``tools/mcp_tool.py::_env_ref_name``):

    * ``${VAR}`` — legacy bare name, resolved via ``os.environ``.
    * ``${env:VAR}`` — Cursor-style SecretRef, same resolution after the
      ``env:`` prefix is stripped.  Before this, the prefixed form worked in
      MCP config but stayed a literal string in config.yaml — a confusing
      half-support.

    Other SecretRef sources (``file:``, ``bitwarden:``, ``vault:``, ...)
    are NOT resolved here — external secret backends inject their values
    into the environment at startup (the ``secrets:`` block), so a config
    ref only ever needs the env shape.  Unknown prefixes warn once and stay
    verbatim so callers can detect them.
    """
    raw = m.group(0)
    inner = m.group(1).strip()
    if inner.startswith("env:"):
        name = inner[len("env:"):].strip()
        if not name:
            return raw
        val = os.environ.get(name)
        if val is not None:
            return val
        logger.warning(
            "Config ref %r: %s is not set (check ~/.hermes/.env); "
            "keeping the literal placeholder", raw, name,
        )
        return raw
    if ":" in inner and re.match(r"^[a-z][a-z0-9_-]*:", inner):
        # Looks like a SecretRef with a non-env source.  Values from vault
        # backends arrive via the secrets: block as env vars — point there
        # instead of silently treating "bitwarden:FOO" as a var named
        # "bitwarden:FOO".
        logger.warning(
            "Config ref %r uses source %r which is not resolvable in "
            "config.yaml — external secret sources inject env vars at "
            "startup, so reference the variable as ${env:NAME} instead",
            raw, inner.split(":", 1)[0],
        )
        return raw
    # Legacy ``${VAR}`` — bare name.
    return os.environ.get(inner, raw)


def _env_ref_var_name(ref: str) -> Optional[str]:
    """Normalize a ``${...}`` body to the env-var name it reads, or None
    when the ref uses a non-env source and never touches the environment."""
    ref = ref.strip()
    if ref.startswith("env:"):
        name = ref[len("env:"):].strip()
        return name or None
    if ":" in ref and re.match(r"^[a-z][a-z0-9_-]*:", ref):
        return None
    return ref


def _expand_env_vars(obj):
    """Recursively expand ``${VAR}`` / ``${env:VAR}`` references in config
    values.

    Only string values are processed; dict keys, numbers, booleans, and
    None are left untouched.  Unresolved references (variable not in
    ``os.environ``) are kept verbatim so callers can detect them.
    """
    if isinstance(obj, str):
        return re.sub(r"\${([^}]+)}", _env_expand_match, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def _env_ref_snapshot(obj, snapshot=None):
    """Map every ``${VAR}`` / ``${env:VAR}`` name referenced in config values
    to its current ``os.environ`` value (``None`` when unset).

    Stored alongside cached ``load_config()`` results so a cache hit can
    detect that the cached expansion was made against a *different*
    environment — e.g. a ``load_config()`` that ran before
    ``load_hermes_dotenv()`` populated the process env, or an env var
    rotated in-process after the first load. File mtime/size alone cannot
    see either case (#58514).

    ``${env:VAR}`` refs are tracked under the real variable name; refs
    with a non-env source prefix never read the environment, so they are
    excluded from the snapshot.
    """
    if snapshot is None:
        snapshot = {}
    if isinstance(obj, str):
        for raw in re.findall(r"\${([^}]+)}", obj):
            name = _env_ref_var_name(raw)
            if name is not None:
                snapshot[name] = os.environ.get(name)
    elif isinstance(obj, dict):
        for value in obj.values():
            _env_ref_snapshot(value, snapshot)
    elif isinstance(obj, list):
        for item in obj:
            _env_ref_snapshot(item, snapshot)
    return snapshot


def _items_by_unique_name(items):
    """Return a name-indexed dict only when all items have unique string names."""
    if not isinstance(items, list):
        return None
    indexed = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        name = item["name"]
        if name in indexed:
            return None
        indexed[name] = item
    return indexed


def _preserve_env_ref_templates(current, raw, loaded_expanded=None):
    """Restore raw ``${VAR}`` templates when a value is otherwise unchanged.

    ``load_config()`` expands env refs for runtime use. When a caller later
    persists that config after modifying some unrelated setting, keep the
    original on-disk template instead of writing the expanded plaintext
    secret back to ``config.yaml``.

    Prefer preserving the raw template when ``current`` still matches either
    the value previously returned by ``load_config()`` for this config path or
    the current environment expansion of ``raw``. This handles env-var
    rotation between load and save while still treating mixed literal/template
    string edits as caller-owned once their rendered value diverges.
    """
    if isinstance(current, str) and isinstance(raw, str) and re.search(r"\${[^}]+}", raw):
        if current == raw:
            return raw
        if isinstance(loaded_expanded, str) and current == loaded_expanded:
            return raw
        if _expand_env_vars(raw) == current:
            return raw
        return current

    if isinstance(current, dict) and isinstance(raw, dict):
        return {
            key: _preserve_env_ref_templates(
                value,
                raw.get(key),
                loaded_expanded.get(key) if isinstance(loaded_expanded, dict) else None,
            )
            for key, value in current.items()
        }

    if isinstance(current, list) and isinstance(raw, list):
        # Prefer matching named config objects (e.g. custom_providers) by name
        # so harmless reordering doesn't drop the original template. If names
        # are duplicated, fall back to positional matching instead of silently
        # shadowing one entry.
        current_by_name = _items_by_unique_name(current)
        raw_by_name = _items_by_unique_name(raw)
        loaded_by_name = _items_by_unique_name(loaded_expanded)
        if current_by_name is not None and raw_by_name is not None:
            return [
                _preserve_env_ref_templates(
                    item,
                    raw_by_name.get(item.get("name")),
                    loaded_by_name.get(item.get("name")) if loaded_by_name is not None else None,
                )
                for item in current
            ]
        return [
            _preserve_env_ref_templates(
                item,
                raw[index] if index < len(raw) else None,
                loaded_expanded[index]
                if isinstance(loaded_expanded, list) and index < len(loaded_expanded)
                else None,
            )
            for index, item in enumerate(current)
        ]

    return current


def _explicit_config_paths(config: Dict[str, Any]) -> Set[Tuple[str, ...]]:
    """Return leaf paths explicitly present in a raw config dict.

    Computed on the **raw** (un-normalized, un-expanded) config so that
    values injected by normalisation (e.g. ``agent.max_turns`` from
    ``DEFAULT_CONFIG``) are not mistakenly treated as user-set.

    Used by ``save_config`` to build the *preserve* set passed to
    ``_strip_default_values`` so only user-authored keys survive the
    defaults-strip pass.
    """
    paths: Set[Tuple[str, ...]] = set()

    def _walk(value: Any, path: Tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(child, path + (key,))
            return
        if path:
            paths.add(path)

    _walk(config, ())
    return paths


def _strip_default_values(
    config: Dict[str, Any],
    defaults: Dict[str, Any] = DEFAULT_CONFIG,
    preserve_keys: Optional[Set[Tuple[str, ...]]] = None,
) -> Dict[str, Any]:
    """Return *config* without keys whose values match *defaults*.

    Keys in *preserve_keys* (explicitly present in the user's raw config,
    before any normalisation) are always kept even when they equal the
    default, so user-set values such as ``memory.user_char_limit: 2200``
    survive a ``save_config`` round-trip.

    Nested dicts whose every child is stripped are removed entirely so
    default-only subtrees (e.g. ``gateway``) never bloat ``config.yaml``
    when the user has nothing to say about them.
    """
    preserve_keys = {("_config_version",)} | set(preserve_keys or ())

    def _strip(value: Any, default: Any, path: Tuple[str, ...]) -> Any:
        if path in preserve_keys:
            return copy.deepcopy(value)

        if isinstance(value, dict) and value:
            default_dict = default if isinstance(default, dict) else {}
            stripped: Dict[str, Any] = {}
            for key, child in value.items():
                child_default = default_dict.get(key)
                stripped_child = _strip(child, child_default, path + (key,))
                if stripped_child is not None:
                    stripped[key] = stripped_child
            if stripped:
                return stripped
            # Entire subtree stripped — remove it
            return None

        if value == default:
            return None

        return copy.deepcopy(value)

    result: Dict[str, Any] = {}
    for key, value in config.items():
        stripped = _strip(value, defaults.get(key), (key,))
        if stripped is not None:
            result[key] = stripped
    return result


def _normalize_root_model_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Move stale root-level provider/base_url/context_length into model section.

    Some users (or older code) placed ``provider:``, ``base_url:``, or
    ``context_length:`` at the config root instead of inside ``model:``.
    These root-level keys are only used as a fallback when the corresponding
    ``model.*`` key is empty — they never override an existing value.
    After migration the root-level keys are removed so they can't cause
    confusion on subsequent loads.

    Also aliases ``api_base`` → ``base_url`` (issue #8919). ``api_base`` is the
    intuitive name OpenAI-SDK / LiteLLM users reach for, and ``hermes config set``
    blindly accepts any dotted key — so ``model.api_base`` got written, confirmed,
    and then silently ignored by the runtime resolver (which reads only
    ``model.base_url``), causing requests to fall back to OpenRouter. We migrate
    the alias to the canonical key (fallback-only — never override an explicit
    ``base_url``) and drop the alias so it can't confuse later loads.

    Finally, canonicalizes the model-id key to ``model.default`` (issue #34500).
    The runtime resolver and ~14 other readers select the chat model via
    ``model.default``; ``model.model`` was already aliased inline at some sites
    but ``model.name`` was not, so a custom-provider config like
    ``model: {name: <id>, provider: <custom>}`` resolved to an empty model and
    the API request went out with ``model=`` (HTTP 400 from OpenAI-compatible
    backends) — while display paths (``hermes status``/``dump``) read ``name``
    and *showed* the model, making the failure silent. Normalizing here (the
    single load/save chokepoint) means every reader, present and future, sees a
    populated ``default`` and the stale alias is migrated out of config.yaml on
    the next save. Precedence: ``default`` > ``model`` > ``name`` (never
    overrides an explicit ``default``, so existing configs are unaffected).
    """
    # Only act if there's something to migrate: root-level keys, an api_base
    # alias, or a model dict whose id lives under a non-canonical key.
    model_in = config.get("model")
    model_has_alias = isinstance(model_in, dict) and model_in.get("api_base")
    # A model dict needs canonicalization if its id lives under a non-canonical
    # key (``model``/``name``) — either because ``default`` is empty (we must
    # promote the alias) or because ``default`` is set but a stale alias still
    # lingers (we must drop it so config.yaml ends up canonical).
    model_needs_canon = isinstance(model_in, dict) and (
        model_in.get("model") or model_in.get("name")
    )
    has_root = any(
        config.get(k) for k in ("provider", "base_url", "context_length", "api_base")
    )
    if not has_root and not model_has_alias and not model_needs_canon:
        return config

    config = dict(config)
    model = config.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
    else:
        model = dict(model)
    config["model"] = model

    for key in ("provider", "base_url", "context_length"):
        root_val = config.get(key)
        if root_val and not model.get(key):
            model[key] = root_val
        config.pop(key, None)

    # api_base is an alias for base_url, at the root OR inside model.
    for alias_val in (config.get("api_base"), model.get("api_base")):
        if alias_val and not model.get("base_url"):
            model["base_url"] = alias_val
    config.pop("api_base", None)
    model.pop("api_base", None)

    # Canonicalize the model id to ``default``. ``model`` and ``name`` are
    # last-resort aliases (in that order) — only consulted when ``default`` is
    # empty, then dropped so later loads/saves can't reintroduce the ambiguity.
    if not (model.get("default") or ""):
        alias = model.get("model") or model.get("name")
        if alias:
            model["default"] = alias
    if model.get("default"):
        # Drop the now-redundant aliases so config.yaml ends up canonical.
        model.pop("model", None)
        model.pop("name", None)

    return config


def _normalize_max_turns_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy root-level max_turns into agent.max_turns.

    Only injects the schema default when the user actually set max_turns
    somewhere (root level or under ``agent``).  A bare ``load_config()``
    call that passes the result straight to ``save_config()`` should not
    materialise ``agent.max_turns`` in config.yaml when the user never set
    it — that makes the default sticky and blocks future schema changes.
    """
    config = dict(config)
    agent_config = dict(config.get("agent") or {})

    had_root = "max_turns" in config
    had_agent = "max_turns" in agent_config

    if had_root and not had_agent:
        agent_config["max_turns"] = config["max_turns"]

    # Only inject the default when the user explicitly set max_turns
    # (either root-level or under agent).  Otherwise leave it absent so
    # save_config can omit it and the schema default fills in at runtime.
    if not had_root and not had_agent:
        pass  # deliberately do not inject DEFAULT_CONFIG default
    elif "max_turns" not in agent_config:
        agent_config["max_turns"] = DEFAULT_CONFIG["agent"]["max_turns"]

    if agent_config:
        config["agent"] = agent_config
    else:
        config.pop("agent", None)
    config.pop("max_turns", None)
    return config


def is_provider_enabled(provider_cfg: Optional[Dict[str, Any]]) -> bool:
    """Return whether a ``providers.<name>`` config block is enabled.

    A provider is enabled by default. Only an explicit ``enabled: false`` in
    the block hides it from the model picker, ``/models`` listings, the
    runtime resolver and the doctor / status output.

    Backward-compat: configs without the ``enabled`` key keep working as
    before — the default is ``True``.

    Pass any non-dict (None, list, string) and you get ``True`` too, so
    malformed entries don't disappear silently; they'll still be flagged
    by the existing validation paths.
    """
    if not isinstance(provider_cfg, dict):
        return True
    flag = provider_cfg.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    # YAML can produce strings for "true"/"false" depending on quoting.
    if isinstance(flag, str):
        return flag.strip().lower() not in {"false", "0", "no", "off"}
    return bool(flag)


def cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Traverse nested dict keys safely, returning ``default`` on any miss.

    Canonical helper for the ``cfg.get("X", {}).get("Y", default)`` pattern
    that appears 50+ times across the codebase. Handles three common gotchas
    in one place:

      1. Missing intermediate keys (returns ``default``, no KeyError).
      2. An intermediate value that's not a dict (e.g. a user wrote a string
         where a section was expected). Returns ``default`` instead of
         AttributeError on ``.get()``.
      3. ``cfg is None`` (callers sometimes pass ``load_config() or None``).

    Named ``cfg_get`` rather than ``cfg_path`` to avoid shadowing the
    ubiquitous ``cfg_path = _hermes_home / "config.yaml"`` local variable
    that appears in gateway/run.py, cron/scheduler.py, main.py, etc.

    Explicit ``None`` values are returned as-is (matches ``dict.get(key,
    default)`` semantics — ``default`` is only returned when the key is
    *absent*, not when it's present but set to ``None``).

    Examples:
        >>> cfg_get({"agent": {"reasoning_effort": "high"}}, "agent", "reasoning_effort")
        'high'
        >>> cfg_get({}, "agent", "reasoning_effort", default="medium")
        'medium'
        >>> cfg_get({"agent": "oops_a_string"}, "agent", "reasoning_effort", default="low")
        'low'
        >>> cfg_get(None, "anything", default=42)
        42
        >>> cfg_get({"a": {"b": None}}, "a", "b", default="def")  # explicit None preserved
        >>> cfg_get({"a": {"b": False}}, "a", "b", default=True)  # falsy values preserved
        False
    """
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        if key not in node:
            return default
        node = node[key]
    return node



def read_raw_config() -> Dict[str, Any]:
    """Read ~/.hermes/config.yaml as-is, without merging defaults or migrating.

    Returns the raw YAML dict, or ``{}`` if the file doesn't exist or can't
    be parsed.  Use this for lightweight config reads where you just need a
    single value and don't want the overhead of ``load_config()``'s deep-merge
    + migration pipeline.

    Cached on the config file's (mtime_ns, size) — same strategy as
    ``load_config()``. Returns a deepcopy on every call since some callers
    mutate the result before passing to ``save_config()``.
    """
    with _CONFIG_LOCK:
        try:
            config_path = get_config_path()
            st = config_path.stat()
            cache_key = (st.st_mtime_ns, st.st_size)
        except (FileNotFoundError, OSError):
            return {}

        path_key = str(config_path)
        cached = _RAW_CONFIG_CACHE.get(path_key)
        if cached is not None and cached[:2] == cache_key:
            return copy.deepcopy(cached[2])

        try:
            with open(config_path, encoding="utf-8") as f:
                data = fast_safe_load(f) or {}
        except Exception as e:
            _warn_config_parse_failure(config_path, e)
            return {}

        if not isinstance(data, dict):
            data = {}
        _RAW_CONFIG_CACHE[path_key] = (cache_key[0], cache_key[1], copy.deepcopy(data))
        return data


def read_user_config_raw(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read a user ``config.yaml`` EXACTLY as written on disk.

    No DEFAULT_CONFIG merge, no managed-scope overlay, no ``${ENV_VAR}``
    expansion, no migration, no root-model normalization, no caching.

    ONLY legal for write-back round-trips and raw-file diagnostics —
    behavioral reads must use load_config()/load_config_readonly().

    Legal call sites, exhaustively:

      * WRITE-BACK ROUND-TRIPS (read → mutate one key → save): merging
        defaults or the managed overlay here would persist hundreds of
        default keys (or administrator-pinned values) into the user's file
        on the next save. Raw is *correct*, not an optimization.
      * RAW-FILE DIAGNOSTICS (doctor, deprecation sweeps): these inspect
        what the user actually wrote — stale root keys, drift against .env —
        and merged defaults would produce false positives.
      * PRESENCE-SENSITIVE ENV BRIDGES (gateway/send bridges that only
        export a key when the user explicitly set it): a defaults merge
        would make every key "present" and bridge the entire DEFAULT_CONFIG
        into the environment. These sites must still apply
        ``managed_scope.apply_managed_overlay`` + ``_expand_env_vars``
        inline, which they do.

    Semantics (deliberately mirrors the bare ``open()+yaml.safe_load()``
    pattern this replaces, so migrated sites keep their exact failure
    behavior):

      * missing file → ``{}``
      * unparseable YAML / other I/O errors → raises (callers that want
        fail-open already wrap in try/except; callers with last-known-good
        or warn semantics rely on the exception)
      * non-dict YAML root → ``{}``

    ``config_path`` defaults to :func:`get_config_path` (profile-aware).
    Pass an explicit path when the caller resolves its own home (gateway
    ``_hermes_home``, tui profile override, multi-profile probes).
    """
    if config_path is None:
        config_path = get_config_path()
    try:
        with open(config_path, encoding="utf-8") as f:
            data = fast_safe_load(f) or {}
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def read_raw_config_readonly() -> Dict[str, Any]:
    """Fast-path variant of ``read_raw_config()`` for callers that ONLY READ.

    Returns the cached raw-config dict directly, skipping the per-call
    ``copy.deepcopy`` that ``read_raw_config()`` performs (needed there
    because some callers mutate the result before ``save_config``).

    **Mutating the returned dict corrupts the in-process cache for every
    subsequent caller.** Only use on read-only paths — e.g. per-turn policy
    checks like the shared-metrics gate, which runs 2-3x per agent turn and
    was paying a full config deepcopy each time.

    Same (mtime_ns, size) freshness key as ``read_raw_config()`` — an edited
    config.yaml is picked up on the next call.
    """
    with _CONFIG_LOCK:
        try:
            config_path = get_config_path()
            st = config_path.stat()
            cache_key = (st.st_mtime_ns, st.st_size)
        except (FileNotFoundError, OSError):
            return {}

        path_key = str(config_path)
        cached = _RAW_CONFIG_CACHE.get(path_key)
        if cached is not None and cached[:2] == cache_key:
            return cached[2]

        try:
            with open(config_path, encoding="utf-8") as f:
                data = fast_safe_load(f) or {}
        except Exception as e:
            _warn_config_parse_failure(config_path, e)
            return {}

        if not isinstance(data, dict):
            data = {}
        # Store and return THE SAME object (identity invariant): the first
        # caller must see the exact dict later cache hits return, so a test
        # asserting ``ro1 is ro2`` holds from the very first call.
        cached_copy = copy.deepcopy(data)
        _RAW_CONFIG_CACHE[path_key] = (cache_key[0], cache_key[1], cached_copy)
        return cached_copy


def require_readable_config_before_write(config_path: Optional[Path] = None) -> None:
    """Refuse to replace an existing config.yaml that cannot be read."""
    if config_path is None:
        config_path = get_config_path()
    try:
        config_path.stat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"Refusing to overwrite {config_path}: existing config.yaml cannot be accessed "
            f"({exc}). Fix the file permissions or move it aside first."
        ) from exc

    try:
        with open(config_path, "rb") as f:
            f.read(1)
    except OSError as exc:
        raise RuntimeError(
            f"Refusing to overwrite {config_path}: existing config.yaml cannot be read "
            f"({exc}). Fix the file permissions or move it aside first."
        ) from exc


def atomic_config_write(config_path: Path, data: Any, **kwargs: Any) -> None:
    """Fail-closed atomic write for ``config.yaml``.

    The single chokepoint every config-update path should use instead of
    calling :func:`utils.atomic_yaml_write` directly. It runs
    :func:`require_readable_config_before_write` first, so a full-file
    replacement can never silently clobber an existing ``config.yaml`` that
    degraded to an empty dict on read (permission error, broken mount,
    transient I/O). New-file creation still works when the path is absent.

    Root cause this guards: ``read_raw_config()`` returns ``{}`` for BOTH an
    absent file and an unreadable-but-present file. Callers that read then
    overwrite can't tell the two apart, so an unreadable config would be
    replaced with only defaults or the single edited section. Routing every
    write through this helper enforces the invariant in one place rather than
    relying on each of ~15 independent write sites to remember the guard.

    ``kwargs`` are forwarded verbatim to ``atomic_yaml_write``
    (``sort_keys``, ``default_flow_style``, ``extra_content``, ...).
    """
    from utils import atomic_yaml_write

    require_readable_config_before_write(config_path)
    atomic_yaml_write(config_path, data, **kwargs)


def load_config() -> Dict[str, Any]:
    """Load configuration from ~/.hermes/config.yaml.

    Cached on the config file's (mtime_ns, size). Returns a deepcopy of
    the cached value when unchanged, since most call sites mutate the
    result (e.g. ``cfg["model"]["default"] = ...`` before ``save_config``).
    The cache is keyed on ``str(config_path)`` so profile switches
    (which change ``HERMES_HOME`` and therefore ``get_config_path()``)
    don't collide.

    Read-only callers should use ``load_config_readonly()`` to skip the
    defensive deepcopy — that path matters in agent-loop hot spots like
    ``get_provider_request_timeout`` which is called once per API turn.
    """
    return _load_config_impl(want_deepcopy=True)


async def load_config_readonly() -> Dict[str, Any]:
    """Load behavioral configuration without synchronous file I/O.

    Preserve the upstream read-only cache contract through native async file
    operations: file signatures and referenced environment values invalidate
    the cache, repeat hits return the same object, managed policy wins at the
    leaf, and malformed edits retain the last-known-good configuration.
    """
    import aiofiles
    import aiofiles.os

    async def _signature(
        path: Path, *, suppress_os_error: bool = False
    ) -> Optional[Tuple[int, int]]:
        try:
            stat_result = await aiofiles.os.stat(path)
        except FileNotFoundError:
            return None
        except OSError:
            if suppress_os_error:
                return None
            raise
        return stat_result.st_mtime_ns, stat_result.st_size

    async def _read_yaml(path: Path) -> Any:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            return fast_safe_load(await handle.read()) or {}

    async with _get_async_config_lock():
        config_path = get_config_path()
        path_key = str(config_path)
        user_sig = await _signature(config_path)

        from hermes_cli import managed_scope

        managed_dir = managed_scope._managed_dir_candidate()
        managed_path = Path(managed_dir) / "config.yaml" if managed_dir else None
        managed_sig = (
            await _signature(managed_path, suppress_os_error=True)
            if managed_path
            else None
        )
        managed_cache_sig = managed_sig or (0, 0)

        if user_sig is not None:
            cache_sig: Optional[Tuple[int, int, int, int]] = (
                user_sig[0],
                user_sig[1],
                managed_cache_sig[0],
                managed_cache_sig[1],
            )
        elif managed_sig is not None:
            cache_sig = (0, 0, managed_sig[0], managed_sig[1])
        else:
            cache_sig = None

        cached = _LOAD_CONFIG_CACHE.get(path_key)
        if cached is not None and cache_sig is not None and cached[:4] == cache_sig:
            env_snapshot = cached[5] if len(cached) > 5 else {}
            if all(
                os.environ.get(key) == value
                for key, value in env_snapshot.items()
            ):
                return cached[4]

        config = copy.deepcopy(DEFAULT_CONFIG)
        if user_sig is not None:
            try:
                user_config = await _read_yaml(config_path)
                if "max_turns" in user_config:
                    agent_user_config = dict(user_config.get("agent") or {})
                    if agent_user_config.get("max_turns") is None:
                        agent_user_config["max_turns"] = user_config["max_turns"]
                    user_config["agent"] = agent_user_config
                    user_config.pop("max_turns", None)
                config = _deep_merge(config, user_config)
            except Exception as exc:
                last_known_good = _LAST_EXPANDED_CONFIG_BY_PATH.get(path_key)
                await _warn_config_parse_failure_from_event_loop(
                    config_path,
                    exc,
                    fallback=(
                        "last-known-good" if last_known_good is not None else "defaults"
                    ),
                )
                if last_known_good is not None:
                    retained = _expand_env_vars(copy.deepcopy(last_known_good))
                    if cache_sig is not None:
                        _LOAD_CONFIG_CACHE[path_key] = (*cache_sig, retained, {})
                    return retained

        normalized = _normalize_root_model_keys(_normalize_max_turns_config(config))
        expanded = _expand_env_vars(normalized)
        managed_config: Dict[str, Any] = {}
        if managed_path is not None and managed_sig is not None:
            try:
                parsed_managed_config = await _read_yaml(managed_path)
                if isinstance(parsed_managed_config, dict):
                    managed_config = parsed_managed_config
            except Exception as exc:
                logger.warning(
                    "managed scope: failed to parse %s: %s — IGNORING this managed "
                    "file. Admin policy from this file is NOT being applied. Fix "
                    "and restart.",
                    managed_path,
                    exc,
                )
        if managed_config:
            managed_expanded = _expand_env_vars(managed_config)
            expanded = _deep_merge(expanded, managed_expanded)

        _LAST_EXPANDED_CONFIG_BY_PATH[path_key] = copy.deepcopy(expanded)
        if cache_sig is not None:
            cached_copy = copy.deepcopy(expanded)
            env_snapshot = _env_ref_snapshot(normalized)
            if managed_config:
                _env_ref_snapshot(managed_config, env_snapshot)
            _LOAD_CONFIG_CACHE[path_key] = (*cache_sig, cached_copy, env_snapshot)
            return cached_copy

        _LOAD_CONFIG_CACHE.pop(path_key, None)
        return expanded


TERMINAL_CONFIG_ENV_MAP = {
    "cwd": "TERMINAL_CWD",
    "timeout": "TERMINAL_TIMEOUT",
    "local_persistent": "TERMINAL_LOCAL_PERSISTENT",
}


def _terminal_env_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def terminal_config_env_var_for_key(key: str) -> Optional[str]:
    """Return the env var mirrored by a ``terminal.*`` config key."""
    prefix = "terminal."
    if not key.startswith(prefix):
        return None
    return TERMINAL_CONFIG_ENV_MAP.get(key[len(prefix):])


def apply_terminal_config_to_env(
    *,
    env: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    override: Optional[bool] = None,
) -> Dict[str, str]:
    """Bridge ``terminal.*`` config into the env vars terminal tools read.

    ``tools.terminal_tool`` is intentionally environment-driven because it also
    runs in child processes (TUI, dashboard PTY, gateway workers).  This helper
    gives those child-process launch paths the same config bridge as classic
    CLI without importing ``cli.py`` and paying for its startup side effects.

    When the user config contains a ``terminal`` section, config.yaml is
    authoritative and overrides existing env values.  Otherwise defaults only
    backfill missing env vars so exported/.env values keep working.
    """
    target = os.environ if env is None else env

    raw_config = read_raw_config()
    file_has_terminal_config = isinstance(raw_config.get("terminal"), dict)
    should_override = file_has_terminal_config if override is None else override

    cfg = config if config is not None else load_config()
    terminal_cfg = cfg.get("terminal", {}) if isinstance(cfg, dict) else {}
    if not isinstance(terminal_cfg, dict):
        return target

    for cfg_key, env_var in TERMINAL_CONFIG_ENV_MAP.items():
        if cfg_key not in terminal_cfg:
            continue
        value = terminal_cfg[cfg_key]
        if cfg_key == "cwd":
            raw_cwd = str(value or "").strip()
            if raw_cwd in {".", "auto", "cwd"}:
                continue
            if isinstance(value, str):
                value = os.path.expanduser(value)
        if should_override or env_var not in target:
            target[env_var] = _terminal_env_value(value)
    return target


def _load_config_impl(*, want_deepcopy: bool) -> Dict[str, Any]:
    with _CONFIG_LOCK:
        ensure_hermes_home()
        config_path = get_config_path()
        path_key = str(config_path)

        try:
            st = config_path.stat()
            user_sig: Optional[Tuple[int, int]] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            user_sig = None

        # Managed scope: fold the managed config file's (mtime, size) into the
        # cache signature so editing /etc/hermes/config.yaml invalidates the
        # cached merged result. (0, 0) means "no managed config file".
        from hermes_cli import managed_scope

        managed_dir = managed_scope.get_managed_dir()
        managed_cfg_path = (managed_dir / "config.yaml") if managed_dir else None
        try:
            mst = managed_cfg_path.stat() if managed_cfg_path else None
            managed_sig = (mst.st_mtime_ns, mst.st_size) if mst else (0, 0)
        except OSError:
            managed_sig = (0, 0)

        # Combined cache signature: user file + managed file. None only when the
        # user config is absent AND no managed file exists (nothing to cache on).
        if user_sig is not None:
            cache_sig: Optional[Tuple[int, int, int, int]] = (
                user_sig[0],
                user_sig[1],
                managed_sig[0],
                managed_sig[1],
            )
        elif managed_sig != (0, 0):
            cache_sig = (0, 0, managed_sig[0], managed_sig[1])
        else:
            cache_sig = None

        cached = _LOAD_CONFIG_CACHE.get(path_key)
        if cached is not None and cache_sig is not None and cached[:4] == cache_sig:
            # File signatures match, but the cached expansion is only valid if
            # every ${VAR} it was expanded against still has the same value.
            # Without this, a load_config() that ran before load_hermes_dotenv()
            # pins unexpanded literals (e.g. auxiliary.<task>.api_key) for the
            # life of the process (#58514).
            env_snapshot = cached[5] if len(cached) > 5 else {}
            if all(os.environ.get(k) == v for k, v in env_snapshot.items()):
                return copy.deepcopy(cached[4]) if want_deepcopy else cached[4]

        config = copy.deepcopy(DEFAULT_CONFIG)

        if user_sig is not None:
            try:
                with open(config_path, encoding="utf-8") as f:
                    user_config = fast_safe_load(f) or {}

                if "max_turns" in user_config:
                    agent_user_config = dict(user_config.get("agent") or {})
                    if agent_user_config.get("max_turns") is None:
                        agent_user_config["max_turns"] = user_config["max_turns"]
                    user_config["agent"] = agent_user_config
                    user_config.pop("max_turns", None)

                config = _deep_merge(config, user_config)
            except Exception as e:
                # Last-known-good fallback (port of openai/codex#31188's
                # invariant: a parse failure in a policy/config file must not
                # silently replace the effective policy with an empty/default
                # one). Falling through to DEFAULT_CONFIG here drops EVERY user
                # override — including security-critical ``approvals.deny``
                # rules, which are supposed to block commands even under yolo.
                # A long-running gateway whose user mid-edits config.yaml into
                # broken YAML would silently lose those rules on the next load.
                # Within a running process we still have the last successfully
                # loaded config — keep serving it until the file is fixed.
                # Fresh processes with no last-known-good keep the existing
                # DEFAULT_CONFIG fallback.
                lkg = _LAST_EXPANDED_CONFIG_BY_PATH.get(path_key)
                _warn_config_parse_failure(
                    config_path,
                    e,
                    fallback="last-known-good" if lkg is not None else "defaults",
                )
                if lkg is not None:
                    # save_config() stores the pre-expansion normalized dict
                    # (env-ref templates preserved); the load path stores the
                    # expanded one. Expand defensively — idempotent when the
                    # stored value is already expanded.
                    from typing import cast as _cast
                    lkg_copy: Dict[str, Any] = _cast(
                        Dict[str, Any], _expand_env_vars(copy.deepcopy(lkg))
                    )
                    if cache_sig is not None:
                        # Cache under the corrupt file's signature (empty env
                        # snapshot: always valid) so repeated loads don't
                        # re-parse the broken file; fixing the file changes the
                        # signature and triggers a normal reload.
                        _empty_env: Dict[str, Optional[str]] = {}
                        _LOAD_CONFIG_CACHE[path_key] = (
                            cache_sig[0], cache_sig[1],
                            cache_sig[2], cache_sig[3],
                            lkg_copy, _empty_env,
                        )
                    return copy.deepcopy(lkg_copy) if want_deepcopy else lkg_copy

        normalized = _normalize_root_model_keys(_normalize_max_turns_config(config))
        expanded = _expand_env_vars(normalized)
        # Managed scope wins at the leaf. Applied AFTER user expansion so a user
        # ${VAR} cannot shadow a managed literal: managed values are expanded only
        # against the process environment, never against user-config-defined refs.
        # This deliberately inverts the usual env-over-config precedence for the
        # keys the managed layer pins — see docs/design/managed-scope.md §4.1.
        managed_config = managed_scope.load_managed_config()
        if managed_config:
            managed_expanded = _expand_env_vars(managed_config)
            expanded = _deep_merge(expanded, managed_expanded)
        _LAST_EXPANDED_CONFIG_BY_PATH[path_key] = copy.deepcopy(expanded)
        if cache_sig is not None:
            # Cache stores a separate deepcopy so subsequent ``load_config()``
            # (deepcopy=True) callers can mutate freely without affecting the
            # cached value, and ``load_config_readonly()`` (deepcopy=False)
            # callers all see the same stable cached object. The cached tuple is
            # (user_mtime, user_size, managed_mtime, managed_size, value,
            # env_ref_snapshot). The snapshot records the environment values
            # this expansion was made against so later loads can detect env
            # drift (late .env load, in-process rotation) — see cache hit above.
            cached_copy = copy.deepcopy(expanded)
            env_snapshot = _env_ref_snapshot(normalized)
            if managed_config:
                _env_ref_snapshot(managed_config, env_snapshot)
            _LOAD_CONFIG_CACHE[path_key] = (*cache_sig, cached_copy, env_snapshot)
            # On the readonly path return the same cached object subsequent
            # calls will see — keeps "two readonly calls return the same
            # object" invariant that callers may rely on for identity checks.
            if not want_deepcopy:
                return cached_copy
        else:
            _LOAD_CONFIG_CACHE.pop(path_key, None)
        # First-load result is a fresh dict (not aliased to the cache); safe
        # to return directly. For the deepcopy=True path this is the
        # canonical "freshly-built mutable result" the function has always
        # returned. For the deepcopy=False path with no cache (e.g. config
        # file missing), it's also fine — callers get an isolated object.
        return expanded


_SECURITY_COMMENT = """
# ── Security ──────────────────────────────────────────────────────────
# Secret redaction is ON by default — strings that look like API keys,
# tokens, and passwords are masked in tool output, logs, and chat
# responses before the model or user ever sees them. Set redact_secrets
# to false to disable (e.g. when developing the redactor itself).
#
# security:
#   redact_secrets: true
"""

_FALLBACK_COMMENT = """
# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#   bedrock      (AWS IAM / boto3)     — AWS Bedrock (Converse API)
#
# For custom OpenAI-compatible endpoints, add base_url and key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
"""


_COMMENTED_SECTIONS = """
# ── Security ──────────────────────────────────────────────────────────
# Secret redaction is ON by default. Set to false to pass tool output,
# logs, and chat responses through unmodified (e.g. for redactor dev).
#
# security:
#   redact_secrets: true

# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#   bedrock      (AWS IAM / boto3)     — AWS Bedrock (Converse API)
#
# For custom OpenAI-compatible endpoints, add base_url and key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
"""


def save_config(
    config: Dict[str, Any],
    *,
    strip_defaults: bool = True,
    preserve_keys: Optional[Set[Tuple[str, ...]]] = None,
    merge_existing: bool = False,
):
    """Save configuration to ~/.hermes/config.yaml.\n

    Default values from ``DEFAULT_CONFIG`` are not written to disk unless
    the user explicitly set them (i.e. the path exists in the raw config
    before any normalisation).  This prevents config.yaml from being
    contaminated with schema defaults on every save, which makes future
    default changes invisible to users.

    When ``merge_existing`` is True, the on-disk raw config is deep-merged
    under *config* before writing so partial callers (migration steps via
    ``_persist_migration``) cannot drop unrelated sections the caller omitted.
    Full-document replacement callers (dashboard raw YAML editor, callers that
    already deep-merge) must leave this False so intentional deletions survive.
    """
    with _CONFIG_LOCK:
        if is_managed():
            managed_error("save configuration")
            return
        # Managed scope: strip any leaf the managed layer pins, so a bulk write
        # (wizard / programmatic save) never persists a user value that would
        # silently lose to managed on the next load. Single-key `config set`
        # hard-rejects (see set_config_value); this is the mechanical safety net
        # for bulk writes so the unmanaged remainder still lands.
        from hermes_cli import managed_scope

        managed_keys = managed_scope.managed_config_keys()
        if managed_keys:
            config, _stripped = _strip_dotted_keys(copy.deepcopy(config), managed_keys)
            if _stripped:
                print(
                    f"Note: {len(_stripped)} managed setting(s) were not saved "
                    f"(managed by your administrator): {', '.join(sorted(_stripped))}",
                    file=sys.stderr,
                )
        from utils import atomic_yaml_write

        ensure_hermes_home()
        config_path = get_config_path()
        require_readable_config_before_write(config_path)
        # Compute explicit user paths BEFORE any normalisation --------
        # _normalize_max_turns_config may inject agent.max_turns from
        # DEFAULT_CONFIG; using the raw dict preserves which paths the
        # user actually set so _strip_default_values can keep them.
        _raw_for_paths = read_raw_config()
        explicit_raw_paths: Optional[Set[Tuple[str, ...]]] = (
            _explicit_config_paths(_raw_for_paths) if _raw_for_paths else None
        )
        if merge_existing and _raw_for_paths:
            config = _merge_partial_save(_raw_for_paths, config)
        # ----------------------------------------------------------------

        current_normalized = _normalize_root_model_keys(_normalize_max_turns_config(config))
        normalized = current_normalized
        raw_existing = (
            _normalize_root_model_keys(_normalize_max_turns_config(_raw_for_paths))
            if _raw_for_paths
            else {}
        )
        if raw_existing:
            normalized = _preserve_env_ref_templates(
                normalized,
                raw_existing,
                _LAST_EXPANDED_CONFIG_BY_PATH.get(str(config_path)),
            )

        # Strip schema-default values so the user's custom settings are not
        # silently reset on every save.  Keys the user explicitly set (paths
        # from the raw pre-normalisation config) are always preserved.
        effective_preserve_keys: Set[Tuple[str, ...]] = {("_config_version",)}
        if explicit_raw_paths:
            effective_preserve_keys.update(explicit_raw_paths)
        if preserve_keys:
            effective_preserve_keys.update(preserve_keys)

        if strip_defaults and effective_preserve_keys:
            # _preserve_env_ref_templates may return Any; cast for type-checker.
            from typing import cast as _cast
            normalized = _cast(Dict[str, Any], normalized)
            normalized = _strip_default_values(
                normalized,  # type: ignore[arg-type]
                DEFAULT_CONFIG,
                preserve_keys=effective_preserve_keys,
            )

        # Build optional commented-out sections for features that are off by
        # default or only relevant when explicitly configured.
        parts = []
        sec = normalized.get("security", {})
        if not sec or sec.get("redact_secrets") is None:
            parts.append(_SECURITY_COMMENT)
        fb = normalized.get("fallback_model", {})
        fb_is_valid = False
        if isinstance(fb, list):
            fb_is_valid = any(isinstance(e, dict) and e.get("provider") and e.get("model") for e in fb)
        elif isinstance(fb, dict):
            fb_is_valid = bool(fb.get("provider") and fb.get("model"))
        if not fb_is_valid:
            parts.append(_FALLBACK_COMMENT)

        atomic_yaml_write(
            config_path,
            normalized,
            extra_content="".join(parts) if parts else None,
        )
        _secure_file(config_path)
        _RAW_CONFIG_CACHE.pop(str(config_path), None)
        _LAST_EXPANDED_CONFIG_BY_PATH[str(config_path)] = copy.deepcopy(current_normalized)


def _parse_env_value(raw_value: str) -> str:
    """Parse the small .env value subset Hermes writes itself."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        quoted = value[1:-1]
        parsed: list[str] = []
        i = 0
        while i < len(quoted):
            ch = quoted[i]
            if ch == "\\" and i + 1 < len(quoted):
                next_ch = quoted[i + 1]
                if next_ch in {'"', "\\"}:
                    parsed.append(next_ch)
                    i += 2
                    continue
            parsed.append(ch)
            i += 1
        return "".join(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def load_env() -> Dict[str, str]:
    """Load environment variables from ~/.hermes/.env.

    Normalizes line endings before parsing while treating each assignment's
    value as opaque data for boundary discovery.

    The parsed dict is memoised keyed on the .env file mtime, because
    ``get_env_value()`` is called dozens-to-hundreds of times per
    interactive menu render (`hermes tools`, `hermes setup`, status
    panels). Sanitisation is O(lines), so re-parsing the
    same file on every call was burning ~300ms of CPU per `hermes tools`
    menu paint on top of the OAuth-refresh slowness. The mtime check
    invalidates the cache when the user edits .env mid-process.
    """
    global _env_cache
    env_path = get_env_path()

    try:
        mtime = env_path.stat().st_mtime
        size = env_path.stat().st_size
        cache_key = (str(env_path), mtime, size)
    except FileNotFoundError:
        cache_key = (str(env_path), None, None)
    except Exception:
        cache_key = None

    if cache_key is not None and _env_cache is not None:
        cached_key, cached_vars = _env_cache
        if cached_key == cache_key:
            return dict(cached_vars)

    env_vars: Dict[str, str] = {}

    if env_path.exists():
        # On Windows, open() defaults to the system locale (cp1252) which can
        # fail on UTF-8 .env files. Always use explicit UTF-8; tolerate BOM
        # via utf-8-sig since users may edit .env in Notepad which adds one.
        open_kw = {"encoding": "utf-8-sig", "errors": "replace"}
        with open(env_path, **open_kw) as f:
            raw_lines = f.readlines()
        # Normalize line endings without interpreting value contents as syntax.
        lines = _sanitize_env_lines(raw_lines)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                # Strip the bash-compatible ``export `` prefix so lines like
                # ``export API_KEY=...`` parse as ``API_KEY`` rather than being
                # stored under the wrong key ``"export API_KEY"`` (#6659).
                if line.startswith('export '):
                    line = line[7:]
                key, _, value = line.partition('=')
                env_vars[key.strip()] = _parse_env_value(value)

    if cache_key is not None:
        _env_cache = (cache_key, dict(env_vars))

    return env_vars


# Module-level memo for load_env(), keyed on (path, mtime, size).
# Editing .env bumps mtime → next load_env() rebuilds. invalidate_env_cache()
# is the explicit knob for writers that update .env via this module
# (set_env_value, save_env, etc.) without relying on filesystem mtime
# resolution.
_env_cache: Optional[Tuple[Tuple[str, Optional[float], Optional[int]], Dict[str, str]]] = None


def invalidate_env_cache() -> None:
    """Clear the load_env() process-level memo.

    Writers that mutate .env (set_env_value, save_env, etc.) call this
    to guarantee the next load_env() sees their change even on
    filesystems with coarse mtime resolution. Reads invalidate naturally
    via the mtime/size check.
    """
    global _env_cache
    _env_cache = None


def _sanitize_env_lines(lines: list) -> list:
    """Normalize .env line endings without changing assignment semantics.

    Content after the first ``=`` is opaque value data. A known variable name
    embedded in that value must never be reinterpreted as another assignment;
    concatenated assignments are ambiguous and therefore remain on one line.
    """
    sanitized: list[str] = []
    for line in lines:
        raw = line.rstrip("\r\n")
        stripped = raw.strip()

        # Preserve blank lines and comments
        if not stripped or stripped.startswith("#"):
            sanitized.append(raw + "\n")
            continue

        sanitized.append(stripped + "\n")

    return sanitized


def sanitize_env_file() -> int:
    """Read, sanitize, and rewrite ~/.hermes/.env in place.

    Returns the number of lines whose safe formatting was normalized. Returns
    0 when no changes are needed.
    """
    env_path = get_env_path()
    if not env_path.exists():
        return 0

    read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
    write_kw = {"encoding": "utf-8"}

    with open(env_path, **read_kw) as f:
        original_lines = f.readlines()

    sanitized = _sanitize_env_lines(original_lines)

    if sanitized == original_lines:
        return 0

    # Count lines whose normalized representation differs.
    fixes = abs(len(sanitized) - len(original_lines))
    if fixes == 0:
        fixes = sum(1 for a, b in zip(original_lines, sanitized) if a != b)
        fixes += abs(len(sanitized) - len(original_lines))

    fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix=".tmp", prefix=".env_")
    try:
        with os.fdopen(fd, "w", **write_kw) as f:
            f.writelines(sanitized)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, env_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _secure_file(env_path)
    invalidate_env_cache()
    return fixes


def _check_non_ascii_credential(key: str, value: str) -> str:
    """Warn and strip non-ASCII characters from credential values.

    API keys and tokens must be pure ASCII — they are sent as HTTP header
    values which httpx/httpcore encode as ASCII.  Non-ASCII characters
    (commonly introduced by copy-pasting from rich-text editors or PDFs
    that substitute lookalike Unicode glyphs for ASCII letters) cause
    ``UnicodeEncodeError: 'ascii' codec can't encode character`` at
    request time.

    Returns the sanitized (ASCII-only) value.  Prints a warning if any
    non-ASCII characters were found and removed.
    """
    try:
        value.encode("ascii")
        return value  # all ASCII — nothing to do
    except UnicodeEncodeError:
        pass

    # Build a readable list of the offending characters
    bad_chars: list[str] = []
    for i, ch in enumerate(value):
        if ord(ch) > 127:
            bad_chars.append(f"  position {i}: {ch!r} (U+{ord(ch):04X})")
    sanitized = value.encode("ascii", errors="ignore").decode("ascii")

    print(
        f"\n  Warning: {key} contains non-ASCII characters that will break API requests.\n"
        f"  This usually happens when copy-pasting from a PDF, rich-text editor,\n"
        f"  or web page that substitutes lookalike Unicode glyphs for ASCII letters.\n"
        f"\n"
        + "\n".join(f"  {line}" for line in bad_chars[:5])
        + ("\n  ... and more" if len(bad_chars) > 5 else "")
        + "\n\n  The non-ASCII characters have been stripped automatically.\n"
        "  If authentication fails, re-copy the key from the provider's dashboard.\n",
        file=sys.stderr,
    )
    return sanitized


def _quote_env_value(value: str) -> str:
    """Quote .env values containing characters with special dotenv meaning."""
    if value == "":
        return value
    # Internal whitespace (space/tab/etc.) must be quoted so shell `set -a; . file`
    # word-splits don't break paths like macOS "Application Support". Leading/
    # trailing whitespace is already covered by the strip check; any() covers
    # internal runs that strip() would leave alone.
    needs_quoting = (
        "#" in value
        or '"' in value
        or "'" in value
        or value != value.strip()
        or any(c.isspace() for c in value)
    )
    if not needs_quoting:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _env_line_defines_key(line: str, key: str) -> bool:
    """True when a .env line assigns ``key`` — plain or ``export``-prefixed.

    ``load_env()`` accepts the bash-compatible ``export KEY=value`` form
    (#6659), so the writers must recognise the same shape. Otherwise a
    hand-added ``export`` line is invisible to save (duplicate appended) and
    remove (line survives → the value resurrects on the next load, #40041).
    """
    stripped = line.strip()
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    return stripped.startswith(f"{key}=")


def save_env_value(key: str, value: str):
    """Save or update a value in ~/.hermes/.env."""
    if is_managed():
        managed_error(f"set {key}")
        return
    # Managed scope guard: a managed env key can't be set by the user — the
    # managed .env wins at load anyway. Distinct from is_managed() above.
    from hermes_cli import managed_scope

    if managed_scope.is_env_managed(key):
        managed_dir = managed_scope.get_managed_dir()
        src = (managed_dir / ".env") if managed_dir else "the managed scope"
        print(
            f"Cannot set {key}: it is managed by your administrator ({src}) "
            f"and cannot be changed.",
            file=sys.stderr,
        )
        return
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    _reject_denylisted_env_var(key)
    value = value.replace("\n", "").replace("\r", "")
    # API keys / tokens must be ASCII — strip non-ASCII with a warning.
    value = _check_non_ascii_credential(key, value)
    ensure_hermes_home()
    env_path = get_env_path()

    # On Windows, open() defaults to the system locale (cp1252) which can
    # cause OSError errno 22 on UTF-8 .env files.
    read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
    write_kw = {"encoding": "utf-8"}

    lines = []
    if env_path.exists():
        with open(env_path, **read_kw) as f:
            lines = f.readlines()
        # Normalize safe line formatting without interpreting values as syntax.
        lines = _sanitize_env_lines(lines)

    serialized_value = _quote_env_value(value)

    # Find and update or append. Match both ``KEY=`` and the bash-compatible
    # ``export KEY=`` form — load_env() parses export lines (#6659), so a
    # user-added ``export GITHUB_TOKEN=...`` shows as set in every UI. If the
    # writer didn't match it, a save would append a SECOND line and a later
    # delete of that line would silently resurrect the old exported value
    # (#40041: "token detected but cannot be replaced through the UI").
    found = False
    for i, line in enumerate(lines):
        if _env_line_defines_key(line, key):
            lines[i] = f"{key}={serialized_value}\n"
            found = True
            break

    if not found:
        # Ensure there's a newline at the end of the file before appending
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={serialized_value}\n")
    
    fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix='.tmp', prefix='.env_')
    # Preserve original permissions so Docker volume mounts aren't clobbered.
    original_mode = None
    if env_path.exists():
        try:
            original_mode = stat.S_IMODE(env_path.stat().st_mode)
        except OSError:
            pass
    try:
        with os.fdopen(fd, 'w', **write_kw) as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, env_path)
        # Preserve the original file mode (e.g. 0640 for Docker volume mounts)
        # instead of letting _secure_file unconditionally tighten to 0600.
        if original_mode is not None:
            try:
                os.chmod(env_path, original_mode)
            except OSError:
                pass
        else:
            _secure_file(env_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    os.environ[key] = value
    invalidate_env_cache()


def custom_endpoint_key_env(identity: str) -> str:
    """Env var name holding a custom endpoint's API key.

    ``identity`` is whatever names the endpoint on the calling path — the
    Desktop panel's endpoint id, or ``host:port`` for the CLI setup flow.
    Two properties matter:

    - It keys off the endpoint's own identity, not just its hostname, so two
      endpoints on one host (``127.0.0.1:8000`` and ``:8001``) get separate
      slots instead of the second save clobbering the first's credential.
    - The fixed ``HERMES_CUSTOM_`` prefix keeps the result a valid POSIX name
      even when the slug starts with a digit, which every IP-based local
      endpoint does (``127.0.0.1`` → ``127_0_0_1``). ``save_env_value``
      rejects digit-leading names outright.
    """
    slug = re.sub(r"[^A-Z0-9]+", "_", str(identity or "").upper()).strip("_")
    return f"HERMES_CUSTOM_{slug}_API_KEY" if slug else "HERMES_CUSTOM_API_KEY"


def remove_env_value(key: str) -> bool:
    """Remove a key from ~/.hermes/.env and os.environ.

    Returns True if the key was found and removed, False otherwise.
    """
    if is_managed():
        managed_error(f"remove {key}")
        return False
    # Managed scope guard: a managed env key can't be removed by the user.
    from hermes_cli import managed_scope

    if managed_scope.is_env_managed(key):
        managed_dir = managed_scope.get_managed_dir()
        src = (managed_dir / ".env") if managed_dir else "the managed scope"
        print(
            f"Cannot remove {key}: it is managed by your administrator ({src}) "
            f"and cannot be changed.",
            file=sys.stderr,
        )
        return False
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    env_path = get_env_path()
    if not env_path.exists():
        os.environ.pop(key, None)
        return False

    read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
    write_kw = {"encoding": "utf-8"}

    with open(env_path, **read_kw) as f:
        lines = f.readlines()
    lines = _sanitize_env_lines(lines)

    new_lines = [line for line in lines if not _env_line_defines_key(line, key)]
    found = len(new_lines) < len(lines)

    if found:
        fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix='.tmp', prefix='.env_')
        # Preserve original permissions so Docker volume mounts aren't clobbered.
        original_mode = None
        try:
            original_mode = stat.S_IMODE(env_path.stat().st_mode)
        except OSError:
            pass
        try:
            with os.fdopen(fd, 'w', **write_kw) as f:
                f.writelines(new_lines)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, env_path)
            # Preserve the original file mode (e.g. 0640 for Docker volume
            # mounts) instead of letting _secure_file unconditionally tighten
            # to 0600. Mirrors save_env_value().
            if original_mode is not None:
                try:
                    os.chmod(env_path, original_mode)
                except OSError:
                    pass
            else:
                _secure_file(env_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    os.environ.pop(key, None)
    invalidate_env_cache()
    return found


def save_anthropic_oauth_token(value: str, save_fn=None):
    """Persist an Anthropic OAuth/setup token and clear the API-key slot."""
    writer = save_fn or save_env_value
    writer("ANTHROPIC_TOKEN", value)
    writer("ANTHROPIC_API_KEY", "")


def use_anthropic_claude_code_credentials(save_fn=None):
    """Use Claude Code's own credential files instead of persisting env tokens."""
    writer = save_fn or save_env_value
    writer("ANTHROPIC_TOKEN", "")
    writer("ANTHROPIC_API_KEY", "")


def save_anthropic_api_key(value: str, save_fn=None):
    """Persist an Anthropic API key and clear the OAuth/setup-token slot."""
    writer = save_fn or save_env_value
    writer("ANTHROPIC_API_KEY", value)
    writer("ANTHROPIC_TOKEN", "")


def reload_env() -> int:
    """Re-read ~/.hermes/.env into os.environ. Returns count of vars updated.

    Adds/updates vars that changed and removes vars that were deleted from
    the .env file (but only vars known to Hermes — OPTIONAL_ENV_VARS and
    _EXTRA_ENV_KEYS — to avoid clobbering unrelated environment).
    """
    env_vars = load_env()
    known_keys = set(OPTIONAL_ENV_VARS.keys()) | _EXTRA_ENV_KEYS
    count = 0
    for key, value in env_vars.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            count += 1
    # Remove known Hermes vars that are no longer in .env
    for key in known_keys:
        if key not in env_vars and key in os.environ:
            del os.environ[key]
            count += 1
    return count


def get_env_value(key: str) -> Optional[str]:
    """Read a credential without escaping the active profile scope."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except Exception:
        value = os.environ.get(key)
    else:
        try:
            value = get_secret(key)
        except UnscopedSecretError:
            raise
        except Exception:
            value = os.environ.get(key)
    if value is not None:
        return value

    env_vars = load_env()
    return env_vars.get(key)


async def get_env_value_prefer_dotenv(key: str) -> Optional[str]:
    """Async credential lookup for agent/provider execution paths.

    The dotenv file is read through ``aiofiles`` before falling back to the
    profile-scoped process environment.
    """
    try:
        import aiofiles

        env_path = get_env_path()
        env_vars: Dict[str, str] = {}
        try:
            async with aiofiles.open(
                env_path, "r", encoding="utf-8-sig", errors="replace"
            ) as handle:
                raw_lines = await handle.readlines()
        except FileNotFoundError:
            raw_lines = []
        for line in _sanitize_env_lines(raw_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:]
            env_key, raw_value = stripped.split("=", 1)
            env_key = env_key.strip()
            if env_key:
                env_vars[env_key] = _parse_env_value(raw_value)
        value = env_vars.get(key)
        if value:
            return value
    except Exception:
        # A credential read should remain best-effort; the scoped environment
        # fallback below preserves the same behavior as the sync helper.
        pass

    try:
        from agent.secret_scope import (
            UnscopedSecretError,
            get_secret as _get_secret,
        )
    except Exception:
        return os.environ.get(key)
    try:
        return _get_secret(key)
    except UnscopedSecretError:
        raise
    except Exception:
        return os.environ.get(key)


# ── Profile-driven env var injection ─────────────────────────────────────────
# Any provider registered in providers/ with auth_type="api_key" automatically
# gets its env_vars exposed in OPTIONAL_ENV_VARS without editing this file.
# Runs once at import time.

_profile_env_vars_injected = False


def _inject_profile_env_vars() -> None:
    """Populate OPTIONAL_ENV_VARS from provider profiles not already listed.

    Called once at module load time. Idempotent — repeated calls are no-ops.
    """
    global _profile_env_vars_injected
    if _profile_env_vars_injected:
        return
    try:
        from providers import _discovered, _list_providers_cached

        if not _discovered:
            return
        for _pp in _list_providers_cached():
            if _pp.auth_type not in {"api_key",}:
                continue
            for _var in _pp.env_vars:
                if _var in OPTIONAL_ENV_VARS:
                    continue
                _is_key = not _var.endswith("_BASE_URL") and not _var.endswith("_URL")
                OPTIONAL_ENV_VARS[_var] = {
                    "description": f"{_pp.display_name or _pp.name} {'API key' if _is_key else 'base URL override'}",
                    "prompt": f"{_pp.display_name or _pp.name} {'API key' if _is_key else 'base URL (leave empty for default)'}",
                    "url": _pp.signup_url or None,
                    "password": _is_key,
                    "category": "provider",
                    "advanced": True,
                }
        _profile_env_vars_injected = True
    except Exception:
        # Provider discovery is an awaited runtime boundary. Leave the
        # sentinel unset so deferred runtime initialization can retry after
        # discovery completes.
        return


# Populate immediately for synchronous CLI/config callers.  Async agent
# construction may defer this until provider discovery has crossed its
# awaited boundary.
_inject_profile_env_vars()
