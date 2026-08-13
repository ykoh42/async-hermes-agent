"""File passthrough registry for remote terminal backends.

Remote backends (Docker, Modal, SSH) create sandboxes with no host files.
This module ensures that credential files, skill directories, and host-side
cache directories (documents, images, audio, screenshots) are mounted or
synced into those sandboxes so the agent can access them.

Credential registration and directory discovery perform native awaited file
I/O.  Context-local registry operations and cache-path translations remain
small synchronous transformations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import stat
import threading
import uuid
import weakref
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import aiofiles.os

from hermes_cli.config import cfg_get

try:  # pragma: no cover - exercised via the fail-closed test below
    from agent.file_safety import get_read_block_error
except ImportError:  # noqa: F401 - sentinel consumed in register_credential_file
    get_read_block_error = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_realpath = aiofiles.os.wrap(os.path.realpath)
_chmod = aiofiles.os.wrap(os.chmod)
_utime = aiofiles.os.wrap(os.utime)


# Session-scoped list of credential files to mount. ContextVar prevents one
# concurrently served profile/session from seeing another session's entries.
_registered_files_var: ContextVar[Dict[str, str]] = ContextVar("_registered_files")


def _get_registered() -> Dict[str, str]:
    """Get or create this context's registered credential-file mapping."""
    try:
        return _registered_files_var.get()
    except LookupError:
        value: Dict[str, str] = {}
        _registered_files_var.set(value)
        return value


# Config entries retain upstream's once-loaded cache semantics, scoped to the
# canonical Hermes profile.  A process may serve multiple profiles on one
# event loop, so a single process-global list would mount profile A's
# credentials into profile B's sandbox.
_config_files: List[Dict[str, str]] | None = None
_config_files_by_home: dict[str, List[Dict[str, str]]] = {}
_config_files_cache_guard = threading.RLock()
_config_file_locks_guard = threading.RLock()
_config_file_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()


def _resolve_hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


async def _config_profile_key() -> str:
    """Return the filesystem-canonical active profile cache key."""
    return os.path.normcase(str(await _realpath(_resolve_hermes_home())))


def _config_file_lock(profile_key: str) -> asyncio.Lock:
    """Return a live loop-local lock without retaining a closed event loop."""
    loop = asyncio.get_running_loop()
    with _config_file_locks_guard:
        for candidate in tuple(_config_file_locks):
            if candidate.is_closed():
                _config_file_locks.pop(candidate, None)
        locks = _config_file_locks.setdefault(loop, {})
        lock_ref = locks.get(profile_key)
        lock = lock_ref() if lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            locks[profile_key] = weakref.ref(lock)
        return lock


def _cached_config_files(profile_key: str) -> List[Dict[str, str]] | None:
    """Return this profile's cache while honoring the legacy reset hook."""
    global _config_files
    with _config_files_cache_guard:
        # Existing callers and tests reset the private upstream cache by
        # assigning ``_config_files = None``. Preserve that behavior without
        # making the compatibility snapshot authoritative across profiles.
        if _config_files is None and _config_files_by_home:
            _config_files_by_home.clear()
        cached = _config_files_by_home.get(profile_key)
        if cached is not None:
            _config_files = cached
        return cached


def _store_config_files(
    profile_key: str,
    result: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    global _config_files
    with _config_files_cache_guard:
        cached = _config_files_by_home.setdefault(profile_key, result)
        _config_files = cached
        return cached


async def _resolve_within(path: Path, root: Path) -> tuple[Path | None, str | None]:
    """Resolve symlinks and return a contained path, or a validation error."""
    try:
        resolved = Path(await _realpath(path))
        resolved_root = Path(await _realpath(root))
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        return None, f"Path escapes allowed directory: {exc}"
    return resolved, None


async def register_credential_file(
    relative_path: str,
    container_base: str = "/root/.hermes",
) -> bool:
    """Register a credential file for mounting into remote sandboxes.

    ``relative_path`` is relative to ``HERMES_HOME``. Absolute paths, paths
    whose resolved target escapes that home, and files denied by the canonical
    read guard are rejected. The return value matches upstream: ``True`` only
    when an existing regular file was registered.
    """
    hermes_home = _resolve_hermes_home()

    if os.path.isabs(relative_path):
        logger.warning(
            "credential_files: rejected absolute path %r (must be relative to HERMES_HOME)",
            relative_path,
        )
        return False

    resolved, containment_error = await _resolve_within(
        hermes_home / relative_path,
        hermes_home,
    )
    if containment_error or resolved is None:
        logger.warning(
            "credential_files: rejected path traversal %r (%s)",
            relative_path,
            containment_error,
        )
        return False
    if not await aiofiles.os.path.isfile(resolved):
        logger.debug("credential_files: skipping %s (not found)", resolved)
        return False

    # Containment is insufficient because HERMES_HOME itself contains master
    # stores such as .env, auth.json, mcp-tokens/, and the Bitwarden cache.
    if get_read_block_error is None:
        logger.error(
            "credential_files: refusing %r — agent.file_safety could not be "
            "imported, so the master-store deny-list cannot be consulted",
            relative_path,
        )
        return False
    try:
        denied = await get_read_block_error(str(resolved))
    except Exception:
        logger.exception(
            "credential_files: refusing %r — read guard raised",
            relative_path,
        )
        return False
    if denied:
        logger.warning(
            "credential_files: refused %r — it is a credential store the agent "
            "is denied from reading; a skill may mount its own service token, "
            "not the master key files",
            relative_path,
        )
        return False

    container_path = f"{container_base.rstrip('/')}/{relative_path}"
    _get_registered()[container_path] = str(resolved)
    logger.debug("credential_files: registered %s -> %s", resolved, container_path)
    return True


async def register_credential_files(
    entries: list,
    container_base: str = "/root/.hermes",
) -> List[str]:
    """Register skill frontmatter entries and return unavailable paths."""
    missing: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            relative_path = entry.strip()
        elif isinstance(entry, dict):
            relative_path = str(entry.get("path") or entry.get("name") or "").strip()
        else:
            continue
        if not relative_path:
            continue
        if not await register_credential_file(relative_path, container_base):
            missing.append(relative_path)
    return missing


async def _load_config_files() -> List[Dict[str, str]]:
    """Load ``terminal.credential_files`` once for the active profile."""
    profile_key = await _config_profile_key()
    cached = _cached_config_files(profile_key)
    if cached is not None:
        return cached

    async with _config_file_lock(profile_key):
        cached = _cached_config_files(profile_key)
        if cached is not None:
            return cached

        result: List[Dict[str, str]] = []
        try:
            from hermes_cli.config import read_user_config_raw

            hermes_home = _resolve_hermes_home()
            config = await read_user_config_raw()
            credential_files = cfg_get(config, "terminal", "credential_files")
            if isinstance(credential_files, list):
                for item in credential_files:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    relative_path = item.strip()
                    if os.path.isabs(relative_path):
                        logger.warning(
                            "credential_files: rejected absolute config path %r",
                            relative_path,
                        )
                        continue
                    resolved, containment_error = await _resolve_within(
                        hermes_home / relative_path,
                        hermes_home,
                    )
                    if containment_error or resolved is None:
                        logger.warning(
                            "credential_files: rejected config path traversal %r (%s)",
                            relative_path,
                            containment_error,
                        )
                        continue
                    if await aiofiles.os.path.isfile(resolved):
                        result.append(
                            {
                                "host_path": str(resolved),
                                "container_path": f"/root/.hermes/{relative_path}",
                            }
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Could not read terminal.credential_files from config: %s",
                exc,
            )

        return _store_config_files(profile_key, result)


async def get_credential_file_mounts() -> List[Dict[str, str]]:
    """Return existing skill-registered and user-configured mount entries."""
    mounts: Dict[str, str] = {}

    for container_path, host_path in _get_registered().items():
        if await aiofiles.os.path.isfile(host_path):
            mounts[container_path] = host_path

    for entry in await _load_config_files():
        container_path = entry["container_path"]
        if container_path not in mounts and await aiofiles.os.path.isfile(
            entry["host_path"]
        ):
            mounts[container_path] = entry["host_path"]

    return [
        {"host_path": host_path, "container_path": container_path}
        for container_path, host_path in mounts.items()
    ]


async def _walk_tree(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Enumerate directories, regular files, and symlinks without following links."""
    directories: list[Path] = []
    files: list[Path] = []
    symlinks: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            names = await aiofiles.os.listdir(directory)
        except OSError:
            continue
        for name in names:
            path = directory / name
            try:
                metadata = await aiofiles.os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                symlinks.append(path)
            elif stat.S_ISDIR(metadata.st_mode):
                directories.append(path)
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(path)
    return directories, files, symlinks


async def _remove_tree(root: Path) -> None:
    """Remove one owned tree using awaited directory and unlink operations."""
    try:
        root_metadata = await aiofiles.os.stat(root, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_metadata.st_mode):
        await aiofiles.os.remove(root)
        return
    directories, files, symlinks = await _walk_tree(root)
    for path in (*files, *symlinks):
        try:
            await aiofiles.os.remove(path)
        except FileNotFoundError:
            pass
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        try:
            await aiofiles.os.rmdir(directory)
        except FileNotFoundError:
            pass
    try:
        await aiofiles.os.rmdir(root)
    except FileNotFoundError:
        pass


async def _copy_file(source: Path, target: Path) -> None:
    """Copy file bytes asynchronously and preserve executable permission bits."""
    await aiofiles.os.makedirs(target.parent, exist_ok=True)
    metadata = await aiofiles.os.stat(source, follow_symlinks=False)
    source_mode = stat.S_IMODE(metadata.st_mode)

    def open_with_source_mode(path: str, flags: int) -> int:
        return os.open(path, flags, source_mode)

    async with aiofiles.open(source, "rb") as source_handle:
        async with aiofiles.open(
            target,
            "wb",
            opener=open_with_source_mode,
        ) as target_handle:
            while chunk := await source_handle.read(64 * 1024):
                await target_handle.write(chunk)
    await _chmod(target, source_mode)
    await _utime(
        target,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        follow_symlinks=False,
    )


async def _make_safe_skills_tempdir() -> Path:
    """Create an owned symlink-sanitization directory with awaited I/O."""
    configured_temp = os.getenv("TMPDIR") or os.getenv("TEMP") or os.getenv("TMP")
    parent = Path(configured_temp) if configured_temp else Path("/tmp")
    if not await aiofiles.os.path.isdir(parent):
        parent = _resolve_hermes_home() / "cache"
        await aiofiles.os.makedirs(parent, exist_ok=True)
    for _ in range(10):
        candidate = parent / f"hermes-skills-safe-{uuid.uuid4().hex}"
        try:
            await aiofiles.os.mkdir(candidate, mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a symlink-safe skills directory")


_safe_skills_tempdir: Path | None = None


async def _safe_skills_path(skills_dir: Path) -> str:
    """Return ``skills_dir`` if symlink-free, else an awaited sanitized copy."""
    global _safe_skills_tempdir

    directories, files, symlinks = await _walk_tree(skills_dir)
    if not symlinks:
        return str(skills_dir)

    for link in symlinks:
        try:
            target = await aiofiles.os.readlink(link)
        except OSError:
            target = "<unreadable>"
        logger.warning(
            "credential_files: skipping symlink in skills dir: %s -> %s",
            link,
            target,
        )

    if _safe_skills_tempdir is not None:
        await _remove_tree(_safe_skills_tempdir)

    safe_dir = await _make_safe_skills_tempdir()
    try:
        for directory in directories:
            target = safe_dir / directory.relative_to(skills_dir)
            await aiofiles.os.makedirs(target, exist_ok=True)
        for source in files:
            await _copy_file(source, safe_dir / source.relative_to(skills_dir))
    except BaseException:
        await _remove_tree(safe_dir)
        raise

    _safe_skills_tempdir = safe_dir
    logger.info("credential_files: created symlink-safe skills copy at %s", safe_dir)
    return str(safe_dir)


async def get_skills_directory_mount(
    container_base: str = "/root/.hermes",
) -> list[Dict[str, str]]:
    """Return local and external skill-directory mount entries."""
    mounts: list[Dict[str, str]] = []
    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if await aiofiles.os.path.isdir(skills_dir):
        mounts.append(
            {
                "host_path": await _safe_skills_path(skills_dir),
                "container_path": f"{container_base.rstrip('/')}/skills",
            }
        )

    try:
        from agent.skill_utils import get_external_skills_dirs

        external_dirs = await get_external_skills_dirs()
    except ImportError:
        external_dirs = []
    for index, external_dir in enumerate(external_dirs):
        if await aiofiles.os.path.isdir(external_dir):
            mounts.append(
                {
                    "host_path": await _safe_skills_path(external_dir),
                    "container_path": (
                        f"{container_base.rstrip('/')}/external_skills/{index}"
                    ),
                }
            )
    return mounts


async def iter_skills_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return regular skill files, excluding every symlink."""
    result: List[Dict[str, str]] = []
    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if await aiofiles.os.path.isdir(skills_dir):
        _directories, files, _symlinks = await _walk_tree(skills_dir)
        container_root = f"{container_base.rstrip('/')}/skills"
        result.extend(
            {
                "host_path": str(item),
                "container_path": f"{container_root}/{item.relative_to(skills_dir)}",
            }
            for item in files
        )

    try:
        from agent.skill_utils import get_external_skills_dirs

        external_dirs = await get_external_skills_dirs()
    except ImportError:
        external_dirs = []
    for index, external_dir in enumerate(external_dirs):
        if not await aiofiles.os.path.isdir(external_dir):
            continue
        _directories, files, _symlinks = await _walk_tree(external_dir)
        container_root = f"{container_base.rstrip('/')}/external_skills/{index}"
        result.extend(
            {
                "host_path": str(item),
                "container_path": f"{container_root}/{item.relative_to(external_dir)}",
            }
            for item in files
        )
    return result


# Each pair is (new_subpath, legacy_name), matching get_hermes_dir().
_CACHE_DIRS: list[tuple[str, str]] = [
    ("cache/documents", "document_cache"),
    ("cache/images", "image_cache"),
    ("cache/audio", "audio_cache"),
    ("cache/videos", "video_cache"),
    ("cache/screenshots", "browser_screenshots"),
    ("cache/web", "web_cache"),
    ("cache/delegation", "delegation_cache"),
    ("images", "images"),
]


# Path translation is intentionally synchronous and pure. Awaited mount
# discovery refreshes this profile/base-keyed table before a remote backend
# exposes those paths to the agent.
_cache_mounts_by_scope: dict[tuple[str, str], tuple[Dict[str, str], ...]] = {}


def _cache_mount_scope(container_base: str) -> tuple[str, str]:
    return str(_resolve_hermes_home()), container_base.rstrip("/")


async def get_cache_directory_mounts(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return existing cache-directory bind mount entries."""
    from hermes_constants import get_hermes_dir

    mounts: List[Dict[str, str]] = []
    for new_subpath, old_name in _CACHE_DIRS:
        host_dir = await get_hermes_dir(new_subpath, old_name)
        if await aiofiles.os.path.isdir(host_dir):
            mounts.append(
                {
                    "host_path": str(host_dir),
                    "container_path": f"{container_base.rstrip('/')}/{new_subpath}",
                }
            )
    _cache_mounts_by_scope[_cache_mount_scope(container_base)] = tuple(
        dict(mount) for mount in mounts
    )
    return mounts


def _known_cache_mounts(container_base: str) -> tuple[Dict[str, str], ...]:
    return _cache_mounts_by_scope.get(_cache_mount_scope(container_base), ())


def map_cache_path_to_container(
    host_path: str,
    container_base: str = "/root/.hermes",
) -> Optional[str]:
    """Purely map a host cache path using the latest awaited mount discovery."""
    path = Path(host_path)
    for mount in _known_cache_mounts(container_base):
        host_dir = Path(mount["host_path"])
        try:
            relative = path.relative_to(host_dir)
        except ValueError:
            continue
        return posixpath.join(mount["container_path"], relative.as_posix())
    return None


def from_agent_visible_cache_path(
    container_path: str,
    container_base: str = "/root/.hermes",
) -> str:
    """Translate a Docker-visible cache path back to its host mount path."""
    from agent.secret_scope import get_secret

    if get_secret("TERMINAL_ENV", "local") != "docker":
        return container_path
    path = Path(container_path)
    for mount in _known_cache_mounts(container_base):
        try:
            relative = path.relative_to(mount["container_path"])
        except ValueError:
            continue
        return str(Path(mount["host_path"]) / relative)
    return container_path


def to_agent_visible_cache_path(
    host_path: str,
    container_base: str = "/root/.hermes",
) -> str:
    """Translate a host cache path to its Docker-visible mount path."""
    from agent.secret_scope import get_secret

    if get_secret("TERMINAL_ENV", "local") != "docker":
        return host_path
    mapped = map_cache_path_to_container(host_path, container_base=container_base)
    return mapped if mapped is not None else host_path


async def iter_cache_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return regular files below all existing cache directories."""
    result: List[Dict[str, str]] = []
    for mount in await get_cache_directory_mounts(container_base):
        host_dir = Path(mount["host_path"])
        _directories, files, _symlinks = await _walk_tree(host_dir)
        result.extend(
            {
                "host_path": str(item),
                "container_path": (
                    f"{mount['container_path']}/{item.relative_to(host_dir)}"
                ),
            }
            for item in files
        )
    return result


def clear_credential_files() -> None:
    """Reset the context-scoped registry (for example on session reset)."""
    _get_registered().clear()
