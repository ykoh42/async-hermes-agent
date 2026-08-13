"""Native-async skill usage telemetry and provenance storage.

Hermes stores per-skill ownership and activity metadata in
``$HERMES_HOME/skills/.usage.json``.  The sidecar format is retained while its
read/modify/write boundary is serialized with a cancellation-safe,
non-blocking file lock so concurrent agents and processes do not lose updates.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import errno
import json
import logging
import os
from pathlib import Path
import stat
from typing import Any
from collections.abc import Callable
import uuid

import aiofiles
import aiofiles.os

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

try:  # pragma: no branch - exactly one platform lock is normally available
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}
PROTECTED_BUILTIN_SKILLS: set[str] = {"plan"}

_os_open = aiofiles.os.wrap(os.open)
_os_close = aiofiles.os.wrap(os.close)
_os_fsync = aiofiles.os.wrap(os.fsync)
_os_fstat = aiofiles.os.wrap(os.fstat)
_os_lseek = aiofiles.os.wrap(os.lseek)
_os_write = aiofiles.os.wrap(os.write)
_os_lstat = aiofiles.os.wrap(os.lstat)
_os_chmod = aiofiles.os.wrap(os.chmod)
_os_utime = aiofiles.os.wrap(os.utime)
_os_lexists = aiofiles.os.wrap(os.path.lexists)


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def latest_activity_at(record: dict[str, Any]) -> str | None:
    """Return the newest use, view, or patch timestamp."""
    latest_dt: datetime | None = None
    latest_raw: str | None = None
    for key in ("last_used_at", "last_viewed_at", "last_patched_at"):
        raw = record.get(key)
        parsed = _parse_iso_timestamp(raw)
        if parsed is not None and (latest_dt is None or parsed > latest_dt):
            latest_dt = parsed
            latest_raw = str(raw)
    return latest_raw


def activity_count(record: dict[str, Any]) -> int:
    total = 0
    for key in ("use_count", "view_count", "patch_count"):
        try:
            total += int(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _empty_record() -> dict[str, Any]:
    return {
        "created_by": None,
        "use_count": 0,
        "view_count": 0,
        "last_used_at": None,
        "last_viewed_at": None,
        "patch_count": 0,
        "last_patched_at": None,
        "created_at": _now_iso(),
        "state": STATE_ACTIVE,
        "pinned": False,
        "archived_at": None,
    }


def _is_curator_managed_record(record: Any) -> bool:
    """Return whether a sidecar record opts a skill into autonomous curation."""
    return isinstance(record, dict) and (
        record.get("created_by") == "agent"
        or record.get("agent_created") is True
    )


def is_protected_builtin(skill_name: str) -> bool:
    return skill_name in PROTECTED_BUILTIN_SKILLS


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one accepted sidecar/filesystem mutation through cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
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


async def _acquire_platform_lock(fd: int) -> None:
    """Acquire one cross-process lock without blocking the event loop."""
    if fcntl is not None:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                await asyncio.sleep(0.01)
    if msvcrt is not None:  # pragma: no cover - Windows
        stat_result = await _os_fstat(fd)
        if stat_result.st_size == 0:
            await _os_write(fd, b" ")
        await _os_lseek(fd, 0, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                await asyncio.sleep(0.01)


async def _release_platform_lock(fd: int) -> None:
    if fcntl is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        try:
            await _os_lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


async def _release_and_close_lock(fd: int, acquired: bool) -> None:
    """Release and close one owned lock descriptor as an indivisible cleanup."""
    try:
        if acquired:
            await _release_platform_lock(fd)
    finally:
        await _os_close(fd)


@asynccontextmanager
async def _usage_file_lock():
    """Serialize sidecar mutations across tasks and processes."""
    if fcntl is None and msvcrt is None:
        yield
        return
    lock_path = _usage_file().with_suffix(".json.lock")
    await aiofiles.os.makedirs(lock_path.parent, exist_ok=True)
    fd = await _os_open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        await _acquire_platform_lock(fd)
        acquired = True
        yield
    finally:
        await _finish_owned_task(
            asyncio.create_task(_release_and_close_lock(fd, acquired))
        )


async def load_usage() -> dict[str, dict[str, Any]]:
    """Read the entire usage map, returning an empty map when absent/corrupt."""
    path = _usage_file()
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, dict)
    }


async def save_usage(data: dict[str, dict[str, Any]]) -> None:
    """Atomically persist the usage map; failures remain best-effort."""
    path = _usage_file()
    temporary: Path | None = None
    error: Exception | None = None

    async def _cleanup() -> None:
        if temporary is not None:
            with contextlib.suppress(OSError):
                await aiofiles.os.remove(temporary)

    try:
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        temporary = path.parent / f".usage_{uuid.uuid4().hex}.tmp"
        async with aiofiles.open(
            temporary,
            "x",
            encoding="utf-8",
            opener=lambda raw_path, flags: os.open(raw_path, flags, 0o600),
        ) as handle:
            await handle.write(
                json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
            )
            await handle.flush()
            await _os_fsync(handle.fileno())
        await aiofiles.os.replace(temporary, path)
    except asyncio.CancelledError:
        await _finish_owned_task(asyncio.create_task(_cleanup()))
        raise
    except Exception as exc:
        error = exc
    await _finish_owned_task(asyncio.create_task(_cleanup()))
    if error is not None:
        logger.debug(
            "Failed to write %s: %s",
            path,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


async def get_record(skill_name: str) -> dict[str, Any]:
    record = (await load_usage()).get(skill_name)
    if not isinstance(record, dict):
        return _empty_record()
    result = _empty_record()
    result.update(record)
    return result


async def _mutate(
    skill_name: str,
    mutator: Callable[[dict[str, Any]], None],
    *,
    require_curation_eligible: bool = False,
) -> None:
    if not skill_name:
        return
    try:
        if require_curation_eligible and not await is_curation_eligible(skill_name):
            return
        async with _usage_file_lock():
            data = await load_usage()
            record = data.get(skill_name)
            if not isinstance(record, dict):
                record = _empty_record()
            mutator(record)
            data[skill_name] = record
            await save_usage(data)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("skill usage mutation failed for %s", skill_name, exc_info=True)


async def bump_view(skill_name: str) -> None:
    def apply(record: dict[str, Any]) -> None:
        record["view_count"] = int(record.get("view_count") or 0) + 1
        record["last_viewed_at"] = _now_iso()

    await _mutate(skill_name, apply)


async def bump_use(skill_name: str) -> None:
    def apply(record: dict[str, Any]) -> None:
        record["use_count"] = int(record.get("use_count") or 0) + 1
        record["last_used_at"] = _now_iso()

    await _mutate(skill_name, apply)


async def bump_patch(skill_name: str) -> None:
    def apply(record: dict[str, Any]) -> None:
        record["patch_count"] = int(record.get("patch_count") or 0) + 1
        record["last_patched_at"] = _now_iso()

    await _mutate(skill_name, apply)


async def mark_agent_created(skill_name: str) -> None:
    await _mutate(
        skill_name,
        lambda record: record.__setitem__("created_by", "agent"),
        require_curation_eligible=True,
    )


async def set_pinned(skill_name: str, pinned: bool) -> None:
    await _mutate(
        skill_name,
        lambda record: record.__setitem__("pinned", bool(pinned)),
        require_curation_eligible=True,
    )


async def set_state(skill_name: str, state: str) -> None:
    if state not in _VALID_STATES:
        return

    def apply(record: dict[str, Any]) -> None:
        record["state"] = state
        if state == STATE_ARCHIVED:
            record["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            record["archived_at"] = None

    await _mutate(skill_name, apply, require_curation_eligible=True)


async def set_sync(skill_name: str, sync: bool) -> None:
    await _mutate(
        skill_name,
        lambda record: record.__setitem__("sync", bool(sync)),
        require_curation_eligible=True,
    )


async def is_sync_enabled(skill_name: str) -> bool:
    return (await get_record(skill_name)).get("sync") is True


async def seed_record_if_missing(skill_name: str) -> None:
    if not skill_name or not await is_curation_eligible(skill_name):
        return
    try:
        async with _usage_file_lock():
            data = await load_usage()
            if isinstance(data.get(skill_name), dict):
                return
            data[skill_name] = _empty_record()
            await save_usage(data)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug(
            "skill usage seed failed for %s", skill_name, exc_info=True
        )


async def forget(skill_name: str) -> None:
    if not skill_name:
        return
    try:
        async with _usage_file_lock():
            data = await load_usage()
            if data.pop(skill_name, None) is not None:
                await save_usage(data)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("skill usage deletion failed for %s", skill_name, exc_info=True)


async def is_curator_managed(skill_name: str) -> bool:
    return _is_curator_managed_record((await load_usage()).get(skill_name))


async def _read_bundled_manifest_names() -> set[str]:
    try:
        async with aiofiles.open(
            _skills_dir() / ".bundled_manifest",
            encoding="utf-8",
        ) as handle:
            lines = (await handle.read()).splitlines()
    except OSError:
        return set()
    return {
        name
        for line in lines
        if (name := line.strip().split(":", 1)[0].strip())
    }


async def _read_hub_installed_names() -> set[str]:
    try:
        async with aiofiles.open(
            _skills_dir() / ".hub" / "lock.json",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            data = json.loads(await handle.read())
    except (OSError, json.JSONDecodeError):
        return set()
    installed = data.get("installed") if isinstance(data, dict) else None
    if not isinstance(installed, dict):
        return set()
    names = {str(name) for name in installed}
    skills_dir = _skills_dir()
    realpath = aiofiles.os.wrap(os.path.realpath)
    for entry in installed.values():
        if not isinstance(entry, dict):
            continue
        install_path = entry.get("install_path")
        if not isinstance(install_path, str) or not install_path.strip():
            continue
        skill_dir = Path(install_path)
        if not skill_dir.is_absolute():
            skill_dir = skills_dir / skill_dir
        try:
            resolved_skill = Path(await realpath(skill_dir))
            resolved_root = Path(await realpath(skills_dir))
            resolved_skill.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        skill_md = resolved_skill / "SKILL.md"
        if await aiofiles.os.path.exists(skill_md):
            names.add(await _read_skill_name(skill_md, resolved_skill.name))
    return names


async def _prune_builtins_enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = await load_config_readonly()
        curator = cfg.get("curator") if isinstance(cfg, dict) else None
        if isinstance(curator, dict):
            return bool(curator.get("prune_builtins", True))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Failed to read curator.prune_builtins: %s", exc)
    return True


def _suppressed_file() -> Path:
    return _skills_dir() / ".curator_suppressed"


async def read_suppressed_names() -> set[str]:
    path = _suppressed_file()
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            lines = (await handle.read()).splitlines()
    except OSError as exc:
        logger.debug("Failed to read curator suppression list: %s", exc)
        return set()
    return {
        line
        for raw in lines
        if (line := raw.strip()) and not line.startswith("#")
    }


async def _write_suppressed_names(names: set[str]) -> None:
    path = _suppressed_file()
    temporary = path.parent / f".curator_suppressed_{uuid.uuid4().hex}.tmp"
    data = "\n".join(sorted(names)) + ("\n" if names else "")
    error: Exception | None = None

    async def _cleanup() -> None:
        with contextlib.suppress(OSError):
            await aiofiles.os.remove(temporary)

    try:
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        async with aiofiles.open(
            temporary,
            "x",
            encoding="utf-8",
            opener=lambda raw_path, flags: os.open(raw_path, flags, 0o600),
        ) as handle:
            await handle.write(data)
            await handle.flush()
            await _os_fsync(handle.fileno())
        await aiofiles.os.replace(temporary, path)
    except asyncio.CancelledError:
        await _finish_owned_task(asyncio.create_task(_cleanup()))
        raise
    except Exception as exc:
        error = exc
    await _finish_owned_task(asyncio.create_task(_cleanup()))
    if error is not None:
        logger.debug(
            "Failed to write curator suppression list: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


async def add_suppressed_name(skill_name: str) -> None:
    if not skill_name:
        return
    try:
        async with _usage_file_lock():
            names = await read_suppressed_names()
            if skill_name not in names:
                names.add(skill_name)
                await _write_suppressed_names(names)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug(
            "Failed to add curator suppression for %s", skill_name, exc_info=True
        )


async def remove_suppressed_name(skill_name: str) -> None:
    if not skill_name:
        return
    try:
        async with _usage_file_lock():
            names = await read_suppressed_names()
            if skill_name in names:
                names.discard(skill_name)
                await _write_suppressed_names(names)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug(
            "Failed to remove curator suppression for %s",
            skill_name,
            exc_info=True,
        )


async def is_hub_installed(skill_name: str) -> bool:
    return skill_name in await _read_hub_installed_names()


async def is_bundled(skill_name: str) -> bool:
    return skill_name in await _read_bundled_manifest_names()


async def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Read the upstream ``name:`` frontmatter field from one skill file."""
    try:
        async with aiofiles.open(
            skill_md,
            encoding="utf-8",
            errors="replace",
        ) as handle:
            text = await handle.read(4000)
    except OSError:
        return fallback
    in_frontmatter = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter and stripped.startswith("name:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return fallback


async def _find_local_skill_dir(skill_name: str) -> Path | None:
    from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files

    root = _skills_dir()
    if not await aiofiles.os.path.isdir(root):
        return None
    async for skill_md in iter_skill_index_files(root, "SKILL.md"):
        if await is_excluded_skill_path(skill_md, root=root):
            continue
        if await _read_skill_name(skill_md, skill_md.parent.name) == skill_name:
            return skill_md.parent
    return None


async def _find_skill_dir(skill_name: str) -> Path | None:
    """Compatibility name for local skill discovery."""
    return await _find_local_skill_dir(skill_name)


async def _find_external_skill_dir(skill_name: str) -> Path | None:
    from agent.skill_utils import (
        get_all_skills_dirs,
        is_excluded_skill_path,
        iter_skill_index_files,
    )

    roots = await get_all_skills_dirs()
    for root in roots[1:]:
        if not await aiofiles.os.path.isdir(root):
            continue
        async for skill_md in iter_skill_index_files(root, "SKILL.md"):
            if await is_excluded_skill_path(skill_md, root=root):
                continue
            if await _read_skill_name(skill_md, skill_md.parent.name) == skill_name:
                return skill_md.parent
    return None


async def _copy_regular_file(source: Path, destination: Path, metadata) -> None:
    async with (
        aiofiles.open(source, "rb") as source_handle,
        aiofiles.open(destination, "xb") as destination_handle,
    ):
        while chunk := await source_handle.read(1024 * 1024):
            await destination_handle.write(chunk)
        await destination_handle.flush()
        await _os_fsync(destination_handle.fileno())
    await _os_chmod(destination, stat.S_IMODE(metadata.st_mode))
    await _os_utime(
        destination,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
    )


async def _copy_tree(source: Path, destination: Path) -> None:
    """Native-async equivalent of shutil.copytree(..., symlinks=True)."""
    metadata = await _os_lstat(source)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(source)
    await aiofiles.os.mkdir(destination, mode=stat.S_IMODE(metadata.st_mode))
    for name in await aiofiles.os.listdir(source):
        source_child = source / name
        destination_child = destination / name
        child_metadata = await _os_lstat(source_child)
        if stat.S_ISLNK(child_metadata.st_mode):
            await aiofiles.os.symlink(
                await aiofiles.os.readlink(source_child),
                destination_child,
            )
        elif stat.S_ISDIR(child_metadata.st_mode):
            await _copy_tree(source_child, destination_child)
        elif stat.S_ISREG(child_metadata.st_mode):
            await _copy_regular_file(
                source_child,
                destination_child,
                child_metadata,
            )
        else:
            raise OSError(
                errno.ENOTSUP,
                "unsupported special file in skill tree",
                str(source_child),
            )
    await _os_chmod(destination, stat.S_IMODE(metadata.st_mode))
    await _os_utime(
        destination,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
    )


async def _remove_tree(path: Path) -> None:
    try:
        metadata = await _os_lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        await aiofiles.os.remove(path)
        return
    for name in await aiofiles.os.listdir(path):
        await _remove_tree(path / name)
    await aiofiles.os.rmdir(path)


async def _move_tree(source: Path, destination: Path) -> None:
    """Move a directory, preserving upstream's cross-device fallback."""
    try:
        await aiofiles.os.replace(source, destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    staging = destination.with_name(
        f".{destination.name}.hermes-move-{uuid.uuid4().hex}.tmp"
    )
    error: BaseException | None = None
    try:
        await _copy_tree(source, staging)
        await aiofiles.os.replace(staging, destination)
        await _remove_tree(source)
    except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised after cleanup
        error = exc
    except Exception as exc:
        error = exc
    if await _os_lexists(staging):
        await _finish_owned_task(asyncio.create_task(_remove_tree(staging)))
    if error is not None:
        raise error


async def is_agent_created(skill_name: str) -> bool:
    """Whether a skill has no bundled, hub, or external owner."""
    off_limits = await _read_bundled_manifest_names() | await _read_hub_installed_names()
    if skill_name in off_limits:
        return False
    return not (
        await _find_local_skill_dir(skill_name) is None
        and await _find_external_skill_dir(skill_name) is not None
    )


def _external_read_only_message(skill_name: str) -> str:
    return (
        f"skill '{skill_name}' lives in skills.external_dirs; "
        "external skills are read-only to the curator"
    )


async def is_curation_eligible(
    skill_name: str,
    skill_path: Path | None = None,
) -> bool:
    from agent.skill_utils import is_external_skill_path

    if skill_path is not None and await is_external_skill_path(skill_path):
        return False
    if is_protected_builtin(skill_name):
        return False
    if await is_hub_installed(skill_name):
        return False
    if await is_bundled(skill_name):
        return await _prune_builtins_enabled()
    local_dir = await _find_local_skill_dir(skill_name)
    if local_dir is not None:
        return not await is_external_skill_path(local_dir)
    if await _find_external_skill_dir(skill_name) is not None:
        return False
    return True


async def list_agent_created_skill_names() -> list[str]:
    from agent.skill_utils import is_excluded_skill_path, is_external_skill_path, iter_skill_index_files

    base = _skills_dir()
    if not await aiofiles.os.path.isdir(base):
        return []
    hub = await _read_hub_installed_names()
    bundled = await _read_bundled_manifest_names()
    prune_builtins = await _prune_builtins_enabled()
    usage = await load_usage()
    names: list[str] = []
    async for skill_md in iter_skill_index_files(base, "SKILL.md"):
        if await is_excluded_skill_path(skill_md, root=base):
            continue
        if await is_external_skill_path(skill_md):
            continue
        name = await _read_skill_name(skill_md, skill_md.parent.name)
        if name in hub or is_protected_builtin(name):
            continue
        if name in bundled:
            if prune_builtins:
                names.append(name)
            continue
        if _is_curator_managed_record(usage.get(name)):
            names.append(name)
    return sorted(set(names))


async def list_archived_skill_names() -> list[str]:
    archive_root = _archive_dir()
    if not await aiofiles.os.path.isdir(archive_root):
        return []
    names = []
    for name in await aiofiles.os.listdir(archive_root):
        if await aiofiles.os.path.isdir(archive_root / name):
            names.append(name)
    return sorted(set(names))


async def list_unmanaged_skill_names() -> list[str]:
    from agent.skill_utils import is_excluded_skill_path, is_external_skill_path, iter_skill_index_files

    base = _skills_dir()
    if not await aiofiles.os.path.isdir(base):
        return []
    hub = await _read_hub_installed_names()
    bundled = await _read_bundled_manifest_names()
    usage = await load_usage()
    names: list[str] = []
    async for skill_md in iter_skill_index_files(base, "SKILL.md"):
        if await is_excluded_skill_path(skill_md, root=base):
            continue
        if await is_external_skill_path(skill_md):
            continue
        name = await _read_skill_name(skill_md, skill_md.parent.name)
        if name in hub or name in bundled or is_protected_builtin(name):
            continue
        if _is_curator_managed_record(usage.get(name)):
            continue
        if await is_curation_eligible(name, skill_md):
            names.append(name)
    return sorted(set(names))


async def unmanaged_report() -> list[dict[str, Any]]:
    usage = await load_usage()
    rows: list[dict[str, Any]] = []
    for name in await list_unmanaged_skill_names():
        raw = usage.get(name)
        record = dict(raw) if isinstance(raw, dict) else _empty_record()
        for key, value in _empty_record().items():
            record.setdefault(key, value)
        row = {"name": name, **record}
        row["has_provenance_key"] = isinstance(raw, dict) and "created_by" in raw
        row["has_record"] = isinstance(raw, dict)
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return rows


async def _archive_skill_owned(skill_name: str) -> tuple[bool, str]:
    from agent.skill_utils import is_external_skill_path

    local_skill_dir = await _find_local_skill_dir(skill_name)
    if local_skill_dir is None and await _find_external_skill_dir(skill_name) is not None:
        return False, _external_read_only_message(skill_name)
    if not await is_curation_eligible(skill_name, local_skill_dir):
        if is_protected_builtin(skill_name):
            return False, (
                f"skill '{skill_name}' is a protected built-in; it backs "
                "load-bearing UX and is never archived or consolidated"
            )
        if await is_hub_installed(skill_name):
            return False, f"skill '{skill_name}' is hub-installed; never archive"
        return False, (
            f"skill '{skill_name}' is a bundled built-in; enable "
            "curator.prune_builtins to allow pruning it"
        )

    async with _usage_file_lock():
        skill_dir = await _find_local_skill_dir(skill_name)
        if skill_dir is None:
            return False, f"skill '{skill_name}' not found"
        if await is_external_skill_path(skill_dir):
            return False, (
                f"skill '{skill_name}' lives in skills.external_dirs; external "
                "skills are read-only to the curator"
            )
        if await aiofiles.os.path.islink(skill_dir):
            return False, f"refusing to archive symlinked skill '{skill_name}'"

        root = _skills_dir()
        realpath = aiofiles.os.wrap(os.path.realpath)
        try:
            resolved_skill = Path(await realpath(skill_dir))
            resolved_root = Path(await realpath(root))
            relative = resolved_skill.relative_to(resolved_root)
        except (OSError, ValueError):
            return False, f"skill '{skill_name}' is outside the local skills root"
        if not relative.parts or relative.parts[0] == ".archive":
            return False, f"skill '{skill_name}' is not an active local skill"

        archive_root = root / ".archive"
        try:
            await aiofiles.os.makedirs(archive_root, exist_ok=True)
        except OSError as exc:
            return False, f"failed to create archive dir: {exc}"
        destination = archive_root / skill_dir.name
        if await aiofiles.os.path.exists(destination):
            stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            destination = archive_root / f"{skill_dir.name}-{stamp}"
        try:
            await _move_tree(skill_dir, destination)
        except OSError as exc:
            return False, f"failed to archive: {exc}"

        if await is_bundled(skill_name):
            try:
                suppressed = await read_suppressed_names()
                suppressed.add(skill_name)
                await _write_suppressed_names(suppressed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Failed to persist built-in suppression for %s",
                    skill_name,
                    exc_info=True,
                )

        try:
            data = await load_usage()
            record = data.get(skill_name)
            if not isinstance(record, dict):
                record = _empty_record()
            record["state"] = STATE_ARCHIVED
            record["archived_at"] = _now_iso()
            data[skill_name] = record
            await save_usage(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "Failed to persist archived state for %s",
                skill_name,
                exc_info=True,
            )
        return True, f"archived to {destination}"


async def archive_skill(skill_name: str) -> tuple[bool, str]:
    """Move one curator-eligible skill to the recoverable archive."""
    return await _finish_owned_task(
        asyncio.create_task(_archive_skill_owned(skill_name))
    )


async def adopt_skill(skill_name: str) -> tuple[bool, str]:
    """Opt an existing local skill into autonomous curation management."""
    from agent.skill_utils import is_external_skill_path

    if not skill_name:
        return False, "no skill name given"
    if is_protected_builtin(skill_name):
        return False, f"'{skill_name}' is a protected built-in; the curator never manages it"
    if await is_hub_installed(skill_name):
        return False, f"'{skill_name}' is hub-installed; its upstream owns it"
    if await is_bundled(skill_name):
        return False, (
            f"'{skill_name}' is a bundled built-in — it is governed by "
            "curator.prune_builtins, not by adoption"
        )
    skill_dir = await _find_local_skill_dir(skill_name)
    if skill_dir is None:
        if await _find_external_skill_dir(skill_name) is not None:
            return False, (
                f"'{skill_name}' lives in skills.external_dirs and is "
                "read-only to the curator"
            )
        return False, f"skill '{skill_name}' not found"
    if await is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    if await is_curator_managed(skill_name):
        return True, f"'{skill_name}' is already curator-managed"
    await mark_agent_created(skill_name)
    if not await is_curator_managed(skill_name):
        return False, f"could not mark '{skill_name}' as curator-managed"
    return True, f"adopted '{skill_name}' into curator management"


async def _iter_archive_dirs(root: Path):
    try:
        names = await aiofiles.os.listdir(root)
    except OSError:
        return
    for name in names:
        candidate = root / name
        if not await aiofiles.os.path.isdir(candidate):
            continue
        yield candidate
        if not await aiofiles.os.path.islink(candidate):
            async for descendant in _iter_archive_dirs(candidate):
                yield descendant


async def _restore_skill_owned(skill_name: str) -> tuple[bool, str]:
    if await is_hub_installed(skill_name):
        return False, (
            f"skill '{skill_name}' is now hub-installed; restore would "
            "shadow the upstream version"
        )
    if await is_bundled(skill_name) and not await _prune_builtins_enabled():
        return False, (
            f"skill '{skill_name}' is now bundled; restore would shadow "
            "the upstream version"
        )
    archive_root = _archive_dir()
    if not await aiofiles.os.path.isdir(archive_root):
        return False, "no archive directory"

    all_dirs = [path async for path in _iter_archive_dirs(archive_root)]
    candidates = [path for path in all_dirs if path.name == skill_name]
    if not candidates:
        prefix = f"{skill_name}-"
        candidates = sorted(
            [
                path
                for path in all_dirs
                if path.name.startswith(prefix)
                and len(path.name) - len(prefix) == 14
                and path.name[len(prefix):].isdigit()
            ],
            reverse=True,
        )
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"

    source = candidates[0]
    destination = _skills_dir() / skill_name
    if await aiofiles.os.path.exists(destination):
        return False, f"destination already exists: {destination}"
    if await aiofiles.os.path.islink(source):
        return False, f"refusing to restore symlinked skill '{skill_name}'"
    realpath = aiofiles.os.wrap(os.path.realpath)
    try:
        resolved_source = Path(await realpath(source))
        resolved_archive = Path(await realpath(archive_root))
        resolved_source.relative_to(resolved_archive)
    except (OSError, ValueError):
        return False, f"skill '{skill_name}' is outside the archive root"

    async with _usage_file_lock():
        try:
            await _move_tree(source, destination)
        except OSError as exc:
            return False, f"failed to restore: {exc}"
        try:
            suppressed = await read_suppressed_names()
            if skill_name in suppressed:
                suppressed.discard(skill_name)
                await _write_suppressed_names(suppressed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "Failed to clear built-in suppression for %s",
                skill_name,
                exc_info=True,
            )
        try:
            data = await load_usage()
            record = data.get(skill_name)
            if not isinstance(record, dict):
                record = _empty_record()
            record["state"] = STATE_ACTIVE
            record["archived_at"] = None
            data[skill_name] = record
            await save_usage(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "Failed to persist restored state for %s",
                skill_name,
                exc_info=True,
            )
    return True, f"restored to {destination}"


async def restore_skill(skill_name: str) -> tuple[bool, str]:
    return await _finish_owned_task(
        asyncio.create_task(_restore_skill_owned(skill_name))
    )


async def provenance(skill_name: str) -> str:
    if await is_hub_installed(skill_name):
        return "hub"
    if await is_bundled(skill_name):
        return "bundled"
    return "agent"


async def curated_report() -> list[dict[str, Any]]:
    data = await load_usage()
    rows: list[dict[str, Any]] = []
    for name in await list_agent_created_skill_names():
        raw = data.get(name)
        persisted = isinstance(raw, dict)
        record = dict(raw) if isinstance(raw, dict) else _empty_record()
        for key, value in _empty_record().items():
            record.setdefault(key, value)
        row = {"name": name, **record, "_persisted": persisted}
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        row["provenance"] = await provenance(name)
        rows.append(row)
    return rows


async def agent_created_report() -> list[dict[str, Any]]:
    """Compatibility alias for :func:`curated_report`."""
    return await curated_report()


async def usage_report() -> list[dict[str, Any]]:
    from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files

    base = _skills_dir()
    if not await aiofiles.os.path.isdir(base):
        return []
    data = await load_usage()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    async for skill_md in iter_skill_index_files(base, "SKILL.md"):
        if await is_excluded_skill_path(skill_md, root=base):
            continue
        name = await _read_skill_name(skill_md, skill_md.parent.name)
        if name in seen:
            continue
        seen.add(name)
        raw = data.get(name)
        persisted = isinstance(raw, dict)
        record = dict(raw) if isinstance(raw, dict) else _empty_record()
        for key, value in _empty_record().items():
            record.setdefault(key, value)
        row = {
            "name": name,
            **record,
            "provenance": await provenance(name),
            "_persisted": persisted,
        }
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return sorted(rows, key=lambda row: row["name"])
