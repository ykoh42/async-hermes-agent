"""Native-async skill usage telemetry and provenance storage.

Hermes stores per-skill ownership and activity metadata in
``$HERMES_HOME/skills/.usage.json``.  The sidecar format is retained while its
read/modify/write boundary is serialized with a cancellation-safe,
non-blocking file lock so concurrent agents and processes do not lose updates.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Set
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
PROTECTED_BUILTIN_SKILLS: Set[str] = {"plan"}

_os_open = aiofiles.os.wrap(os.open)
_os_close = aiofiles.os.wrap(os.close)
_os_fsync = aiofiles.os.wrap(os.fsync)
_os_fstat = aiofiles.os.wrap(os.fstat)
_os_lseek = aiofiles.os.wrap(os.lseek)
_os_write = aiofiles.os.wrap(os.write)


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_record() -> Dict[str, Any]:
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


def _release_platform_lock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


async def _release_and_close_lock(fd: int, acquired: bool) -> None:
    """Release and close one owned lock descriptor as an indivisible cleanup."""
    try:
        if acquired:
            _release_platform_lock(fd)
    finally:
        await _os_close(fd)


@asynccontextmanager
async def _usage_file_lock():
    """Serialize sidecar mutations across tasks and processes."""
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


async def load_usage() -> Dict[str, Dict[str, Any]]:
    """Read the entire usage map, returning an empty map when absent/corrupt."""
    path = _usage_file()
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, dict)
    }


async def save_usage(data: Dict[str, Dict[str, Any]]) -> None:
    """Atomically persist the usage map in the upstream JSON format."""
    path = _usage_file()
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    temporary = path.parent / f".usage_{uuid.uuid4().hex}.tmp"
    try:
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
    except BaseException:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


async def get_record(skill_name: str) -> Dict[str, Any]:
    record = (await load_usage()).get(skill_name)
    if not isinstance(record, dict):
        return _empty_record()
    result = _empty_record()
    result.update(record)
    return result


async def _mutate(skill_name: str, mutator: Callable[[Dict[str, Any]], None]) -> None:
    if not skill_name:
        return
    try:
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
    def apply(record: Dict[str, Any]) -> None:
        record["view_count"] = int(record.get("view_count") or 0) + 1
        record["last_viewed_at"] = _now_iso()

    await _mutate(skill_name, apply)


async def bump_use(skill_name: str) -> None:
    def apply(record: Dict[str, Any]) -> None:
        record["use_count"] = int(record.get("use_count") or 0) + 1
        record["last_used_at"] = _now_iso()

    await _mutate(skill_name, apply)


async def bump_patch(skill_name: str) -> None:
    def apply(record: Dict[str, Any]) -> None:
        record["patch_count"] = int(record.get("patch_count") or 0) + 1
        record["last_patched_at"] = _now_iso()

    await _mutate(skill_name, apply)


async def mark_agent_created(skill_name: str) -> None:
    await _mutate(skill_name, lambda record: record.__setitem__("created_by", "agent"))


async def set_pinned(skill_name: str, pinned: bool) -> None:
    await _mutate(skill_name, lambda record: record.__setitem__("pinned", bool(pinned)))


async def set_state(skill_name: str, state: str) -> None:
    if state not in {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}:
        return

    def apply(record: Dict[str, Any]) -> None:
        record["state"] = state
        if state == STATE_ARCHIVED:
            record["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            record["archived_at"] = None

    await _mutate(skill_name, apply)


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


async def _read_bundled_manifest_names() -> Set[str]:
    try:
        async with aiofiles.open(
            _skills_dir() / ".bundled_manifest",
            encoding="utf-8",
        ) as handle:
            lines = (await handle.read()).splitlines()
    except (FileNotFoundError, OSError, UnicodeError):
        return set()
    return {
        name
        for line in lines
        if (name := line.strip().split(":", 1)[0].strip())
    }


async def _read_hub_installed_names() -> Set[str]:
    try:
        async with aiofiles.open(
            _skills_dir() / ".hub" / "lock.json",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            data = json.loads(await handle.read())
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return set()
    installed = data.get("installed") if isinstance(data, dict) else None
    if not isinstance(installed, dict):
        return set()
    names = {str(name) for name in installed}
    skills_dir = _skills_dir()
    realpath = aiofiles.os.wrap(os.path.realpath)
    resolved_root = Path(await realpath(skills_dir))
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
            resolved_skill.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        skill_md = resolved_skill / "SKILL.md"
        if await aiofiles.os.path.exists(skill_md):
            names.add(await _read_skill_name(skill_md, resolved_skill.name))
    return names


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


async def _archive_skill_owned(skill_name: str) -> tuple[bool, str]:
    from agent.skill_utils import is_external_skill_path

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
        await aiofiles.os.makedirs(archive_root, exist_ok=True)
        destination = archive_root / skill_dir.name
        if await aiofiles.os.path.exists(destination):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            destination = archive_root / f"{skill_dir.name}-{stamp}"
        try:
            await aiofiles.os.replace(skill_dir, destination)
        except OSError as exc:
            return False, f"failed to archive: {exc}"

        data = await load_usage()
        record = data.get(skill_name)
        if not isinstance(record, dict):
            record = _empty_record()
        record["state"] = STATE_ARCHIVED
        record["archived_at"] = _now_iso()
        data[skill_name] = record
        await save_usage(data)
        return True, f"archived to {destination}"


async def archive_skill(skill_name: str) -> tuple[bool, str]:
    """Move one curator-managed skill to the recoverable archive directory."""
    return await _finish_owned_task(
        asyncio.create_task(_archive_skill_owned(skill_name))
    )


async def adopt_skill(skill_name: str) -> tuple[bool, str]:
    """Opt an existing local skill into autonomous curation management."""
    from agent.skill_utils import is_external_skill_path

    skill_dir = await _find_local_skill_dir(skill_name)
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    if await is_external_skill_path(skill_dir):
        return False, (
            f"skill '{skill_name}' lives in skills.external_dirs; external "
            "skills are read-only to the curator"
        )
    if (
        is_protected_builtin(skill_name)
        or await is_hub_installed(skill_name)
        or await is_bundled(skill_name)
    ):
        return False, f"skill '{skill_name}' is not eligible for curation"
    await mark_agent_created(skill_name)
    if not await is_curator_managed(skill_name):
        return False, f"could not mark '{skill_name}' as curator-managed"
    return True, f"adopted '{skill_name}' into curator management"
