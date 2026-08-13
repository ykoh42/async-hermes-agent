"""Native-async workspace and project-root resolution for LSP."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterable

import aiofiles.os


_workspace_cache: dict[str, tuple[str | None, bool]] = {}


async def normalize_path(path: str) -> str:
    """Normalize a path for use as a stable map key without resolving links."""
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    expanded = await expanduser(path)
    return await aiofiles.os.path.abspath(expanded)


async def find_git_worktree(start: str) -> str | None:
    """Walk up from ``start`` looking for a ``.git`` file or directory."""
    try:
        start_path = Path(await normalize_path(start))
        if await aiofiles.os.path.isfile(start_path):
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        return None

    cache_key = str(start_path)
    cached = _workspace_cache.get(cache_key)
    if cached is not None:
        return cached[0]

    current = start_path
    for _ in range(64):
        try:
            if await aiofiles.os.path.exists(current / ".git"):
                resolved = str(current)
                _workspace_cache[cache_key] = (resolved, True)
                return resolved
        except OSError:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    _workspace_cache[cache_key] = (None, False)
    return None


async def is_inside_workspace(path: str, workspace_root: str) -> bool:
    """Return True iff ``path`` is inside or equal to ``workspace_root``."""
    normalized_path = await normalize_path(path)
    normalized_root = await normalize_path(workspace_root)
    if normalized_path == normalized_root:
        return True
    return _normalized_path_is_inside(normalized_path, normalized_root)


def _normalized_path_is_inside(path: str, workspace_root: str) -> bool:
    try:
        common = os.path.commonpath([path, workspace_root])
    except ValueError:
        return False
    return common == workspace_root


async def nearest_root(
    start: str,
    markers: Iterable[str],
    *,
    excludes: Iterable[str] | None = None,
    ceiling: str | None = None,
) -> str | None:
    """Return the nearest ancestor containing a requested project marker."""
    try:
        start_path = Path(await normalize_path(start))
        if await aiofiles.os.path.isfile(start_path):
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        return None
    ceiling_path = Path(await normalize_path(ceiling)) if ceiling else None

    markers_list = list(markers)
    excludes_list = list(excludes) if excludes else []
    current = start_path
    for _ in range(64):
        for exclude in excludes_list:
            try:
                if await aiofiles.os.path.exists(current / exclude):
                    return None
            except OSError:
                continue
        for marker in markers_list:
            try:
                if await aiofiles.os.path.exists(current / marker):
                    return str(current)
            except OSError:
                continue
        if ceiling_path is not None and current == ceiling_path:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return None


async def resolve_workspace_for_file(
    file_path: str,
    *,
    cwd: str | None = None,
) -> tuple[str | None, bool]:
    """Resolve the git workspace root that gates LSP for ``file_path``."""
    active_cwd = cwd or await aiofiles.os.getcwd()
    cwd_root = await find_git_worktree(active_cwd)
    if cwd_root is not None and await is_inside_workspace(file_path, cwd_root):
        return cwd_root, True
    file_root = await find_git_worktree(file_path)
    if file_root is not None:
        return file_root, True
    return None, False


def clear_cache() -> None:
    """Clear cached workspace-resolution results."""
    _workspace_cache.clear()


__all__ = [
    "find_git_worktree",
    "is_inside_workspace",
    "nearest_root",
    "normalize_path",
    "resolve_workspace_for_file",
    "clear_cache",
]
