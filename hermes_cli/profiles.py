"""Profile identity helpers for ``HERMES_HOME``-isolated agent instances."""

from __future__ import annotations

import os
import re
from pathlib import Path

import aiofiles.os

from hermes_constants import get_default_hermes_root, get_hermes_home


_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RESERVED_NAMES = frozenset({"hermes", "default", "test", "tmp", "root", "sudo"})


async def _get_profiles_root() -> Path:
    """Return the profile directory rooted at the default Hermes home."""
    return (await get_default_hermes_root()) / "profiles"


def normalize_profile_name(name: str) -> str:
    """Return the canonical on-disk profile identifier."""
    stripped = str(name).strip()
    if not stripped:
        raise ValueError("profile name cannot be empty")
    if stripped.casefold() == "default":
        return "default"
    return stripped.lower()


def validate_profile_name(name: str) -> None:
    """Raise ``ValueError`` when *name* is not a safe profile identifier."""
    if name == "default":
        return
    if not _PROFILE_ID_RE.fullmatch(name):
        raise ValueError(
            f"Invalid profile name {name!r}. Must match "
            "[a-z0-9][a-z0-9_-]{0,63}"
        )
    if name in _RESERVED_NAMES:
        raise ValueError(f"Profile name {name!r} is reserved")


async def get_profile_dir(name: str) -> Path:
    """Resolve a profile name to its isolated Hermes home."""
    canonical = normalize_profile_name(name)
    if canonical == "default":
        return await get_default_hermes_root()
    return (await _get_profiles_root()) / canonical


def _absolute_path(path: Path, cwd: Path | None) -> Path:
    """Apply ``abspath`` semantics using an already-resolved working directory."""
    value = os.fspath(path)
    if not os.path.isabs(value):
        if cwd is None:
            raise RuntimeError("working directory is unavailable")
        value = os.path.join(cwd, value)
    return Path(os.path.normpath(value))


def _profile_name_from_context(
    active_home: Path,
    default_root: Path | None,
    cwd: Path | None,
) -> str:
    """Classify a profile without performing filesystem I/O."""
    if default_root is None:
        raise RuntimeError("default Hermes root is unavailable")
    resolved = _absolute_path(active_home, cwd)
    default_root = _absolute_path(default_root, cwd)
    if resolved == default_root:
        return "default"

    try:
        relative = resolved.relative_to(default_root / "profiles")
    except ValueError:
        return "custom"

    parts = relative.parts
    if len(parts) == 1 and _PROFILE_ID_RE.fullmatch(parts[0]):
        return parts[0]
    return "custom"


async def _resolve_profile_context() -> tuple[Path, Path | None]:
    """Resolve the process root and cwd used for synchronous profile lookups."""
    default_root = await get_default_hermes_root()
    try:
        cwd = Path(await aiofiles.os.wrap(os.getcwd)())
    except OSError:
        cwd = None
    return default_root, cwd


async def get_active_profile_name() -> str:
    """Infer the current profile name from ``HERMES_HOME``.

    Preserve Hermes' canonical-root handling without resolving paths on the
    event-loop thread.
    """
    default_root, cwd = await _resolve_profile_context()
    return _profile_name_from_context(get_hermes_home(), default_root, cwd)
