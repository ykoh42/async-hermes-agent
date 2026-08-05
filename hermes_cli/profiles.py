"""Profile identity helpers for ``HERMES_HOME``-isolated agent instances."""

from __future__ import annotations

import os
import re
from pathlib import Path

from hermes_constants import get_default_hermes_root, get_hermes_home


_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RESERVED_NAMES = frozenset({"hermes", "default", "test", "tmp", "root", "sudo"})


def _get_profiles_root() -> Path:
    """Return the profile directory rooted at the default Hermes home."""
    return get_default_hermes_root() / "profiles"


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


def get_profile_dir(name: str) -> Path:
    """Resolve a profile name to its isolated Hermes home."""
    canonical = normalize_profile_name(name)
    if canonical == "default":
        return get_default_hermes_root()
    return _get_profiles_root() / canonical


def get_active_profile_name() -> str:
    """Infer the current profile name from ``HERMES_HOME``.

    Path normalization is lexical and does not touch the filesystem, so this
    helper remains synchronous even though agent I/O is fully asynchronous.
    """
    resolved = Path(os.path.abspath(get_hermes_home()))
    default_root = Path(os.path.abspath(get_default_hermes_root()))
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
