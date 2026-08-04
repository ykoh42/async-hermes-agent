"""Async file-safety checks shared by model-facing file and media tools."""

from __future__ import annotations

import os
from pathlib import Path

import aiofiles.os


_realpath = aiofiles.os.wrap(os.path.realpath)

_BLOCKED_PROJECT_ENV_BASENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    ".env.staging",
    ".envrc",
}
PROFILE_SCOPED_AREAS = ("skills", "plugins", "cron", "memories")


def _hermes_home_path() -> Path:
    """Return the active profile-aware Hermes home without import cycles."""
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def _hermes_root_path() -> Path:
    """Return the profile-independent Hermes root without import cycles."""
    try:
        from hermes_constants import get_default_hermes_root

        return get_default_hermes_root()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


async def _canonical(path: str | os.PathLike[str]) -> str:
    return await _realpath(os.path.expanduser(os.fspath(path)))


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


async def _hermes_dirs() -> list[str]:
    directories: list[str] = []
    for base in (_hermes_home_path(), _hermes_root_path()):
        try:
            resolved = await _canonical(base)
        except (OSError, ValueError):
            continue
        if resolved not in directories:
            directories.append(resolved)
    return directories


async def _write_denied_paths(home: str) -> set[str]:
    paths = [
        os.path.join(home, ".ssh", "authorized_keys"),
        os.path.join(home, ".ssh", "id_rsa"),
        os.path.join(home, ".ssh", "id_ed25519"),
        os.path.join(home, ".ssh", "config"),
        os.path.join(home, ".netrc"),
        os.path.join(home, ".pgpass"),
        os.path.join(home, ".npmrc"),
        os.path.join(home, ".pypirc"),
        os.path.join(home, ".git-credentials"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]
    for base in (_hermes_home_path(), _hermes_root_path()):
        paths.extend(
            str(base / name)
            for name in (
                ".env",
                ".anthropic_oauth.json",
                "cache/bws_cache.enc.json",
            )
        )
    return {await _canonical(path) for path in paths}


async def _write_denied_prefixes(home: str) -> list[str]:
    paths = [
        os.path.join(home, name)
        for name in (
            ".ssh",
            ".aws",
            ".gnupg",
            ".kube",
            ".docker",
            ".azure",
            ".config/gh",
            ".config/gcloud",
        )
    ] + ["/etc/sudoers.d", "/etc/systemd"]
    return [await _canonical(path) for path in paths]


async def _safe_write_roots() -> set[str]:
    roots = os.getenv("HERMES_WRITE_SAFE_ROOT", "")
    if not roots:
        return set()
    resolved: set[str] = set()
    for root in roots.split(os.pathsep):
        if not root:
            continue
        try:
            resolved.add(await _canonical(root))
        except (OSError, ValueError):
            continue
    return resolved


async def _classify_write_denial(path: str) -> str | None:
    home = await _canonical(os.path.expanduser("~"))
    resolved = await _canonical(path)

    if resolved in await _write_denied_paths(home):
        return "credential"
    if any(_is_within(resolved, prefix) for prefix in await _write_denied_prefixes(home)):
        return "credential"

    for base in await _hermes_dirs():
        state_db = await _canonical(os.path.join(base, "state.db"))
        sessions = await _canonical(os.path.join(base, "sessions"))
        mcp_tokens = await _canonical(os.path.join(base, "mcp-tokens"))
        pairing = await _canonical(os.path.join(base, "pairing"))
        if resolved == state_db or _is_within(resolved, sessions):
            return "state"
        if _is_within(resolved, mcp_tokens) or _is_within(resolved, pairing):
            return "credential"

    safe_roots = await _safe_write_roots()
    if safe_roots and not any(_is_within(resolved, root) for root in safe_roots):
        return "safe_root"
    return None


async def get_write_denied_error(path: str, *, verb: str = "Write") -> str | None:
    """Return a user-facing error when a model-facing write is blocked."""
    denial = await _classify_write_denial(path)
    if denial is None:
        return None
    if denial == "safe_root":
        roots = os.pathsep.join(sorted(await _safe_write_roots()))
        return (
            f"{verb} denied: '{path}' is outside HERMES_WRITE_SAFE_ROOT "
            f"({roots}). Unset the variable or add this path's directory prefix."
        )
    if denial == "state":
        return f"{verb} denied: '{path}' is protected Hermes session state."
    return f"{verb} denied: '{path}' is a protected system/credential file."


async def get_read_block_error(path: str) -> str | None:
    """Return a user-facing error when a model-facing read is blocked."""
    resolved = await _canonical(path)
    hermes_dirs = await _hermes_dirs()

    for base in hermes_dirs:
        hub = await _canonical(os.path.join(base, "skills", ".hub"))
        if _is_within(resolved, hub):
            return (
                f"Access denied: {path} is an internal Hermes cache file and "
                "cannot be read directly to prevent prompt injection. Use the "
                "skills_list or skill_view tools instead."
            )

    credential_names = (
        "auth.json",
        "auth.lock",
        ".anthropic_oauth.json",
        ".env",
        "webhook_subscriptions.json",
        "auth/google_oauth.json",
        "cache/bws_cache.json",
    )
    for base in hermes_dirs:
        for name in credential_names:
            if resolved == await _canonical(os.path.join(base, name)):
                return (
                    f"Access denied: {path} is a Hermes credential store and "
                    "cannot be read directly. Provider tools consume these "
                    "credentials through internal channels."
                )
        mcp_tokens = await _canonical(os.path.join(base, "mcp-tokens"))
        if _is_within(resolved, mcp_tokens):
            return (
                f"Access denied: {path} is a Hermes MCP token file and cannot "
                "be read directly."
            )

    if Path(resolved).name.lower() in _BLOCKED_PROJECT_ENV_BASENAMES:
        return (
            f"Access denied: {path} is a secret-bearing environment file and "
            "cannot be read to prevent credential leakage. If you need to check "
            "the file structure, read .env.example instead."
        )
    return None


async def raise_if_read_blocked(path: str) -> None:
    """Raise ``ValueError`` when :func:`get_read_block_error` blocks *path*."""
    try:
        error = await get_read_block_error(path)
    except (OSError, ValueError):
        return
    if error:
        raise ValueError(error)


def _resolve_active_profile_name() -> str:
    """Return the stable active profile name without filesystem I/O."""
    try:
        home = Path(os.path.abspath(os.path.expanduser(os.fspath(_hermes_home_path()))))
        root = Path(os.path.abspath(os.path.expanduser(os.fspath(_hermes_root_path()))))
    except (OSError, RuntimeError, TypeError, ValueError):
        return "default"
    try:
        relative = home.relative_to(root / "profiles")
    except ValueError:
        return "default"
    return relative.parts[0] if relative.parts else "default"


async def classify_cross_profile_target(path: str) -> dict | None:
    """Describe a write into another profile's scoped state, if any."""
    target = Path(await _canonical(path))
    root = Path(await _canonical(_hermes_root_path()))
    try:
        parts = target.relative_to(root).parts
    except ValueError:
        return None

    if parts and parts[0] in PROFILE_SCOPED_AREAS:
        target_profile, area = "default", parts[0]
    elif len(parts) >= 3 and parts[0] == "profiles" and parts[2] in PROFILE_SCOPED_AREAS:
        target_profile, area = parts[1], parts[2]
    else:
        return None

    active_profile = _resolve_active_profile_name()
    if target_profile == active_profile:
        return None
    return {
        "active_profile": active_profile,
        "target_profile": target_profile,
        "area": area,
        "target_path": str(target),
    }


async def get_cross_profile_warning(path: str) -> str | None:
    """Return the existing soft-guard warning for cross-profile writes."""
    info = await classify_cross_profile_target(path)
    if info is None:
        return None
    return (
        f"Cross-profile write blocked by soft guard: {info['target_path']} "
        f"belongs to Hermes profile {info['target_profile']!r}, but the agent "
        f"is running under profile {info['active_profile']!r}. Editing another "
        f"profile's {info['area']}/ affects that profile's future sessions. "
        "Confirm with the user, then retry with cross_profile=True. "
        "This is defense-in-depth, not a security boundary."
    )
