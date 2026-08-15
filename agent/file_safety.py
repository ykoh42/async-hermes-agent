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


async def _hermes_root_path() -> Path:
    """Return the profile-independent Hermes root without import cycles."""
    try:
        from hermes_constants import get_default_hermes_root

        return await get_default_hermes_root()
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
    for base in (_hermes_home_path(), await _hermes_root_path()):
        try:
            resolved = await _canonical(base)
        except Exception:
            continue
        if resolved not in directories:
            directories.append(resolved)
    return directories


async def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    paths = [
        os.path.join(home, ".ssh", "authorized_keys"),
        os.path.join(home, ".ssh", "id_rsa"),
        os.path.join(home, ".ssh", "id_ed25519"),
        os.path.join(home, ".netrc"),
        os.path.join(home, ".pgpass"),
        os.path.join(home, ".npmrc"),
        os.path.join(home, ".pypirc"),
        os.path.join(home, ".git-credentials"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]
    for base in (_hermes_home_path(), await _hermes_root_path()):
        paths.extend(
            str(base / name)
            for name in (
                ".env",
                ".anthropic_oauth.json",
                "cache/bws_cache.enc.json",
            )
        )
    return {await _canonical(path) for path in paths}


async def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    paths = [
        os.path.join(home, ".ssh"),
        os.path.join(home, ".aws"),
        os.path.join(home, ".gnupg"),
        os.path.join(home, ".kube"),
        "/etc/sudoers.d",
        "/etc/systemd",
        os.path.join(home, ".docker"),
        os.path.join(home, ".azure"),
        os.path.join(home, ".config", "gh"),
        os.path.join(home, ".config", "gcloud"),
    ]
    return [await _canonical(path) + os.sep for path in paths]


async def build_write_approval_paths(home: str) -> set[str]:
    """Return non-credential paths that require an interactive write approval."""
    return {await _canonical(os.path.join(home, ".ssh", "config"))}


async def get_safe_write_roots() -> set[str]:
    """Return resolved ``HERMES_WRITE_SAFE_ROOT`` paths."""
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


async def _classify_write_denial(path: str) -> str | bool | None:
    home = await _canonical(os.path.expanduser("~"))
    resolved = await _canonical(path)

    # ``~/.ssh/config`` is routine to edit but can contain ProxyCommand or
    # Match exec directives.  It is allowed past the denylist only so the
    # model-facing file tool can request the native async approval gate.
    if resolved in await build_write_approval_paths(home):
        return None

    if resolved in await build_write_denied_paths(home):
        return "credential"
    if any(
        resolved.startswith(prefix)
        for prefix in await build_write_denied_prefixes(home)
    ):
        return "credential"

    for base in await _hermes_dirs():
        state_db = await _canonical(os.path.join(base, "state.db"))
        sessions = await _canonical(os.path.join(base, "sessions"))
        mcp_tokens = await _canonical(os.path.join(base, "mcp-tokens"))
        pairing = await _canonical(os.path.join(base, "pairing"))
        if resolved == state_db or _is_within(resolved, sessions):
            return True
        if _is_within(resolved, mcp_tokens) or _is_within(resolved, pairing):
            return "credential"

    safe_roots = await get_safe_write_roots()
    if safe_roots and not any(_is_within(resolved, root) for root in safe_roots):
        return "safe_root"
    return None


async def is_write_denied(path: str) -> bool:
    """Return True if path is blocked by the write denylist or safe root."""
    return await _classify_write_denial(path) is not None


async def get_write_denied_error(path: str, *, verb: str = "Write") -> str | None:
    """Return a user-facing error when a model-facing write is blocked."""
    denial = await _classify_write_denial(path)
    if denial is None:
        return None
    if denial == "safe_root":
        roots = os.pathsep.join(sorted(await get_safe_write_roots()))
        return (
            f"{verb} denied: '{path}' is outside HERMES_WRITE_SAFE_ROOT "
            f"({roots}). Unset the variable or add this path's directory prefix."
        )
    return f"{verb} denied: '{path}' is a protected system/credential file."


async def is_write_approval_required(path: str) -> bool:
    """Return whether a path needs human approval before a model write."""
    home = await _canonical(os.path.expanduser("~"))
    resolved = await _canonical(path)
    return resolved in await build_write_approval_paths(home)


async def get_read_block_error(path: str) -> str | None:
    """Return a user-facing error when a model-facing read is blocked."""
    resolved = await _canonical(path)
    hermes_dirs = await _hermes_dirs()

    for base in hermes_dirs:
        hub = await _canonical(os.path.join(base, "skills", ".hub"))
        if _is_within(resolved, hub):
            return (
                f"Access denied: {path} is an internal Hermes cache file "
                "and cannot be read directly to prevent prompt injection. "
                "Use the "
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
                    f"Access denied: {path} is a Hermes credential store "
                    "and cannot be read directly. Provider tools consume "
                    "these credentials through internal channels. "
                    "(Defense-in-depth — not a security boundary; the "
                    "terminal tool can still bypass.)"
                )
        mcp_tokens = await _canonical(os.path.join(base, "mcp-tokens"))
        if resolved == mcp_tokens:
            return (
                f"Access denied: {path} is the Hermes MCP token directory "
                "and cannot be read directly. (Defense-in-depth — not a "
                "security boundary; the terminal tool can still bypass.)"
            )
        if _is_within(resolved, mcp_tokens):
            return (
                f"Access denied: {path} is a Hermes MCP token file "
                "and cannot be read directly. (Defense-in-depth — not a "
                "security boundary; the terminal tool can still bypass.)"
            )

    if Path(resolved).name.lower() in _BLOCKED_PROJECT_ENV_BASENAMES:
        return (
            f"Access denied: {path} is a secret-bearing environment file and "
            "cannot be read to prevent credential leakage. "
            "If you need to check the file structure, read .env.example instead. "
            "(Defense-in-depth — not a security boundary; the terminal tool can "
            "still bypass.)"
        )
    return None


async def raise_if_read_blocked(path: str) -> None:
    """Raise ``ValueError`` when :func:`get_read_block_error` blocks *path*."""
    try:
        error = await get_read_block_error(path)
    except Exception:
        return
    if error:
        raise ValueError(error)


async def _resolve_active_profile_name() -> str:
    """Return the active profile name derived from ``HERMES_HOME``."""
    try:
        home = Path(await _canonical(_hermes_home_path()))
        root = Path(await _canonical(await _hermes_root_path()))
    except (OSError, RuntimeError):
        return "default"
    try:
        relative = home.relative_to(root / "profiles")
    except ValueError:
        return "default"
    return relative.parts[0] if relative.parts else "default"


async def classify_cross_profile_target(path: str) -> dict | None:
    """Describe a write into another profile's scoped state, if any."""
    try:
        target = Path(await _canonical(path))
        root = Path(await _canonical(await _hermes_root_path()))
    except (OSError, RuntimeError):
        return None
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

    active_profile = await _resolve_active_profile_name()
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
        f"belongs to Hermes profile {info['target_profile']!r}, but the "
        f"agent is running under profile {info['active_profile']!r}. "
        f"Editing another profile's {info['area']}/ will affect that "
        f"profile's future sessions, not the one you are currently in. "
        f"Confirm with the user before proceeding. To bypass this guard "
        f"after explicit user direction, retry the call with "
        f"``cross_profile=True``. (Defense-in-depth — not a security "
        f"boundary; the terminal tool can still bypass.)"
    )


def _find_sandbox_mirror_segments(parts: tuple) -> int | None:
    """Return the index of the inner ``.hermes`` in a sandbox-mirror path."""
    for index, part in enumerate(parts):
        if part != "sandboxes":
            continue
        if index + 5 >= len(parts):
            continue
        if parts[index + 3] == "home" and parts[index + 4] == ".hermes":
            return index + 4
    return None


async def classify_sandbox_mirror_target(path: str) -> dict | None:
    """Classify a write target as a sandbox mirror of Hermes state."""
    try:
        target = Path(await _canonical(path))
    except (OSError, RuntimeError, ValueError):
        return None

    parts = target.parts
    inner_index = _find_sandbox_mirror_segments(parts)
    if inner_index is None:
        return None

    mirror_root = str(Path(*parts[: inner_index + 1]))
    inner_path = (
        str(Path(*parts[inner_index + 1 :]))
        if inner_index + 1 < len(parts)
        else ""
    )
    return {
        "target_path": str(target),
        "mirror_root": mirror_root,
        "inner_path": inner_path,
    }


async def get_sandbox_mirror_warning(path: str) -> str | None:
    """Return a model-facing warning when ``path`` lands in a sandbox mirror."""
    info = await classify_sandbox_mirror_target(path)
    if info is None:
        return None
    return (
        f"Sandbox-mirror write blocked by soft guard: {info['target_path']} "
        f"sits under {info['mirror_root']!r}, which is a per-task mirror "
        f"created by a non-local terminal backend (docker/daytona/etc.). "
        f"Writes here land on a copy that the host Hermes process never "
        f"reads — the authoritative file is likely {info['inner_path']!r} "
        f"under the real HERMES_HOME. Use the host-side tool for "
        f"authoritative state (e.g. ``memory`` for memories), or address "
        f"the host path directly. To bypass this guard after explicit "
        f"user direction, retry the call with ``cross_profile=True``. "
        f"(Defense-in-depth — not a security boundary; the terminal tool "
        f"can still bypass.)"
    )


async def classify_container_mirror_target(
    path: str,
    mirror_prefix: str | None = None,
) -> dict | None:
    """Classify a write target as a container-side sandbox mirror."""
    if not mirror_prefix:
        return None
    try:
        target = Path(await _canonical(path))
        mirror = Path(await _canonical(mirror_prefix))
        inner = target.relative_to(mirror)
    except (OSError, RuntimeError, ValueError):
        return None
    return {
        "target_path": str(target),
        "mirror_root": str(mirror),
        "inner_path": inner.as_posix(),
    }


async def get_container_mirror_warning(
    path: str,
    mirror_prefix: str | None = None,
) -> str | None:
    """Return a warning when ``path`` lands in a container sandbox mirror."""
    info = await classify_container_mirror_target(path, mirror_prefix)
    if info is None:
        return None
    return (
        f"Sandbox-mirror write blocked by soft guard: {info['target_path']} "
        f"sits under {info['mirror_root']!r}, which is the container's "
        f"bind-mounted home — a per-task mirror that the host Hermes "
        f"process never reads. The authoritative file is "
        f"{info['inner_path']!r} under the real HERMES_HOME. Use the "
        f"host-side tool for authoritative state (e.g. ``memory`` for "
        f"memories), or address the host path directly. To bypass after "
        f"explicit user direction, retry with ``cross_profile=True``. "
        f"(Defense-in-depth — not a security boundary; the terminal tool "
        f"can still bypass.)"
    )
