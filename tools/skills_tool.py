#!/usr/bin/env python3
"""
Skills Tool Module

This module provides tools for listing and viewing skill documents.
Skills are organized as directories containing a SKILL.md file (the main instructions)
and optional supporting files like references, templates, and examples.

Inspired by Anthropic's Claude Skills system with progressive disclosure architecture:
- Metadata (name ≤64 chars, description ≤1024 chars) - shown in skills_list
- Full Instructions - loaded via skill_view when needed
- Linked Files (references, templates) - loaded on demand

Directory Structure:
    skills/
    ├── my-skill/
    │   ├── SKILL.md           # Main instructions (required)
    │   ├── references/        # Supporting documentation
    │   │   ├── api.md
    │   │   └── examples.md
    │   ├── templates/         # Templates for output
    │   │   └── template.md
    │   └── assets/            # Supplementary files (agentskills.io standard)
    └── category/              # Category folder for organization
        └── another-skill/
            └── SKILL.md

SKILL.md Format (YAML Frontmatter, agentskills.io compatible):
    ---
    name: skill-name              # Required, max 64 chars
    description: Brief description # Required, max 1024 chars
    version: 1.0.0                # Optional
    license: MIT                  # Optional (agentskills.io)
    platforms: [macos]            # Optional — restrict to specific OS platforms
                                  #   Valid: macos, linux, windows
                                  #   Omit to load on all platforms (default)
    prerequisites:                # Optional — legacy runtime requirements
      env_vars: [API_KEY]         #   Legacy env var names are normalized into
                                  #   required_environment_variables on load.
      commands: [curl, jq]        #   Command checks remain advisory only.
    compatibility: Requires X     # Optional (agentskills.io)
    metadata:                     # Optional, arbitrary key-value (agentskills.io)
      hermes:
        tags: [fine-tuning, llm]
        related_skills: [peft, lora]
    ---

    # Skill Title

    Full instructions and content here...

Available tools:
- skills_list: List skills with metadata (progressive disclosure tier 1)
- skill_view: Load full skill content (progressive disclosure tier 2-3)

Usage:
    from tools.skills_tool import skills_list, skill_view, check_skills_requirements

    # List all skills (returns metadata only - token efficient)
    result = skills_list()

    # View a skill's main content (loads full instructions)
    content = skill_view("axolotl")

    # View a reference file within a skill (loads linked file)
    content = skill_view("axolotl", "references/dataset-formats.md")
"""

import asyncio
import concurrent.futures.thread as _thread_backend_bootstrap  # noqa: F401
import contextvars
import inspect
import json
import logging
import threading
import time

from hermes_constants import get_hermes_home, display_hermes_home
import os
import re
from enum import Enum
from collections.abc import MutableMapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import aiofiles
import aiofiles.os

from agent import skill_preprocessing as _skill_preprocessing
from agent.secret_scope import UnscopedSecretError, get_secret
from hermes_cli import managed_scope as _managed_scope_bootstrap  # noqa: F401
from tools import path_security as _path_security
from tools import skill_manager_tool as _skill_manager_bootstrap  # noqa: F401
from tools import skill_provenance as _skill_provenance_bootstrap  # noqa: F401
from tools.registry import registry, tool_error
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS,
    get_disabled_skill_names as _get_disabled_skill_names,
    get_external_skills_dirs as _external_skills_dirs,
    get_project_skills_dirs as _project_skills_dirs,
    iter_skill_index_files as _iter_skill_index_files,
)

logger = logging.getLogger(__name__)
_realpath = aiofiles.os.wrap(os.path.realpath)
_secret_capture_callback = None

# Per-session skill discovery cache.  _find_all_skills() re-reads every
# SKILL.md on every call; with hundreds of skills this is wasteful.
# Cache validation (mirrors hermes_cli/profiles.py::_count_skills, d5eee133e):
#   - signature = per-dir max mtime of the dir AND its immediate children
#     (one scandir per dir; catches skill add/remove inside categories,
#     which does NOT bump the root dir's mtime), plus the disabled-set
#     (config-driven — changes with no filesystem mtime bump at all)
#   - a short TTL bounds staleness from in-place SKILL.md edits, which
#     bump only the file's mtime, invisible to any directory signature.
# skip_disabled True/False are cached separately.
_SKILLS_CACHE: dict = {}          # {cache_key: (signature, timestamp, skills_list)}
_SKILLS_CACHE_TTL_SECONDS = 30.0
_SKILLS_CACHE_KEY_DISABLED = "with_disabled"
_SKILLS_CACHE_KEY_FILTERED = "filtered"


async def _skills_scan_signature(dirs_to_scan, disabled) -> tuple:
    """Cheap async change-signature for the skill scan inputs.

    Only the directory and its immediate child directories are inspected;
    every filesystem operation is awaited through ``aiofiles.os`` so a large
    skills catalog cannot pause the agent event loop.
    """
    from agent import skill_utils as _skill_utils

    platform = getattr(getattr(_skill_utils, "sys", None), "platform", "")
    sig = []
    for directory in dirs_to_scan:
        try:
            stat_result = await aiofiles.os.stat(directory)
            newest = stat_result.st_mtime
        except OSError:
            continue
        try:
            for name in await aiofiles.os.listdir(directory):
                candidate = directory / name
                try:
                    if not await aiofiles.os.path.isdir(candidate):
                        continue
                    child_stat = await aiofiles.os.stat(candidate)
                    newest = max(newest, child_stat.st_mtime)
                except OSError:
                    continue
        except OSError:
            pass
        sig.append((str(directory), newest))
    return (tuple(sig), frozenset(disabled), platform)


# All skills live in ~/.hermes/skills/ (seeded from bundled skills/ on install).
# This is the single source of truth -- agent edits, hub installs, and bundled
# skills all coexist here without polluting the git repo.
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Return the active profile's skills directory at call time.

    Some long-lived runtimes import this module before the active profile has
    set HERMES_HOME. Keep the legacy SKILLS_DIR module attribute for tests and
    external patchers, but when it has not been patched, resolve from the live
    profile-scoped HERMES_HOME on every call.
    """
    configured = Path(SKILLS_DIR)
    if configured != _SKILLS_DIR_AT_IMPORT:
        return configured
    return get_hermes_home() / "skills"


async def _iter_files(directory: Path):
    """Yield regular files below *directory* without blocking the loop."""
    try:
        names = await aiofiles.os.listdir(directory)
    except OSError:
        return
    directories: list[Path] = []
    for name in names:
        candidate = directory / name
        try:
            if await aiofiles.os.path.isdir(candidate):
                directories.append(candidate)
            elif await aiofiles.os.path.isfile(candidate):
                yield candidate
        except OSError:
            continue
    for child in sorted(directories, key=lambda path: path.name):
        async for result in _iter_files(child):
            yield result


async def _linked_skill_files(skill_dir: Path) -> dict[str, list[str]]:
    """Build the retained linked-file catalog with upstream filters."""
    linked: dict[str, list[str]] = {}
    references = skill_dir / "references"
    if await aiofiles.os.path.isdir(references):
        linked_references = [
            str(path.relative_to(skill_dir))
            async for path in _iter_files(references)
            if path.parent == references and path.suffix == ".md"
        ]
        if linked_references:
            linked["references"] = linked_references

    templates = skill_dir / "templates"
    template_suffixes = {".md", ".py", ".yaml", ".yml", ".json", ".tex", ".sh"}
    if await aiofiles.os.path.isdir(templates):
        linked_templates = [
            str(path.relative_to(skill_dir))
            async for path in _iter_files(templates)
            if path.suffix in template_suffixes
        ]
        if linked_templates:
            linked["templates"] = linked_templates

    assets = skill_dir / "assets"
    if await aiofiles.os.path.isdir(assets):
        linked_assets = [
            str(path.relative_to(skill_dir))
            async for path in _iter_files(assets)
        ]
        if linked_assets:
            linked["assets"] = linked_assets

    scripts = skill_dir / "scripts"
    script_suffixes = {".py", ".sh", ".bash", ".js", ".ts", ".rb"}
    if await aiofiles.os.path.isdir(scripts):
        linked_scripts = [
            str(path.relative_to(skill_dir))
            async for path in _iter_files(scripts)
            if path.parent == scripts and path.suffix in script_suffixes
        ]
        if linked_scripts:
            linked["scripts"] = linked_scripts
    return linked


async def _available_skill_files(skill_dir: Path) -> dict[str, list[str]]:
    """Return the upstream missing-file recovery inventory."""
    available: dict[str, list[str]] = {
        "references": [],
        "templates": [],
        "assets": [],
        "scripts": [],
        "other": [],
    }
    other_suffixes = {".md", ".py", ".yaml", ".yml", ".json", ".tex", ".sh"}
    async for path in _iter_files(skill_dir):
        if path.name == "SKILL.md":
            continue
        relative = str(path.relative_to(skill_dir))
        top_level = relative.split("/", 1)[0]
        if top_level in {"references", "templates", "assets", "scripts"}:
            available[top_level].append(relative)
        elif path.suffix in other_suffixes:
            available["other"].append(relative)
    return {key: value for key, value in available.items() if value}


# Anthropic-recommended limits for progressive disclosure efficiency
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# Platform identifiers for the 'platforms' frontmatter field.
# Maps user-friendly names to sys.platform prefixes.
_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _skill_lookup_path_error(name: str) -> str | None:
    """Return an error if a local skill lookup *name* can escape search roots.

    The skill ``name`` is joined onto each trusted search dir to build the
    on-disk lookup path, so it must stay relative and free of ``..`` segments —
    otherwise ``name="../outside"`` or an absolute path could select a skill
    (and read files) outside the skills directory. Mirrors the ``file_path``
    validation done later via ``tools.path_security``. We also reject Windows
    drive paths (e.g. ``C:\\skills``), whose ``:`` would otherwise be misread as
    a plugin namespace separator.
    """
    if not isinstance(name, str):
        return "Skill name must be a string."
    candidate = name.strip()
    if (
        PurePosixPath(candidate).is_absolute()
        or PureWindowsPath(candidate).is_absolute()
        or PureWindowsPath(candidate).drive
    ):
        return "Skill name must be a relative path within the skills directory."
    if _path_security.has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


async def load_env() -> dict[str, str]:
    """Read the profile environment file without blocking the event loop."""
    env_path = get_hermes_home() / ".env"
    if not await aiofiles.os.path.isfile(env_path):
        return {}

    async with aiofiles.open(env_path, encoding="utf-8") as handle:
        contents = await handle.read()

    env_vars: dict[str, str] = {}
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


def _required_environment_variable_available(
    name: str,
    env_snapshot: dict[str, str],
) -> bool:
    """Preserve dotenv-first readiness without crossing profile scopes."""
    return bool(env_snapshot.get(name) or get_secret(name))


class SkillReadinessStatus(str, Enum):  # noqa: UP042 - preserve legacy str() output
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


def set_secret_capture_callback(callback) -> None:
    """Register a native-async host callback for required skill secrets."""
    global _secret_capture_callback
    if callback is not None and not (
        inspect.iscoroutinefunction(callback)
        or inspect.iscoroutinefunction(getattr(callback, "__call__", None))
    ):
        raise TypeError("skill secret capture callback must be async")
    _secret_capture_callback = callback


_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new instructions:",
    "system prompt:",
    "<system>",
    "]]>",
]


def skill_matches_platform(frontmatter: dict[str, Any]) -> bool:
    """Check if a skill is compatible with the current OS platform.

    Delegates to ``agent.skill_utils.skill_matches_platform`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import skill_matches_platform as _impl
    return _impl(frontmatter)


async def skill_matches_environment(frontmatter: dict[str, Any]) -> bool:
    """Check if a skill is relevant to the current runtime environment.

    Delegates to ``agent.skill_utils.skill_matches_environment`` — kept here
    as a public re-export so existing callers don't need updating. This is an
    offer-time relevance gate (kanban/docker/s6), NOT a hard-compatibility gate;
    explicit skill loads bypass it.
    """
    from agent.skill_utils import skill_matches_environment as _impl
    return await _impl(frontmatter)


def _normalize_prerequisite_values(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(
    frontmatter: dict[str, Any],
) -> tuple[list[str], list[str]]:
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _normalize_setup_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
    setup = frontmatter.get("setup")
    if not isinstance(setup, dict):
        return {"help": None, "collect_secrets": []}

    help_text = setup.get("help")
    normalized_help = (
        str(help_text).strip()
        if isinstance(help_text, str) and help_text.strip()
        else None
    )

    collect_secrets_raw = setup.get("collect_secrets")
    if isinstance(collect_secrets_raw, dict):
        collect_secrets_raw = [collect_secrets_raw]
    if not isinstance(collect_secrets_raw, list):
        collect_secrets_raw = []

    collect_secrets: list[dict[str, Any]] = []
    for item in collect_secrets_raw:
        if not isinstance(item, dict):
            continue

        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue

        prompt = str(item.get("prompt") or f"Enter value for {env_var}").strip()
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()

        entry: dict[str, Any] = {
            "env_var": env_var,
            "prompt": prompt,
            "secret": bool(item.get("secret", True)),
        }
        if provider_url:
            entry["provider_url"] = provider_url
        collect_secrets.append(entry)

    return {
        "help": normalized_help,
        "collect_secrets": collect_secrets,
    }


def _get_required_environment_variables(
    frontmatter: dict[str, Any],
    legacy_env_vars: list[str] | None = None,
) -> list[dict[str, Any]]:
    setup = _normalize_setup_metadata(frontmatter)
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, dict):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    required: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen:
            return
        if not _ENV_VAR_NAME_RE.match(env_name):
            return

        normalized: dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip(),
        }

        help_text = (
            entry.get("help")
            or entry.get("provider_url")
            or entry.get("url")
            or setup.get("help")
        )
        if isinstance(help_text, str) and help_text.strip():
            normalized["help"] = help_text.strip()

        required_for = entry.get("required_for")
        if isinstance(required_for, str) and required_for.strip():
            normalized["required_for"] = required_for.strip()

        if entry.get("optional"):
            normalized["optional"] = True

        seen.add(env_name)
        required.append(normalized)

    for item in required_raw:
        if isinstance(item, str):
            _append_required({"name": item})
            continue
        if isinstance(item, dict):
            _append_required(item)

    for item in setup["collect_secrets"]:
        _append_required(
            {
                "name": item.get("env_var"),
                "prompt": item.get("prompt"),
                "help": item.get("provider_url") or setup.get("help"),
            }
        )

    if legacy_env_vars is None:
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    for env_var in legacy_env_vars:
        _append_required({"name": env_var})

    return required


def _build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: list[str],
    setup_help: str | None = None,
) -> str | None:
    if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
        missing_str = ", ".join(missing) if missing else "required prerequisites"
        note = f"Setup needed before using this skill: missing {missing_str}."
        if setup_help:
            return f"{note} {setup_help}"
        return note
    return None


def _is_gateway_surface() -> bool:
    from gateway.session_context import get_session_env
    from utils import env_var_enabled

    return env_var_enabled("HERMES_GATEWAY_SESSION") or bool(
        get_session_env("HERMES_SESSION_PLATFORM")
    )


def _gateway_setup_hint() -> str:
    from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE

    return GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE


async def _capture_required_environment_variables(
    skill_name: str,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not missing_entries:
        return {
            "missing_names": [],
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }
    missing_names = [entry["name"] for entry in missing_entries]
    from utils import env_var_enabled

    if _is_gateway_surface() and not env_var_enabled("HERMES_INTERACTIVE"):
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": _gateway_setup_hint(),
        }
    if _secret_capture_callback is None:
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    setup_skipped = False
    remaining_names: list[str] = []
    for entry in missing_entries:
        metadata = {"skill_name": skill_name}
        if entry.get("help"):
            metadata["help"] = entry["help"]
        if entry.get("required_for"):
            metadata["required_for"] = entry["required_for"]
        try:
            callback_result = await _secret_capture_callback(
                entry["name"],
                entry["prompt"],
                metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Secret capture callback failed for %s",
                entry["name"],
                exc_info=True,
            )
            callback_result = {
                "success": False,
                "stored_as": entry["name"],
                "validated": False,
                "skipped": True,
            }
        success = isinstance(callback_result, dict) and bool(
            callback_result.get("success")
        )
        skipped = isinstance(callback_result, dict) and bool(
            callback_result.get("skipped")
        )
        if success and not skipped:
            continue
        setup_skipped = True
        remaining_names.append(entry["name"])
    return {
        "missing_names": remaining_names,
        "setup_skipped": setup_skipped,
        "gateway_setup_hint": None,
    }


async def _missing_required_credential_files(
    frontmatter: dict[str, Any],
) -> list[str]:
    """Return unavailable/refused skill credential paths without remote mounts."""
    entries = frontmatter.get("required_credential_files", [])
    if not isinstance(entries, list):
        return []

    hermes_home = get_hermes_home()
    try:
        resolved_home = Path(await _realpath(hermes_home))
    except OSError:
        resolved_home = hermes_home
    missing: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            relative_path = entry.strip()
        elif isinstance(entry, dict):
            relative_path = str(entry.get("path") or entry.get("name") or "").strip()
        else:
            continue
        if not relative_path:
            continue
        if os.path.isabs(relative_path):
            missing.append(relative_path)
            continue
        candidate = hermes_home / relative_path
        try:
            resolved = Path(await _realpath(candidate))
            resolved.relative_to(resolved_home)
        except (OSError, ValueError):
            missing.append(relative_path)
            continue
        if not await aiofiles.os.path.isfile(resolved):
            missing.append(relative_path)
            continue
        try:
            from agent.file_safety import get_read_block_error

            denied = await get_read_block_error(str(resolved))
        except Exception:
            logger.exception(
                "credential_files: refusing %r because the read guard failed",
                relative_path,
            )
            denied = "read guard unavailable"
        if denied:
            missing.append(relative_path)
    return missing


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Delegates to ``agent.skill_utils.parse_frontmatter`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import parse_frontmatter
    return parse_frontmatter(content)


async def _get_category_from_path(skill_path: Path) -> str | None:
    """
    Extract category from skill path based on directory structure.

    For paths like: ~/.hermes/skills/mlops/axolotl/SKILL.md -> "mlops"
    Also works for external skill dirs configured via skills.external_dirs.
    """
    # Try the active profile skills dir first (respects monkeypatching in tests),
    # then fall back to external dirs from config.
    dirs_to_check = [_skills_dir()]
    dirs_to_check.extend(await _external_skills_dirs())
    for skills_dir in dirs_to_check:
        try:
            rel_path = skill_path.relative_to(skills_dir)
            parts = rel_path.parts
            if len(parts) >= 3:
                return parts[0]
        except ValueError:
            continue
    return None


def _parse_tags(tags_value) -> list[str]:
    """
    Parse tags from frontmatter value.

    Handles:
    - Already-parsed list (from yaml.safe_load): [tag1, tag2]
    - String with brackets: "[tag1, tag2]"
    - Comma-separated string: "tag1, tag2"

    Args:
        tags_value: Raw tags value — may be a list or string

    Returns:
        List of tag strings
    """
    if not tags_value:
        return []

    # yaml.safe_load already returns a list for [tag1, tag2]
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]

    # String fallback — handle bracket-wrapped or comma-separated
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]

    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]



async def _is_skill_disabled(name: str, platform: str | None = None) -> bool:
    """Check if a skill is disabled in config.

    Resolves the active platform from (in order of precedence):
    1. Explicit ``platform`` argument
    2. ``HERMES_PLATFORM`` environment variable
    3. ``HERMES_SESSION_PLATFORM`` from gateway session context
    """
    return name in await _get_disabled_skill_names(platform)


def _sort_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))




async def _read_skill_text(path: Path, *, limit: int | None = None) -> str:
    """Read one skill file through the async file boundary."""
    async with aiofiles.open(path, encoding="utf-8") as handle:
        return await (handle.read(limit) if limit is not None else handle.read())


async def _find_all_skills(*, skip_disabled: bool = False) -> list[dict[str, Any]]:
    """List local and external skills through the async file boundary.

    Results are cached per discovery signature and returned as copies because
    callers may annotate individual skill dictionaries.
    """
    cache_key = _SKILLS_CACHE_KEY_DISABLED if skip_disabled else _SKILLS_CACHE_KEY_FILTERED
    disabled = set() if skip_disabled else await _get_disabled_skill_names()
    roots: list[Path] = await _project_skills_dirs()
    active = _skills_dir()
    if await aiofiles.os.path.isdir(active):
        roots.append(active)
    roots.extend(await _external_skills_dirs())
    signature = await _skills_scan_signature(roots, disabled)
    now = time.monotonic()
    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature and now - cached[1] < _SKILLS_CACHE_TTL_SECONDS:
        return [dict(skill) for skill in cached[2]]

    skills: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for root in roots:
        async for skill_md in _iter_skill_index_files(root, "SKILL.md"):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                content = await _read_skill_text(skill_md, limit=4000)
                frontmatter, body = _parse_frontmatter(content)
                if (
                    not skill_matches_platform(frontmatter)
                    or not await skill_matches_environment(frontmatter)
                ):
                    continue
                name = str(frontmatter.get("name", skill_md.parent.name))[:MAX_NAME_LENGTH]
                if name in seen_names or name in disabled:
                    continue
                description = str(frontmatter.get("description", ""))
                if not description:
                    description = next(
                        (
                            line.strip()
                            for line in body.splitlines()
                            if line.strip() and not line.lstrip().startswith("#")
                        ),
                        "",
                    )
                if len(description) > MAX_DESCRIPTION_LENGTH:
                    description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."
                seen_names.add(name)
                skills.append({
                    "name": name,
                    "description": description,
                    "category": await _get_category_from_path(skill_md),
                })
            except (OSError, UnicodeDecodeError, PermissionError) as exc:
                logger.debug("Failed to read skill file %s: %s", skill_md, exc)
            except Exception as exc:
                logger.debug(
                    "Skipping skill at %s: failed to parse: %s", skill_md, exc, exc_info=True
                )

    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(skill) for skill in skills]


async def skills_list(category: str | None = None, task_id: str | None = None) -> str:
    """Native async implementation behind the ``skills_list`` tool."""
    try:
        active_skills_dir = _skills_dir()
        if not await aiofiles.os.path.exists(active_skills_dir):
            await aiofiles.os.makedirs(active_skills_dir, exist_ok=True)
            return json.dumps({
                "success": True,
                "skills": [],
                "categories": [],
                "message": f"No skills found. Skills directory created at {display_hermes_home()}/skills/",
            }, ensure_ascii=False)
        skills = await _find_all_skills()
        if not skills:
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found in skills/ directory.",
                },
                ensure_ascii=False,
            )
        if category:
            skills = [skill for skill in skills if skill.get("category") == category]
        skills = _sort_skills(skills)
        return json.dumps({
            "success": True,
            "skills": skills,
            "categories": sorted({skill["category"] for skill in skills if skill.get("category")}),
            "count": len(skills),
            "hint": "Use skill_view(name) to see full content, tags, and linked files",
        }, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


async def _serve_plugin_skill(
    skill_md: Path,
    namespace: str,
    bare: str,
    *,
    preprocess: bool = True,
    session_id: str | None = None,
) -> str:
    """Read a registered plugin skill through the native async file boundary."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.plugins import _get_disabled_plugins, get_plugin_manager

    config = await load_config_readonly()
    if namespace in await _get_disabled_plugins(config):
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Plugin '{namespace}' is disabled. "
                    f"Re-enable with: hermes plugins enable {namespace}"
                ),
            },
            ensure_ascii=False,
        )

    try:
        content = await _read_skill_text(skill_md)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to read skill '{namespace}:{bare}': {exc}",
            },
            ensure_ascii=False,
        )

    try:
        frontmatter, _ = _parse_frontmatter(content)
    except Exception:
        frontmatter = {}

    if not skill_matches_platform(frontmatter):
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Skill '{namespace}:{bare}' is not supported on this platform."
                ),
                "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
            },
            ensure_ascii=False,
        )

    if any(pattern in content.lower() for pattern in _INJECTION_PATTERNS):
        logger.warning(
            "Plugin skill '%s:%s' contains patterns that may indicate prompt injection",
            namespace,
            bare,
        )

    description = str(frontmatter.get("description", ""))
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

    try:
        siblings = [
            sibling
            for sibling in get_plugin_manager().list_plugin_skills(namespace)
            if sibling != bare
        ]
        if siblings:
            banner = (
                f"[Bundle context: This skill is part of the '{namespace}' plugin.\n"
                f"Sibling skills: {', '.join(siblings)}.\n"
                "Use qualified form to invoke siblings "
                f"(e.g. {namespace}:{siblings[0]}).]\n\n"
            )
        else:
            banner = (
                f"[Bundle context: This skill is part of the '{namespace}' "
                "plugin.]\n\n"
            )
    except Exception:
        banner = ""

    rendered_content = content
    if preprocess:
        try:
            rendered_content = await _skill_preprocessing.preprocess_skill_content(
                content,
                skill_md.parent,
                session_id=session_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "Could not preprocess plugin skill %s:%s",
                namespace,
                bare,
                exc_info=True,
            )

    return json.dumps(
        {
            "success": True,
            "name": f"{namespace}:{bare}",
            "content": f"{banner}{rendered_content}" if banner else rendered_content,
            "description": description,
            "linked_files": None,
            "readiness_status": SkillReadinessStatus.AVAILABLE.value,
        },
        ensure_ascii=False,
    )


async def skill_view(
    name: str,
    file_path: str | None = None,
    task_id: str | None = None,
    preprocess: bool = True,
) -> str:
    """View a local, externally configured, or plugin-provided skill."""
    try:
        lookup_error = _skill_lookup_path_error(name)
        if lookup_error:
            return json.dumps(
                {
                    "success": False,
                    "error": lookup_error,
                    "hint": "Use a skill name or relative path within the skills directory.",
                },
                ensure_ascii=False,
            )
        local_category_name: str | None = None
        if ":" in name:
            from agent.skill_utils import is_valid_namespace, parse_qualified_name
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            namespace, bare = parse_qualified_name(name)
            if not is_valid_namespace(namespace):
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"Invalid namespace '{namespace}' in '{name}'. "
                            "Namespaces must match [a-zA-Z0-9_-]+."
                        ),
                    },
                    ensure_ascii=False,
                )

            await discover_plugins()
            manager = get_plugin_manager()
            plugin_skill_md = manager.find_plugin_skill(name)
            if plugin_skill_md is not None:
                if not await aiofiles.os.path.isfile(plugin_skill_md):
                    manager.remove_plugin_skill(name)
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Skill '{name}' file no longer exists at "
                                f"{plugin_skill_md}. The registry entry has been "
                                "cleaned up — try again after the plugin is reloaded."
                            ),
                        },
                        ensure_ascii=False,
                    )
                return await _serve_plugin_skill(
                    plugin_skill_md,
                    namespace,
                    bare,
                    preprocess=preprocess,
                    session_id=task_id,
                )

            available = manager.list_plugin_skills(namespace)
            if available:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Skill '{bare}' not found in plugin '{namespace}'.",
                        "available_skills": [
                            f"{namespace}:{skill}" for skill in available
                        ],
                        "hint": (
                            f"The '{namespace}' plugin provides "
                            f"{len(available)} skill(s)."
                        ),
                    },
                    ensure_ascii=False,
                )
            if bare:
                local_category_name = f"{namespace}/{bare}"
                local_lookup_error = _skill_lookup_path_error(local_category_name)
                if local_lookup_error:
                    return json.dumps(
                        {
                            "success": False,
                            "error": local_lookup_error,
                            "hint": (
                                "Use a skill name or relative path within the "
                                "skills directory."
                            ),
                        },
                        ensure_ascii=False,
                    )
        active = _skills_dir()
        project_dirs = await _project_skills_dirs()
        roots = list(project_dirs)
        if await aiofiles.os.path.isdir(active):
            roots.append(active)
        roots.extend(await _external_skills_dirs())
        all_dirs = list(roots)
        if not roots:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Skills directory does not exist yet. It will be "
                        "created on first install."
                    ),
                },
                ensure_ascii=False,
            )
        candidates: list[tuple[Path | None, Path, Path]] = []
        seen: set[Path] = set()
        from agent.skill_utils import is_skill_support_path

        for root in roots:
            lookup_names = [name]
            if local_category_name:
                lookup_names.append(local_category_name)
            for lookup_name in lookup_names:
                direct = root / lookup_name
                direct_md = direct / "SKILL.md"
                if (
                    not await is_skill_support_path(direct, root=root)
                    and await aiofiles.os.path.isfile(direct_md)
                ):
                    resolved_direct = Path(await _realpath(direct_md))
                    if resolved_direct not in seen:
                        candidates.append((direct, direct_md, root))
                        seen.add(resolved_direct)
                legacy_md = root / f"{lookup_name}.md"
                if (
                    not await is_skill_support_path(legacy_md, root=root)
                    and await aiofiles.os.path.isfile(legacy_md)
                ):
                    resolved_legacy = Path(await _realpath(legacy_md))
                    if resolved_legacy not in seen:
                        candidates.append((None, legacy_md, root))
                        seen.add(resolved_legacy)
            async for candidate in _iter_skill_index_files(root, "SKILL.md"):
                resolved_candidate = Path(await _realpath(candidate))
                if resolved_candidate in seen:
                    continue
                if candidate.parent.name == name:
                    candidates.append((candidate.parent, candidate, root))
                    seen.add(resolved_candidate)
                    continue
                try:
                    frontmatter, _ = _parse_frontmatter(
                        await _read_skill_text(candidate, limit=4000)
                    )
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue
                if frontmatter.get("name") == name:
                    candidates.append((candidate.parent, candidate, root))
                    seen.add(resolved_candidate)
        if not candidates:
            return json.dumps({
                "success": False,
                "error": f"Skill '{name}' not found.",
                "available_skills": [
                    skill["name"] for skill in await _find_all_skills()
                ][:20],
                "hint": "Use skills_list to see all available skills",
            }, ensure_ascii=False)
        if len(candidates) > 1 and project_dirs:
            project_candidates = []
            for candidate in candidates:
                candidate_path = Path(await _realpath(candidate[1]))
                in_project = False
                for project_dir in project_dirs:
                    try:
                        candidate_path.relative_to(project_dir)
                    except ValueError:
                        continue
                    in_project = True
                    break
                if in_project:
                    project_candidates.append(candidate)
            if project_candidates:
                candidates = project_candidates
        if len(candidates) > 1:
            return json.dumps({
                "success": False,
                "error": f"Ambiguous skill name '{name}'. Load it by its categorized path.",
                "matches": [str(skill_md) for _, skill_md, _ in candidates],
                "hint": "Use the categorized skill path to choose one match.",
            }, ensure_ascii=False)

        skill_dir, skill_md, _skill_root = candidates[0]
        target = skill_md
        if file_path and skill_dir:
            if _path_security.has_traversal_component(file_path):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Path traversal ('..') is not allowed.",
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )
            target = skill_dir / file_path
            try:
                resolved_target = Path(await _realpath(target))
                resolved_target.relative_to(Path(await _realpath(skill_dir)))
            except (ValueError, OSError) as exc:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Path escapes allowed directory: {exc}",
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )
            if not await aiofiles.os.path.isfile(target):
                return json.dumps(
                    {
                        "success": False,
                        "error": f"File '{file_path}' not found in skill '{name}'.",
                        "available_files": await _available_skill_files(skill_dir),
                        "hint": "Use one of the available file paths listed above",
                    },
                    ensure_ascii=False,
                )

        try:
            content = await _read_skill_text(target)
        except UnicodeDecodeError:
            size = (await aiofiles.os.stat(target)).st_size
            return json.dumps({
                "success": True,
                "name": name,
                "file": file_path,
                "content": f"[Binary file: {target.name}, size: {size} bytes]",
                "is_binary": True,
            }, ensure_ascii=False)

        if file_path and skill_dir:
            try:
                from tools.skill_manager_tool import (
                    mark_background_review_skill_read,
                )

                await mark_background_review_skill_read(target)
            except Exception:
                logger.debug(
                    "Could not record background-review skill read for %s",
                    target,
                    exc_info=True,
                )
            return json.dumps({
                "success": True,
                "name": name,
                "file": file_path,
                "content": content,
                "file_type": target.suffix,
                "_source_path": str(target),
            }, ensure_ascii=False)

        resolved_skill = Path(await _realpath(skill_md))
        outside_skills_dir = True
        for trusted_root in all_dirs:
            try:
                resolved_skill.relative_to(Path(await _realpath(trusted_root)))
            except (OSError, ValueError):
                continue
            outside_skills_dir = False
            break
        injection_detected = any(
            pattern in content.lower() for pattern in _INJECTION_PATTERNS
        )
        if outside_skills_dir or injection_detected:
            warnings = []
            if outside_skills_dir:
                warnings.append(
                    "skill file is outside the trusted skills directory "
                    f"(~/.hermes/skills/): {skill_md}"
                )
            if injection_detected:
                warnings.append(
                    "skill content contains patterns that may indicate prompt injection"
                )
            logger.warning(
                "Skill security warning for '%s': %s",
                name,
                "; ".join(warnings),
            )

        frontmatter, _ = _parse_frontmatter(content)
        skill_name = str(
            frontmatter.get(
                "name",
                skill_md.stem if skill_dir is None else skill_dir.name,
            )
        )
        if not skill_matches_platform(frontmatter):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Skill '{name}' is not supported on this platform."
                    ),
                    "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
                },
                ensure_ascii=False,
            )
        if await _is_skill_disabled(skill_name):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Skill '{skill_name}' is disabled. Enable it with "
                        "`hermes skills` or inspect the files directly on disk."
                    ),
                },
                ensure_ascii=False,
            )
        linked_files = (
            await _linked_skill_files(skill_dir) if skill_dir is not None else {}
        )
        metadata = frontmatter.get("metadata")
        hermes_metadata = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
        if not isinstance(hermes_metadata, dict):
            hermes_metadata = {}
        required_environment_variables = _get_required_environment_variables(frontmatter)
        env_snapshot = await load_env()
        missing_environment_entries = [
            entry["name"]
            for entry in required_environment_variables
            if not entry.get("optional")
            and not _required_environment_variable_available(
                entry["name"], env_snapshot
            )
        ]
        missing_entry_names = set(missing_environment_entries)
        capture_result = await _capture_required_environment_variables(
            skill_name,
            [
                entry
                for entry in required_environment_variables
                if entry["name"] in missing_entry_names
            ],
        )
        if missing_environment_entries:
            env_snapshot = await load_env()
        callback_missing = set(capture_result["missing_names"])
        missing_environment_variables = [
            entry["name"]
            for entry in required_environment_variables
            if not entry.get("optional")
            and (
                entry["name"] in callback_missing
                or not _required_environment_variable_available(
                    entry["name"], env_snapshot
                )
            )
        ]
        missing_credential_files = await _missing_required_credential_files(frontmatter)
        available_environment_variables = [
            entry["name"]
            for entry in required_environment_variables
            if entry["name"] not in missing_environment_variables
        ]
        if available_environment_variables:
            from tools.env_passthrough import register_env_passthrough

            register_env_passthrough(available_environment_variables)
        readiness_status = (
            SkillReadinessStatus.SETUP_NEEDED
            if missing_environment_variables or missing_credential_files
            else SkillReadinessStatus.AVAILABLE
        )
        rendered_content = content
        if preprocess:
            try:
                rendered_content = await _skill_preprocessing.preprocess_skill_content(
                    content,
                    skill_dir,
                    session_id=task_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Could not preprocess skill content for %s",
                    skill_name,
                    exc_info=True,
                )

        try:
            from tools.skill_manager_tool import mark_background_review_skill_read

            await mark_background_review_skill_read(skill_md)
        except Exception:
            logger.debug(
                "Could not record background-review skill read for %s",
                skill_md,
                exc_info=True,
            )

        result = {
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": _parse_tags(
                hermes_metadata.get("tags") or frontmatter.get("tags")
            ),
            "related_skills": _parse_tags(
                hermes_metadata.get("related_skills")
                or frontmatter.get("related_skills")
            ),
            "content": rendered_content,
            "path": (
                str(skill_md.relative_to(active))
                if skill_md.is_relative_to(active)
                else str(skill_md.relative_to(skill_md.parent.parent))
            ),
            "skill_dir": str(skill_dir) if skill_dir else None,
            "org_provenance": None,
            "linked_files": linked_files or None,
            "usage_hint": (
                "To view linked files, call skill_view(name, file_path) where "
                "file_path is e.g. 'references/api.md' or 'assets/config.yaml'"
                if linked_files
                else None
            ),
            "required_environment_variables": required_environment_variables,
            "required_commands": [],
            "missing_required_environment_variables": missing_environment_variables,
            "missing_credential_files": missing_credential_files,
            "missing_required_commands": [],
            "setup_needed": readiness_status == SkillReadinessStatus.SETUP_NEEDED,
            "setup_skipped": capture_result["setup_skipped"],
            "readiness_status": readiness_status.value,
            "_source_path": str(skill_md),
        }
        setup_help = next(
            (
                entry["help"]
                for entry in required_environment_variables
                if entry.get("help")
            ),
            None,
        )
        if setup_help:
            result["setup_help"] = setup_help
        if capture_result["gateway_setup_hint"]:
            result["gateway_setup_hint"] = capture_result["gateway_setup_hint"]
        if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
            missing_items = [
                f"env ${name}" for name in missing_environment_variables
            ] + [f"file {path}" for path in missing_credential_files]
            setup_note = _build_setup_note(
                readiness_status,
                missing_items,
                setup_help,
            )
            if setup_note:
                result["setup_note"] = setup_note
        if frontmatter.get("compatibility"):
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata
        return json.dumps(result, ensure_ascii=False)
    except UnscopedSecretError:
        raise
    except Exception as exc:
        return tool_error(str(exc), success=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans').",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}


async def _handle_skills_list(args: dict, **kwargs) -> str:
    """Adapt the registry's JSON-object contract to ``skills_list``."""
    return await skills_list(category=args.get("category"), task_id=kwargs.get("task_id"))


_skill_profile_context: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("skill_view_profile_scope", default=None)
)
_skill_profile_aliases: dict[str, str] = {}
_skill_profile_lock = threading.RLock()
_skill_view_tracker_states: dict[
    str,
    dict[str, dict[tuple[str, str], tuple[str, int, int]]],
] = {}


def _lexical_skill_profile_identity() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_skill_profile_identity() -> str:
    lexical = _lexical_skill_profile_identity()
    active = _skill_profile_context.get()
    if active is not None and active[0] == lexical:
        return active[1]
    with _skill_profile_lock:
        return _skill_profile_aliases.get(lexical, lexical)


async def _activate_skill_profile_scope() -> str:
    lexical = _lexical_skill_profile_identity()
    active = _skill_profile_context.get()
    if active is not None and active[0] == lexical:
        return active[1]
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    expanded = str(await expanduser(lexical))
    is_absolute = (
        expanded.startswith(("/", "\\\\"))
        or (len(expanded) >= 3 and expanded[1] == ":" and expanded[2] in "/\\")
    )
    if not is_absolute:
        expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
    canonical = os.path.normcase(str(await _realpath(expanded)))
    with _skill_profile_lock:
        _skill_profile_aliases[lexical] = canonical
        active_state = _skill_view_tracker_states.setdefault(canonical, {})
        if lexical != canonical:
            staged = _skill_view_tracker_states.pop(lexical, None)
            if staged:
                for task_id, cache in staged.items():
                    active_state.setdefault(task_id, {}).update(cache)
    _skill_profile_context.set((lexical, canonical))
    return canonical


class _ScopedSkillViewTracker(MutableMapping):
    """Dict-compatible active-profile view retained for private test hooks."""

    def _active(self) -> dict:
        profile = _current_skill_profile_identity()
        return _skill_view_tracker_states.setdefault(profile, {})

    def __getitem__(self, key):
        with _skill_profile_lock:
            return self._active()[key]

    def __setitem__(self, key, value) -> None:
        with _skill_profile_lock:
            self._active()[key] = value

    def __delitem__(self, key) -> None:
        with _skill_profile_lock:
            del self._active()[key]

    def __iter__(self):
        with _skill_profile_lock:
            return iter(tuple(self._active()))

    def __len__(self) -> int:
        with _skill_profile_lock:
            return len(self._active())

    def clear(self) -> None:
        with _skill_profile_lock:
            self._active().clear()


_skill_view_tracker: MutableMapping[
    str,
    dict[tuple[str, str], tuple[str, int, int]],
] = _ScopedSkillViewTracker()
_SKILL_VIEW_DEDUP_CAP = 200
_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)"
)


async def _skill_view_fingerprint(payload: dict) -> tuple[str, int, int] | None:
    source = payload.get("_source_path")
    if not source:
        return None
    try:
        stat_result = await aiofiles.os.stat(source)
    except OSError:
        return None
    return str(source), stat_result.st_mtime_ns, stat_result.st_size


async def _check_skill_view_dedup(
    task_id: str | None,
    name: str,
    file_path: str | None,
) -> str | None:
    if not task_id:
        return None
    with _skill_profile_lock:
        cache = _skill_view_tracker.get(str(task_id))
        if not cache:
            return None
        entries = list(cache.items())
    requested_file = file_path or ""
    for key, fingerprint in entries:
        recorded_name, recorded_file = key
        if recorded_file != requested_file:
            continue
        if (
            recorded_name != str(name)
            and not str(name).endswith("/" + recorded_name)
            and not recorded_name.endswith("/" + str(name))
            and str(name).split(":")[-1] != recorded_name
        ):
            continue
        source, mtime_ns, size = fingerprint
        try:
            stat_result = await aiofiles.os.stat(source)
        except OSError:
            with _skill_profile_lock:
                cache.pop(key, None)
            return None
        if (stat_result.st_mtime_ns, stat_result.st_size) != (mtime_ns, size):
            with _skill_profile_lock:
                cache.pop(key, None)
            return None
        return json.dumps(
            {
                "success": True,
                "status": "unchanged",
                "name": recorded_name,
                "file": file_path or "SKILL.md",
                "dedup": True,
                "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE,
            },
            ensure_ascii=False,
        )
    return None


async def _record_skill_view(
    task_id: str | None,
    name: str,
    file_path: str | None,
    payload: dict,
) -> None:
    if not task_id or payload.get("setup_needed"):
        return
    fingerprint = await _skill_view_fingerprint(payload)
    if fingerprint is None:
        return
    key = (str(payload.get("name") or name), file_path or "")
    with _skill_profile_lock:
        cache = _skill_view_tracker.setdefault(str(task_id), {})
        cache[key] = fingerprint
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:
            cache.pop(next(iter(cache)))


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear repeat-view state after compression or session teardown."""
    with _skill_profile_lock:
        if task_id is None:
            _skill_view_tracker.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)


async def _handle_skill_view(args: dict, **kwargs) -> str:
    """Adapt the registry's JSON-object contract to ``skill_view``."""
    await _activate_skill_profile_scope()
    name = args.get("name", "")
    file_path = args.get("file_path")
    task_id = kwargs.get("task_id")
    cached = await _check_skill_view_dedup(task_id, name, file_path)
    if cached is not None:
        return cached
    result = await skill_view(name, file_path=file_path, task_id=task_id)
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return result
    if isinstance(payload, dict) and payload.get("success"):
        await _record_skill_view(task_id, name, file_path, payload)
        resolved_name = str(payload.get("name") or name)
        source_path = payload.get("_source_path")
        if source_path:
            from tools.skill_manager_tool import mark_background_review_skill_read

            await mark_background_review_skill_read(Path(source_path))
        if resolved_name:
            try:
                from tools.skill_usage import bump_use, bump_view

                await bump_view(resolved_name)
                await bump_use(resolved_name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "skill usage telemetry failed for %s",
                    resolved_name,
                    exc_info=True,
                )
        return json.dumps(payload, ensure_ascii=False)
    return result


registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=_handle_skills_list,
    check_fn=check_skills_requirements,
    emoji="📚",
)
registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=_handle_skill_view,
    check_fn=check_skills_requirements,
    emoji="📚",
)
