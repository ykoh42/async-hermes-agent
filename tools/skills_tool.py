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
import json
import logging
import time

from hermes_constants import get_hermes_home, display_hermes_home
import os
import re
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Any, List, Optional, Set, Tuple

import aiofiles
import aiofiles.os

from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS,
)

logger = logging.getLogger(__name__)

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
            iterator = await aiofiles.os.scandir(directory)
            try:
                for entry in iterator:
                    candidate = Path(entry.path)
                    try:
                        if not await aiofiles.os.path.isdir(candidate):
                            continue
                        child_stat = await aiofiles.os.stat(candidate)
                        newest = max(newest, child_stat.st_mtime)
                    except OSError:
                        continue
            finally:
                iterator.close()
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


async def _external_skills_dirs() -> List[Path]:
    """Resolve configured external skill roots without synchronous file I/O."""
    from hermes_cli.config import get_config_path

    config_path = get_config_path()
    if not await aiofiles.os.path.isfile(config_path):
        # Keep test/in-process callers that inject an external-root resolver
        # working without making the normal configured path synchronous.  A
        # patched resolver has no filesystem work of its own; production
        # callers use the async YAML path below.
        from agent import skill_utils as _skill_utils
        legacy_resolver = getattr(_skill_utils, "get_external_skills_dirs", None)
        if getattr(legacy_resolver, "__module__", "") == "unittest.mock":
            return [Path(value) for value in (legacy_resolver() or [])]
        return []
    try:
        async with aiofiles.open(config_path, encoding="utf-8") as handle:
            raw_config = await handle.read()
        import yaml

        config = yaml.safe_load(raw_config) or {}
    except (OSError, UnicodeDecodeError, ValueError):
        return []
    if not isinstance(config, dict):
        return []
    skills_config = config.get("skills")
    if not isinstance(skills_config, dict):
        return []
    raw_dirs = skills_config.get("external_dirs")
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if not isinstance(raw_dirs, list):
        return []

    local_skills = _skills_dir().resolve()
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw_dir in raw_dirs:
        value = str(raw_dir or "").strip()
        if not value:
            continue
        candidate = Path(os.path.expanduser(os.path.expandvars(value)))
        if not candidate.is_absolute():
            candidate = get_hermes_home() / candidate
        candidate = candidate.resolve()
        if candidate == local_skills or candidate in seen:
            continue
        try:
            if await aiofiles.os.path.isdir(candidate):
                seen.add(candidate)
                roots.append(candidate)
        except OSError:
            continue
    return roots


async def _active_org_id(skills_dir: Path) -> str | None:
    marker = skills_dir / "_org" / ".active_org"
    try:
        if not await aiofiles.os.path.isfile(marker):
            return None
        async with aiofiles.open(marker, encoding="utf-8") as handle:
            value = (await handle.read()).strip()
        return value or None
    except (OSError, UnicodeDecodeError):
        return None


async def _iter_skill_index_files(skills_dir: Path, filename: str):
    """Yield skill index files using native async directory operations."""
    from agent.skill_utils import SKILL_SUPPORT_DIRS

    active_org = await _active_org_id(skills_dir)
    org_root = skills_dir / "_org"

    async def walk(directory: Path):
        try:
            iterator = await aiofiles.os.scandir(directory)
        except OSError:
            return
        files: list[str] = []
        directories: list[Path] = []
        try:
            for entry in iterator:
                name = entry.name
                candidate = Path(entry.path)
                try:
                    if await aiofiles.os.path.isdir(candidate):
                        directories.append(candidate)
                    elif name == filename:
                        files.append(name)
                except OSError:
                    continue
        finally:
            iterator.close()

        if filename in files:
            yield directory / filename

        for child in sorted(directories, key=lambda path: path.name):
            relative_parts = child.relative_to(skills_dir).parts
            if any(part in _EXCLUDED_SKILL_DIRS for part in relative_parts):
                continue
            if relative_parts and relative_parts[0] == "_org":
                if active_org is None:
                    continue
                if len(relative_parts) == 2 and relative_parts[1] != active_org:
                    continue
                if len(relative_parts) > 2 and relative_parts[1] != active_org:
                    continue
            if filename in files and child.name in SKILL_SUPPORT_DIRS:
                continue
            async for result in walk(child):
                yield result

    async for result in walk(skills_dir):
        yield result


async def _iter_files(directory: Path):
    """Yield regular files below *directory* without blocking the loop."""
    try:
        iterator = await aiofiles.os.scandir(directory)
    except OSError:
        return
    directories: list[Path] = []
    try:
        for entry in iterator:
            candidate = Path(entry.path)
            try:
                if await aiofiles.os.path.isdir(candidate):
                    directories.append(candidate)
                elif await aiofiles.os.path.isfile(candidate):
                    yield candidate
            except OSError:
                continue
    finally:
        iterator.close()
    for child in sorted(directories, key=lambda path: path.name):
        async for result in _iter_files(child):
            yield result


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


def _skill_lookup_path_error(name: str) -> Optional[str]:
    """Return an error if a local skill lookup *name* can escape search roots.

    The skill ``name`` is joined onto each trusted search dir to build the
    on-disk lookup path, so it must stay relative and free of ``..`` segments —
    otherwise ``name="../outside"`` or an absolute path could select a skill
    (and read files) outside the skills directory. Mirrors the ``file_path``
    validation done later via ``tools.path_security``. We also reject Windows
    drive paths (e.g. ``C:\\skills``), whose ``:`` would otherwise be misread as
    a plugin namespace separator.
    """
    from tools.path_security import has_traversal_component

    if not isinstance(name, str):
        return "Skill name must be a string."
    candidate = name.strip()
    if (
        PurePosixPath(candidate).is_absolute()
        or PureWindowsPath(candidate).is_absolute()
        or PureWindowsPath(candidate).drive
    ):
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


async def _load_env() -> Dict[str, str]:
    """Read the profile environment file without blocking the event loop."""
    env_path = get_hermes_home() / ".env"
    if not await aiofiles.os.path.isfile(env_path):
        return {}

    async with aiofiles.open(env_path, encoding="utf-8") as handle:
        contents = await handle.read()

    env_vars: Dict[str, str] = {}
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new instructions:",
    "system prompt:",
    "<system>",
    "]]>",
)


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Check if a skill is compatible with the current OS platform.

    Delegates to ``agent.skill_utils.skill_matches_platform`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import skill_matches_platform as _impl
    return _impl(frontmatter)


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """Check if a skill is relevant to the current runtime environment.

    Delegates to ``agent.skill_utils.skill_matches_environment`` — kept here
    as a public re-export so existing callers don't need updating. This is an
    offer-time relevance gate (kanban/docker/s6), NOT a hard-compatibility gate;
    explicit skill loads bypass it.
    """
    from agent.skill_utils import skill_matches_environment as _impl
    return _impl(frontmatter)


def _normalize_prerequisite_values(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(
    frontmatter: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _normalize_setup_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
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

    collect_secrets: List[Dict[str, Any]] = []
    for item in collect_secrets_raw:
        if not isinstance(item, dict):
            continue

        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue

        prompt = str(item.get("prompt") or f"Enter value for {env_var}").strip()
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()

        entry: Dict[str, Any] = {
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
    frontmatter: Dict[str, Any],
    legacy_env_vars: List[str] | None = None,
) -> List[Dict[str, Any]]:
    setup = _normalize_setup_metadata(frontmatter)
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, dict):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    required: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: Dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen:
            return
        if not _ENV_VAR_NAME_RE.match(env_name):
            return

        normalized: Dict[str, Any] = {
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
    missing: List[str],
    setup_help: str | None = None,
) -> str | None:
    if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
        missing_str = ", ".join(missing) if missing else "required prerequisites"
        note = f"Setup needed before using this skill: missing {missing_str}."
        if setup_help:
            return f"{note} {setup_help}"
        return note
    return None


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Delegates to ``agent.skill_utils.parse_frontmatter`` — kept here
    as a public re-export so existing callers don't need updating.
    """
    from agent.skill_utils import parse_frontmatter
    return parse_frontmatter(content)


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """
    Extract category from skill path based on directory structure.

    For paths like: ~/.hermes/skills/mlops/axolotl/SKILL.md -> "mlops"
    Also works for external skill dirs configured via skills.external_dirs.
    """
    # Try the active profile skills dir first (respects monkeypatching in tests),
    # then fall back to external dirs from config.
    dirs_to_check = [_skills_dir()]
    try:
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    except Exception:
        pass
    for skills_dir in dirs_to_check:
        try:
            rel_path = skill_path.relative_to(skills_dir)
            parts = rel_path.parts
            if len(parts) >= 3:
                return parts[0]
        except ValueError:
            continue
    return None


def _parse_tags(tags_value) -> List[str]:
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



async def _get_disabled_skill_names(platform: str | None = None) -> Set[str]:
    """Load disabled skill names through the native async config boundary."""
    from hermes_cli.config import get_config_path

    config_path = get_config_path()
    try:
        if not await aiofiles.os.path.isfile(config_path):
            return set()
        async with aiofiles.open(config_path, encoding="utf-8") as handle:
            import yaml

            config = yaml.safe_load(await handle.read()) or {}
    except asyncio.CancelledError:
        raise
    except (OSError, UnicodeDecodeError, ValueError):
        return set()
    if not isinstance(config, dict):
        return set()
    skills_cfg = config.get("skills")
    if not isinstance(skills_cfg, dict):
        return set()

    resolved_platform = platform or os.getenv("HERMES_PLATFORM") or _get_session_platform()

    def _normalize(values: object) -> set[str]:
        if values is None:
            return set()
        if isinstance(values, str):
            values = [values]
        return {str(value).strip() for value in values if str(value).strip()}

    disabled = _normalize(skills_cfg.get("disabled"))
    if resolved_platform:
        platform_disabled = (skills_cfg.get("platform_disabled") or {}).get(
            resolved_platform
        )
        disabled |= _normalize(platform_disabled)
    return disabled


def _get_session_platform() -> str:
    """Resolve the current platform from gateway session context.

    Mirrors the platform-resolution logic in
    ``agent.skill_utils.get_disabled_skill_names`` so that
    ``_is_skill_disabled`` respects ``HERMES_SESSION_PLATFORM``.
    """
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_PLATFORM") or ""
    except Exception:
        return ""


async def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """Check if a skill is disabled in config.

    Resolves the active platform from (in order of precedence):
    1. Explicit ``platform`` argument
    2. ``HERMES_PLATFORM`` environment variable
    3. ``HERMES_SESSION_PLATFORM`` from gateway session context
    """
    try:
        config = {}
        from hermes_cli.config import get_config_path
        config_path = get_config_path()
        if await aiofiles.os.path.isfile(config_path):
            async with aiofiles.open(config_path, encoding="utf-8") as handle:
                import yaml

                config = yaml.safe_load(await handle.read()) or {}
        skills_cfg = config.get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM") or _get_session_platform()
        global_disabled = skills_cfg.get("disabled", [])
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
            if platform_disabled is not None:
                # A globally-disabled skill stays disabled on every platform;
                # the platform list adds to it rather than replacing it. Keep
                # in sync with agent.skill_utils.get_disabled_skill_names.
                return name in platform_disabled or name in global_disabled
        return name in global_disabled
    except asyncio.CancelledError:
        raise
    except Exception:
        return False


def _sort_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))




async def _read_skill_text(path: Path, *, limit: int | None = None) -> str:
    """Read one skill file through the async file boundary."""
    async with aiofiles.open(path, encoding="utf-8") as handle:
        return await (handle.read(limit) if limit is not None else handle.read())


async def _find_all_skills(*, skip_disabled: bool = False) -> List[Dict[str, Any]]:
    """List local and external skills through the async file boundary.

    Results are cached per discovery signature and returned as copies because
    callers may annotate individual skill dictionaries.
    """
    cache_key = _SKILLS_CACHE_KEY_DISABLED if skip_disabled else _SKILLS_CACHE_KEY_FILTERED
    disabled = set() if skip_disabled else await _get_disabled_skill_names()
    roots: list[Path] = []
    active = _skills_dir()
    if await aiofiles.os.path.isdir(active):
        roots.append(active)
    roots.extend(await _external_skills_dirs())
    signature = await _skills_scan_signature(roots, disabled)
    now = time.monotonic()
    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature and now - cached[1] < _SKILLS_CACHE_TTL_SECONDS:
        return [dict(skill) for skill in cached[2]]

    skills: List[Dict[str, Any]] = []
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
                    or not skill_matches_environment(frontmatter)
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
                    "category": _get_category_from_path(skill_md),
                })
            except (OSError, UnicodeDecodeError, PermissionError) as exc:
                logger.debug("Failed to read skill file %s: %s", skill_md, exc)
            except Exception as exc:
                logger.debug(
                    "Skipping skill at %s: failed to parse: %s", skill_md, exc, exc_info=True
                )

    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(skill) for skill in skills]


async def skills_list(category: str = None, task_id: str = None) -> str:
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


async def _serve_plugin_skill(skill_md: Path, namespace: str, bare: str) -> str:
    """Read a registered plugin skill through the native async file boundary."""
    from hermes_cli.config import load_config_readonly_async
    from hermes_cli.plugins import get_plugin_manager

    config = await load_config_readonly_async()
    if namespace in set(cfg_get(config, "plugins", "disabled", default=[]) or []):
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
            f"[Bundle context: This skill is part of the '{namespace}' plugin.]\n\n"
        )

    return json.dumps(
        {
            "success": True,
            "name": f"{namespace}:{bare}",
            "content": f"{banner}{content}",
            "description": description,
            "linked_files": None,
            "readiness_status": SkillReadinessStatus.AVAILABLE.value,
        },
        ensure_ascii=False,
    )


async def skill_view(
    name: str,
    file_path: str = None,
    task_id: str = None,
    preprocess: bool = True,
) -> str:
    """View a local, externally configured, or plugin-provided skill."""
    try:
        lookup_error = _skill_lookup_path_error(name)
        if lookup_error:
            return tool_error(lookup_error, success=False)
        if ":" in name:
            from agent.skill_utils import is_valid_namespace, parse_qualified_name
            from hermes_cli.plugins import get_plugin_manager

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
                return await _serve_plugin_skill(plugin_skill_md, namespace, bare)

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
        active = _skills_dir()
        roots = [active] if await aiofiles.os.path.isdir(active) else []
        roots.extend(await _external_skills_dirs())
        candidates: list[tuple[Path | None, Path, Path]] = []
        seen: set[Path] = set()
        for root in roots:
            direct = root / name
            direct_md = direct / "SKILL.md"
            if await aiofiles.os.path.isfile(direct_md):
                candidates.append((direct, direct_md, root))
                seen.add(direct_md.resolve())
            legacy_md = root / f"{name}.md"
            if await aiofiles.os.path.isfile(legacy_md):
                candidates.append((None, legacy_md, root))
                seen.add(legacy_md.resolve())
            async for candidate in _iter_skill_index_files(root, "SKILL.md"):
                if candidate.resolve() in seen:
                    continue
                if candidate.parent.name == name:
                    candidates.append((candidate.parent, candidate, root))
                    seen.add(candidate.resolve())
                    continue
                try:
                    frontmatter, _ = _parse_frontmatter(
                        await _read_skill_text(candidate, limit=4000)
                    )
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue
                if frontmatter.get("name") == name:
                    candidates.append((candidate.parent, candidate, root))
                    seen.add(candidate.resolve())
        if not candidates:
            return json.dumps({
                "success": False,
                "error": f"Skill '{name}' not found.",
                "available_skills": [
                    skill["name"] for skill in await _find_all_skills()
                ][:20],
                "hint": "Use skills_list to see all available skills",
            }, ensure_ascii=False)
        if len(candidates) > 1:
            return json.dumps({
                "success": False,
                "error": f"Ambiguous skill name '{name}'. Load it by its categorized path.",
                "matches": [str(skill_md) for _, skill_md, _ in candidates],
                "hint": "Use the categorized skill path to choose one match.",
            }, ensure_ascii=False)

        skill_dir, skill_md, skill_root = candidates[0]
        target = skill_md
        if file_path:
            if skill_dir is None:
                return tool_error("This legacy flat skill has no linked files.", success=False)
            from tools.path_security import has_traversal_component, validate_within_dir

            if has_traversal_component(file_path):
                return tool_error("Path traversal ('..') is not allowed.", success=False)
            target = skill_dir / file_path
            traversal_error = validate_within_dir(target, skill_dir)
            if traversal_error:
                return tool_error(traversal_error, success=False)
            if not await aiofiles.os.path.isfile(target):
                return tool_error(
                    f"File '{file_path}' not found in skill '{name}'.", success=False
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

        try:
            from tools.skill_manager_tool import mark_background_review_skill_read

            mark_background_review_skill_read(target)
        except Exception:
            logger.debug(
                "Could not record background-review skill read for %s",
                target,
                exc_info=True,
            )
        if file_path:
            return json.dumps({
                "success": True,
                "name": name,
                "file": file_path,
                "content": content,
                "file_type": target.suffix,
            }, ensure_ascii=False)

        frontmatter, _ = _parse_frontmatter(content)
        skill_name = str(frontmatter.get("name", skill_md.parent.name))
        if await _is_skill_disabled(skill_name):
            return tool_error(f"Skill '{skill_name}' is disabled.", success=False)
        if not skill_matches_platform(frontmatter):
            return tool_error(
                f"Skill '{skill_name}' is not supported on this platform.", success=False
            )
        linked_files = {}
        if skill_dir is not None:
            for folder_name in ("references", "templates", "assets", "scripts"):
                folder = skill_dir / folder_name
                if await aiofiles.os.path.isdir(folder):
                    files = [
                        str(path.relative_to(skill_dir))
                        async for path in _iter_files(folder)
                    ]
                    if files:
                        linked_files[folder_name] = files
        metadata = frontmatter.get("metadata")
        hermes_metadata = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
        if not isinstance(hermes_metadata, dict):
            hermes_metadata = {}
        required_environment_variables = _get_required_environment_variables(frontmatter)
        env_snapshot = await _load_env()
        missing_environment_variables = [
            entry["name"]
            for entry in required_environment_variables
            if not entry.get("optional")
            and not bool(env_snapshot.get(entry["name"]) or os.getenv(entry["name"]))
        ]
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
            if missing_environment_variables
            else SkillReadinessStatus.AVAILABLE
        )
        return json.dumps({
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": _parse_tags(frontmatter.get("tags") or hermes_metadata.get("tags")),
            "related_skills": _parse_tags(
                frontmatter.get("related_skills") or hermes_metadata.get("related_skills")
            ),
            "content": content,
            "path": str(skill_md.relative_to(skill_root)),
            "skill_dir": str(skill_dir) if skill_dir else None,
            "linked_files": linked_files or None,
            "usage_hint": "Use skill_view(name, file_path) to load linked files." if linked_files else None,
            "required_environment_variables": required_environment_variables,
            "missing_required_environment_variables": missing_environment_variables,
            "setup_needed": readiness_status == SkillReadinessStatus.SETUP_NEEDED,
            "setup_note": _build_setup_note(
                readiness_status,
                missing_environment_variables,
            ),
            "readiness_status": readiness_status.value,
        }, ensure_ascii=False)
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


async def _handle_skill_view(args: dict, **kwargs) -> str:
    """Adapt the registry's JSON-object contract to ``skill_view``."""
    return await skill_view(
        args.get("name", ""),
        file_path=args.get("file_path"),
        task_id=kwargs.get("task_id"),
    )


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
