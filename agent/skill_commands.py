"""Native-async skill discovery and invocation-message helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import aiofiles.os

from agent.prompt_cache_boundary import register_stable_prefix
from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)

_skill_commands: dict[str, dict[str, Any]] = {}
_skill_commands_platform: str | None = None
_skill_commands_by_scope: dict[
    tuple[str, str | None],
    dict[str, dict[str, Any]],
] = {}
_skill_commands_projection_scope: tuple[str, str | None] | None = None
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")


def _skill_command_scope_key(platform: str | None) -> tuple[str, str | None]:
    """Return the active skill-root/platform cache key without filesystem I/O."""
    module = sys.modules.get("tools.skills_tool")
    skills_dir = getattr(module, "_skills_dir", None)
    if callable(skills_dir):
        root = skills_dir()
    else:
        from hermes_constants import get_hermes_home

        root = get_hermes_home() / "skills"
    return os.path.normcase(os.path.normpath(os.fspath(root))), platform


def _active_skill_commands(
    platform: str | None,
) -> dict[str, dict[str, Any]] | None:
    """Project the current profile cache onto the upstream private globals."""
    global _skill_commands, _skill_commands_platform, _skill_commands_projection_scope
    key = _skill_command_scope_key(platform)
    if (
        _skill_commands_projection_scope == key
        and not _skill_commands
        and _skill_commands_platform is None
    ):
        _skill_commands_by_scope.pop(key, None)
    cached = _skill_commands_by_scope.get(key)
    if cached is None:
        return None
    _skill_commands = cached
    _skill_commands_platform = platform
    _skill_commands_projection_scope = key
    return cached

# These literals are part of the persisted format emitted by the classic
# runtime.  Keep them byte-stable so existing session previews and memory rows
# remain readable after the async-only migration.
_SKILL_INVOCATION_PREFIX = "[IMPORTANT: The user has invoked the "
_SINGLE_SKILL_MARKER = "The full skill content is loaded below.]"
_SINGLE_SKILL_INSTRUCTION = (
    "The user has provided the following instruction alongside the skill invocation: "
)
_RUNTIME_NOTE = "\n\n[Runtime note:"
_BUNDLE_MARKER = " skill bundle,"
_BUNDLE_USER_INSTRUCTION = "\nUser instruction: "
_BUNDLE_FIRST_SKILL_BLOCK = "\n\n[Loaded as part of the "

_SKILL_NAME_RE = re.compile(re.escape(_SKILL_INVOCATION_PREFIX) + r'"([^"]*)"')

# SQL LIKE pattern matching a historical skill-expanded turn.  The literal
# prefix intentionally contains no LIKE wildcards.
SKILL_SCAFFOLD_SQL_LIKE = _SKILL_INVOCATION_PREFIX + "%"

# Joins head and tail excerpts before ``describe_skill_invocation`` parses
# them.  Never include text from the far side of this marker in a preview.
SKILL_EXCERPT_JOINT = "\x1e"


def append_user_instruction(parts: list, instruction: str) -> str:
    """Append the volatile instruction and return the stable scaffold prefix."""
    stable_prefix = "\n".join(parts) + "\n" + _SINGLE_SKILL_INSTRUCTION
    parts.append(f"{_SINGLE_SKILL_INSTRUCTION}{instruction}")
    return stable_prefix


def extract_user_instruction_from_skill_message(content: Any) -> str | None:
    """Return the actual user instruction from a historical skill turn.

    A non-skill string is returned unchanged.  A bare skill invocation and
    non-string content return ``None`` so memory providers do not persist the
    injected skill body as if it were user-authored text.
    """
    if not isinstance(content, str):
        return None
    if not content.startswith(_SKILL_INVOCATION_PREFIX):
        return content
    if _BUNDLE_MARKER in content:
        return _extract_bundle_user_instruction(content)
    if _SINGLE_SKILL_MARKER in content:
        return _extract_single_skill_user_instruction(content)
    return None


def describe_skill_invocation(content: Any, separator: str = " — ") -> str | None:
    """Render a historical expanded skill turn as its original slash command."""
    if not isinstance(content, str) or not content.startswith(_SKILL_INVOCATION_PREFIX):
        return None

    match = _SKILL_NAME_RE.match(content)
    name = (match.group(1) if match else "").strip()
    label = name if name.startswith("/") else f"/{name}"

    instruction = extract_user_instruction_from_skill_message(content)
    if instruction and instruction is not content:
        instruction = " ".join(instruction.split(SKILL_EXCERPT_JOINT)[0].split())
        if instruction:
            return f"{label}{separator}{instruction}" if name else instruction
    return label if name else None


def _extract_single_skill_user_instruction(message: str) -> str | None:
    marker_index = message.rfind(_SINGLE_SKILL_INSTRUCTION)
    if marker_index < 0:
        return None
    instruction = message[marker_index + len(_SINGLE_SKILL_INSTRUCTION):]
    runtime_index = instruction.find(_RUNTIME_NOTE)
    if runtime_index >= 0:
        instruction = instruction[:runtime_index]
    return instruction.strip() or None


def _extract_bundle_user_instruction(message: str) -> str | None:
    marker_index = message.find(_BUNDLE_USER_INSTRUCTION)
    if marker_index < 0:
        return None
    instruction = message[marker_index + len(_BUNDLE_USER_INSTRUCTION):]
    first_skill_index = instruction.find(_BUNDLE_FIRST_SKILL_BLOCK)
    if first_skill_index >= 0:
        instruction = instruction[:first_skill_index]
    return instruction.strip() or None


def _resolve_skill_commands_platform() -> str | None:
    """Return the platform scope used for disabled-skill filtering."""
    try:
        from gateway.session_context import get_session_env

        platform = os.getenv("HERMES_PLATFORM") or get_session_env(
            "HERMES_SESSION_PLATFORM"
        )
    except Exception:
        platform = os.getenv("HERMES_PLATFORM")
    return platform or None


async def _load_skill_payload(
    skill_identifier: str,
    task_id: str | None = None,
) -> tuple[dict[str, Any], Path | None, str] | None:
    """Load one skill and return its payload, directory, and display name."""
    raw_identifier = (skill_identifier or "").strip()
    if not raw_identifier:
        return None
    try:
        from agent.skill_utils import normalize_skill_lookup_name
        from tools.skills_tool import SKILLS_DIR, skill_view

        normalized = await normalize_skill_lookup_name(raw_identifier)
        loaded_skill = json.loads(
            await skill_view(normalized, task_id=task_id, preprocess=False)
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return None
    if not loaded_skill.get("success"):
        return None

    skill_name = str(loaded_skill.get("name") or normalized)
    skill_path = str(loaded_skill.get("path") or "")
    skill_dir = None
    absolute_skill_dir = loaded_skill.get("skill_dir")
    if absolute_skill_dir:
        skill_dir = Path(absolute_skill_dir)
    elif skill_path:
        try:
            skill_dir = SKILLS_DIR / Path(skill_path).parent
        except Exception:
            skill_dir = None
    return loaded_skill, skill_dir, skill_name


async def _inject_skill_config(
    loaded_skill: dict[str, Any], parts: list[str]
) -> None:
    """Resolve skill-declared config without synchronous config reads."""
    try:
        from agent.skill_utils import (
            SKILL_CONFIG_PREFIX,
            extract_skill_config_vars,
            parse_frontmatter,
        )
        from hermes_cli.config import load_config_readonly

        raw_content = str(
            loaded_skill.get("raw_content")
            or loaded_skill.get("content")
            or ""
        )
        if not raw_content:
            return
        frontmatter, _ = parse_frontmatter(raw_content)
        config_vars = extract_skill_config_vars(frontmatter)
        if not config_vars:
            return
        config = await load_config_readonly()

        def _resolve_dotpath(dotted_key: str):
            current: Any = config
            for key in dotted_key.split("."):
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current

        resolved: dict[str, Any] = {}
        for variable in config_vars:
            logical_key = variable["key"]
            value = _resolve_dotpath(f"{SKILL_CONFIG_PREFIX}.{logical_key}")
            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                value = variable.get("default", "")
            if isinstance(value, str) and ("~" in value or "${" in value):
                value = os.path.expanduser(os.path.expandvars(value))
            resolved[logical_key] = value
        if not resolved:
            return
        parts.extend(
            [
                "",
                f"[Skill config (from {display_hermes_home()}/config.yaml):",
                *(
                    f"  {key} = {value if value else '(not set)'}"
                    for key, value in resolved.items()
                ),
                "]",
            ]
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Could not inject skill config", exc_info=True)


async def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path | None,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
    session_id: str | None = None,
) -> str:
    """Format one loaded skill into the upstream model-facing message."""
    from agent.skill_preprocessing import preprocess_skill_content
    from tools.skills_tool import SKILLS_DIR, _iter_files

    content = await preprocess_skill_content(
        str(loaded_skill.get("content") or ""),
        skill_dir,
        session_id=session_id,
    )
    parts = [activation_note, "", content.strip()]

    if skill_dir:
        parts.extend(
            [
                "",
                f"[Skill directory: {skill_dir}]",
                "Resolve any relative paths in this skill (e.g. `scripts/foo.js`, "
                "`templates/config.yaml`) against that directory, then run them "
                "with the terminal tool using the absolute path.",
            ]
        )

    await _inject_skill_config(loaded_skill, parts)

    if loaded_skill.get("setup_skipped"):
        parts.extend(
            [
                "",
                "[Skill setup note: Required environment setup was skipped. "
                "Continue loading the skill and explain any reduced "
                "functionality if it matters.]",
            ]
        )
    elif loaded_skill.get("gateway_setup_hint"):
        parts.extend(
            ["", f"[Skill setup note: {loaded_skill['gateway_setup_hint']}]"]
        )
    elif loaded_skill.get("setup_needed") and loaded_skill.get("setup_note"):
        parts.extend(["", f"[Skill setup note: {loaded_skill['setup_note']}]"])

    supporting: list[str] = []
    linked_files = loaded_skill.get("linked_files") or {}
    for entries in linked_files.values():
        if isinstance(entries, list):
            supporting.extend(str(entry) for entry in entries)
    if not supporting and skill_dir:
        for subdir in ("references", "templates", "scripts", "assets"):
            directory = skill_dir / subdir
            async for file_path in _iter_files(directory):
                if not await aiofiles.os.path.islink(file_path):
                    supporting.append(str(file_path.relative_to(skill_dir)))

    if supporting and skill_dir:
        try:
            skill_view_target = str(skill_dir.relative_to(SKILLS_DIR))
        except ValueError:
            skill_view_target = skill_dir.name
        parts.extend(["", "[This skill has supporting files:]"])
        parts.extend(f"- {path}  ->  {skill_dir / path}" for path in supporting)
        parts.append(
            f'\nLoad any of these with skill_view(name="{skill_view_target}", '
            'file_path="<path>"), or run scripts directly by absolute path '
            f"(e.g. `node {skill_dir}/scripts/foo.js`)."
        )
    stable_prefix = None
    if user_instruction:
        parts.append("")
        stable_prefix = append_user_instruction(parts, user_instruction)
    if runtime_note:
        parts.extend(["", f"[Runtime note: {runtime_note}]"])
    message = "\n".join(parts)
    if (
        stable_prefix is not None
        and message.startswith(stable_prefix)
        and len(message) > len(stable_prefix)
    ):
        register_stable_prefix(stable_prefix)
    return message


async def scan_skill_commands() -> dict[str, dict[str, Any]]:
    """Scan local and external skill roots into slash-command metadata."""
    global _skill_commands, _skill_commands_platform, _skill_commands_projection_scope

    from tools.skills_tool import (
        _external_skills_dirs,
        _get_disabled_skill_names,
        _iter_skill_index_files,
        _parse_frontmatter,
        _read_skill_text,
        _skills_dir,
        skill_matches_environment,
        skill_matches_platform,
    )
    from agent.skill_utils import get_project_skills_dirs

    scan_platform = _resolve_skill_commands_platform()
    commands: dict[str, dict[str, Any]] = {}
    disabled = await _get_disabled_skill_names(scan_platform)
    roots: list[Path] = await get_project_skills_dirs()
    local_root = _skills_dir()
    if await aiofiles.os.path.isdir(local_root):
        roots.append(local_root)
    roots.extend(await _external_skills_dirs())
    seen_names: set[str] = set()
    for root in roots:
        async for skill_md in _iter_skill_index_files(root, "SKILL.md"):
            try:
                content = await _read_skill_text(skill_md)
                frontmatter, body = _parse_frontmatter(content)
                if not skill_matches_platform(frontmatter):
                    continue
                if not await skill_matches_environment(frontmatter):
                    continue
                name = str(frontmatter.get("name") or skill_md.parent.name)
                if name in seen_names or name in disabled:
                    continue
                description = str(frontmatter.get("description") or "")
                if not description:
                    description = next(
                        (
                            line[:80]
                            for line in (
                                candidate.strip()
                                for candidate in body.strip().split("\n")
                            )
                            if line and not line.startswith("#")
                        ),
                        "",
                    )
                command_name = name.lower().replace(" ", "-").replace("_", "-")
                command_name = _SKILL_INVALID_CHARS.sub("", command_name)
                command_name = _SKILL_MULTI_HYPHEN.sub("-", command_name).strip("-")
                if not command_name:
                    continue
                command_key = f"/{command_name}"
                if command_key in commands:
                    logger.warning(
                        "Skill %r maps to slash command %s already claimed by %r; "
                        "keeping the first and skipping this one.",
                        name,
                        command_key,
                        commands[command_key]["name"],
                    )
                    continue
                seen_names.add(name)
                commands[command_key] = {
                    "name": name,
                    "description": description or f"Invoke the {name} skill",
                    "skill_md_path": str(skill_md),
                    "skill_dir": str(skill_md.parent),
                }
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
    _skill_commands = commands
    _skill_commands_platform = scan_platform
    scope = _skill_command_scope_key(scan_platform)
    _skill_commands_by_scope[scope] = commands
    _skill_commands_projection_scope = scope
    return _skill_commands


async def get_skill_commands() -> dict[str, dict[str, Any]]:
    """Return cached skill commands, refreshing for platform-scope changes."""
    platform = _resolve_skill_commands_platform()
    cached = _active_skill_commands(platform)
    if not cached:
        return await scan_skill_commands()
    return cached


async def reload_skills() -> dict[str, Any]:
    """Rescan skill roots and return the upstream added/removed diff."""
    def _snapshot(commands: dict[str, dict[str, Any]]) -> dict[str, str]:
        return {
            key.lstrip("/"): (info or {}).get("description") or ""
            for key, info in commands.items()
        }

    before = _snapshot(
        _active_skill_commands(_resolve_skill_commands_platform()) or {}
    )
    new_commands = await scan_skill_commands()
    after = _snapshot(new_commands)
    added_names = sorted(set(after) - set(before))
    removed_names = sorted(set(before) - set(after))
    return {
        "added": [
            {"name": name, "description": after[name]} for name in added_names
        ],
        "removed": [
            {"name": name, "description": before[name]}
            for name in removed_names
        ],
        "unchanged": sorted(set(after) & set(before)),
        "total": len(after),
        "commands": len(new_commands),
    }


async def resolve_skill_command_key(command: str) -> str | None:
    """Resolve underscore/hyphen variants to a canonical command key."""
    if not command:
        return None
    command_key = f"/{command.replace('_', '-')}"
    return command_key if command_key in await get_skill_commands() else None


async def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    runtime_note: str = "",
) -> str | None:
    """Build the model-facing message for one slash-skill invocation."""
    skill_info = (await get_skill_commands()).get(cmd_key)
    if not skill_info:
        return None
    loaded = await _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
    if not loaded:
        return None
    loaded_skill, skill_dir, skill_name = loaded
    activation_note = (
        f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating '
        "they want you to follow its instructions. The full skill content is "
        "loaded below.]"
    )
    return await _build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction=user_instruction,
        runtime_note=runtime_note,
        session_id=task_id,
    )


_MAX_STACKED_SKILLS = 5


async def split_stacked_skill_commands(rest: str) -> tuple[list[str], str]:
    """Consume additional leading installed skill-command tokens."""
    keys: list[str] = []
    remaining = rest or ""
    while len(keys) < _MAX_STACKED_SKILLS - 1:
        stripped = remaining.lstrip()
        if not stripped.startswith("/"):
            break
        parts = stripped.split(None, 1)
        token = parts[0]
        tail = parts[1] if len(parts) > 1 else ""
        command_key = await resolve_skill_command_key(token.lstrip("/"))
        if command_key is None or command_key in keys:
            break
        keys.append(command_key)
        remaining = tail
    return keys, remaining.strip()


async def build_stacked_skill_invocation_message(
    cmd_keys: list[str],
    user_instruction: str = "",
    task_id: str | None = None,
) -> tuple[str, list[str], list[str]] | None:
    """Build one message that activates several leading skill commands."""
    commands = await get_skill_commands()
    loaded_names: list[str] = []
    missing: list[str] = []
    skill_blocks: list[str] = []
    seen: set[str] = set()
    for command_key in cmd_keys:
        if not command_key or command_key in seen:
            continue
        seen.add(command_key)
        skill_info = commands.get(command_key)
        if not skill_info:
            missing.append(command_key.lstrip("/"))
            continue
        loaded = await _load_skill_payload(
            skill_info["skill_dir"], task_id=task_id
        )
        if not loaded:
            missing.append(command_key.lstrip("/"))
            continue
        loaded_skill, skill_dir, skill_name = loaded
        skill_blocks.append(
            await _build_skill_message(
                loaded_skill,
                skill_dir,
                f'[Loaded as part of the stacked skill invocation "{skill_name}".]',
                session_id=task_id,
            )
        )
        loaded_names.append(skill_name)
    if not skill_blocks:
        return None
    typed = " ".join(key for key in cmd_keys if key)
    header_lines = [
        f'[IMPORTANT: The user has invoked the "{typed}" stacked skill bundle, '
        f"loading {len(loaded_names)} skills together. Treat every skill below "
        "as active guidance for this turn.]",
        "",
        f"Skills loaded: {', '.join(loaded_names)}",
    ]
    if missing:
        header_lines.append(f"Skills missing (skipped): {', '.join(missing)}")
    if user_instruction:
        header_lines.extend(["", f"User instruction: {user_instruction}"])
    return (
        "\n\n".join(["\n".join(header_lines), *skill_blocks]),
        loaded_names,
        missing,
    )


async def build_preloaded_skills_prompt(
    skill_identifiers: list[str],
    task_id: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Load session-preloaded skills using the native async skill path."""
    from tools.skills_tool import _get_disabled_skill_names

    disabled_names = await _get_disabled_skill_names(
        _resolve_skill_commands_platform()
    )
    prompt_parts: list[str] = []
    loaded_names: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw_identifier in skill_identifiers:
        identifier = (raw_identifier or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        loaded = await _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue
        loaded_skill, skill_dir, skill_name = loaded
        if skill_name in disabled_names or identifier in disabled_names:
            missing.append(identifier)
            continue
        prompt_parts.append(
            await _build_skill_message(
                loaded_skill,
                skill_dir,
                f'[IMPORTANT: The user launched this CLI session with the "{skill_name}" '
                "skill preloaded. Treat its instructions as active guidance for the "
                "duration of this session unless the user overrides them.]",
                session_id=task_id,
            )
        )
        loaded_names.append(skill_name)
    return "\n\n".join(prompt_parts), loaded_names, missing
