#!/usr/bin/env python3
"""Native-async skill creation and editing.

This is the retained library form of Hermes' ``skill_manage`` tool.  It keeps
the upstream public name, arguments, action names, result shape, and file
location while performing every filesystem operation through an awaited I/O
boundary.  Product-specific curator, CLI approval, and organization-sync
workflows are intentionally outside this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional, Tuple

import aiofiles
import aiofiles.os
import yaml

from agent.skill_utils import (
    SKILL_PROMPT_DESC_LIMIT,
    extract_skill_description,
    get_all_skills_dirs,
    is_excluded_skill_path,
    is_skill_description_truncated_for_prompt,
    iter_skill_index_files,
    parse_frontmatter as _parse_frontmatter,
)
from hermes_constants import display_hermes_home, get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000
MAX_SKILL_FILE_BYTES = 1_048_576

VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = frozenset({"references", "templates", "scripts", "assets"})

_realpath = aiofiles.os.wrap(os.path.realpath)
_is_junction = aiofiles.os.wrap(
    lambda path: bool(getattr(path, "is_junction", lambda: False)())
)
_skill_write_lock = asyncio.Lock()


def _skills_dir() -> Path:
    """Return the active profile's local skills directory at call time."""
    configured = Path(SKILLS_DIR)
    if configured != _SKILLS_DIR_AT_IMPORT:
        return configured
    return get_hermes_home() / "skills"


def _validate_name(name: str) -> Optional[str]:
    """Validate a skill name. Return an error message or ``None``."""
    if not name:
        return "Skill name is required."
    if len(name) > MAX_NAME_LENGTH:
        return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.fullmatch(name):
        return (
            f"Invalid skill name '{name}'. Use lowercase letters, numbers, "
            "hyphens, dots, and underscores. Must start with a letter or digit."
        )
    return None


def _validate_category(category: Optional[str]) -> Optional[str]:
    """Validate an optional category as one safe directory component."""
    if category is None:
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    value = category.strip()
    if not value:
        return None
    if len(value) > MAX_NAME_LENGTH:
        return f"Category exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.fullmatch(value):
        return (
            f"Invalid category '{category}'. Use lowercase letters, numbers, "
            "hyphens, dots, and underscores. Categories must be a single "
            "directory name."
        )
    return None


def _validate_frontmatter(content: str, *, new_skill: bool = False) -> Optional[str]:
    """Validate the upstream ``SKILL.md`` frontmatter contract."""
    if not content.strip():
        return "Content cannot be empty."
    normalized = content.lstrip("\ufeff")
    if not normalized.startswith("---"):
        return (
            "SKILL.md must start with YAML frontmatter (---). "
            "See existing skills for format."
        )
    end_match = re.search(r"\n---\s*\n", normalized[3:])
    if not end_match:
        return (
            "SKILL.md frontmatter is not closed. Ensure you have a closing "
            "'---' line."
        )
    yaml_content = normalized[3 : end_match.start() + 3]
    try:
        frontmatter = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        return f"YAML frontmatter parse error: {exc}"
    if not isinstance(frontmatter, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    if "name" not in frontmatter:
        return "Frontmatter must include 'name' field."
    if "description" not in frontmatter:
        return "Frontmatter must include 'description' field."
    description = str(frontmatter["description"])
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    prompt_description = description.strip().strip("'\"")
    if new_skill and len(prompt_description) > SKILL_PROMPT_DESC_LIMIT:
        return (
            f"Description is {len(prompt_description)} chars — new skills must "
            f"fit the {SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget "
            "(one sentence, trigger first, ends with a period)."
        )
    body_start = end_match.end() + 3
    if not normalized[body_start:].strip():
        return (
            "SKILL.md must have content after the frontmatter "
            "(instructions, procedures, etc.)."
        )
    return None


def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
    if len(content) <= MAX_SKILL_CONTENT_CHARS:
        return None
    return (
        f"{label} content is {len(content):,} characters "
        f"(limit: {MAX_SKILL_CONTENT_CHARS:,}). Consider splitting it into "
        "a smaller SKILL.md with supporting files in references/ or templates/."
    )


def _validate_file_path(file_path: str) -> Optional[str]:
    """Validate a supporting-file path without resolving untrusted parents."""
    from tools.path_security import has_traversal_component

    if not file_path:
        return "file_path is required."
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."
    path = Path(file_path)
    if path.is_absolute() or PureWindowsPath(file_path).drive:
        return "file_path must be relative to the skill directory."
    if path.name == "SKILL.md" and len(path.parts) in {1, 2}:
        return None
    if not path.parts or path.parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(path.parts) < 2:
        return (
            "Provide a file path, not just a directory. "
            f"Example: '{path.parts[0]}/myfile.md'"
        )
    return None


async def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file without blocking the event loop."""
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    temporary = path.parent / f".{path.name}.hermes-{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(
            temporary,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            await handle.write(content)
            await handle.flush()
        await aiofiles.os.replace(temporary, path)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass


async def _read_text(path: Path) -> str:
    async with aiofiles.open(path, encoding="utf-8", newline="") as handle:
        return await handle.read()


async def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """Find a local or configured external skill by its directory name."""
    for skills_root in await get_all_skills_dirs():
        if not await aiofiles.os.path.isdir(skills_root):
            continue
        async for skill_md in iter_skill_index_files(skills_root, "SKILL.md"):
            if await is_excluded_skill_path(skill_md, root=skills_root):
                continue
            if skill_md.parent.name == name:
                return {"path": skill_md.parent, "root": skills_root}
    return None


async def _resolve_skill_target(
    skill_dir: Path,
    file_path: str,
) -> Tuple[Optional[Path], Optional[str]]:
    target = skill_dir / file_path
    try:
        resolved_skill = Path(await _realpath(skill_dir))
        resolved_target = Path(await _realpath(target))
        resolved_target.relative_to(resolved_skill)
    except (OSError, ValueError) as exc:
        return None, f"Path escapes allowed directory: {exc}"
    return target, None


async def _is_path_redirect(path: Path) -> bool:
    try:
        return bool(
            await aiofiles.os.path.islink(path)
            or await _is_junction(path)
        )
    except OSError:
        return True


async def _containing_skills_root(skill_dir: Path) -> Optional[Path]:
    resolved_skill = Path(await _realpath(skill_dir))
    for root in await get_all_skills_dirs():
        try:
            resolved_root = Path(await _realpath(root))
            relative = resolved_skill.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        if relative.parts:
            return root
    return None


async def _validate_delete_target(skill_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    if await _is_path_redirect(skill_dir):
        return None, (
            f"Refusing to delete '{skill_dir}': the skill directory is a "
            "symlink/junction. Remove the link target manually if intended."
        )
    root = await _containing_skills_root(skill_dir)
    if root is None:
        return None, (
            f"Refusing to delete '{skill_dir}': path does not resolve inside "
            "any known skills root."
        )
    if Path(await _realpath(skill_dir)) == Path(await _realpath(root)):
        return None, (
            f"Refusing to delete '{skill_dir}': resolves to the skills root "
            "itself, which would remove every installed skill."
        )
    return root, None


async def _remove_tree(path: Path) -> None:
    """Remove one verified skill tree without following symlinks."""
    if await _is_path_redirect(path) or not await aiofiles.os.path.isdir(path):
        await aiofiles.os.remove(path)
        return
    for name in await aiofiles.os.listdir(path):
        await _remove_tree(path / name)
    await aiofiles.os.rmdir(path)


async def _remove_tree_fully(path: Path) -> None:
    """Finish an accepted delete even when the caller is cancelled mid-cleanup."""
    try:
        await _remove_tree(path)
    except asyncio.CancelledError:
        await asyncio.shield(_remove_tree(path))
        raise


async def _cleanup_empty_parent(path: Path, stop: Path) -> None:
    if path == stop or not await aiofiles.os.path.isdir(path):
        return
    if await aiofiles.os.listdir(path):
        return
    await aiofiles.os.rmdir(path)


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    return (
        f"Skill '{name}' not found. Use skills_list() to see available skills."
        f"{suffix}"
    )


def _add_description_prompt_preview(result: Dict[str, Any], content: str) -> None:
    frontmatter, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(frontmatter):
        result["system_prompt_preview"] = (
            f'System prompt will show: "{extract_skill_description(frontmatter)}" '
            f"— keep the trigger self-contained in the first "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars."
        )


async def _create_skill(
    name: str,
    content: str,
    category: str = None,
) -> Dict[str, Any]:
    if error := _validate_name(name):
        return {"success": False, "error": error}
    if error := _validate_category(category):
        return {"success": False, "error": error}
    if error := _validate_frontmatter(content, new_skill=True):
        return {"success": False, "error": error}
    if error := _validate_content_size(content):
        return {"success": False, "error": error}
    if existing := await _find_skill(name):
        return {
            "success": False,
            "error": f"A skill named '{name}' already exists at {existing['path']}.",
        }

    local_root = _skills_dir()
    skill_dir = local_root / category / name if category else local_root / name
    skill_md = skill_dir / "SKILL.md"
    await _atomic_write_text(skill_md, content)

    description = ""
    try:
        frontmatter_end = re.search(r"\n---\s*\n", content[3:])
        if frontmatter_end:
            parsed = yaml.safe_load(content[3 : frontmatter_end.start() + 3])
            description = str(parsed.get("description", ""))[:120]
    except Exception:
        pass

    result: Dict[str, Any] = {
        "success": True,
        "message": f"Skill '{name}' created.",
        "path": str(skill_dir.relative_to(local_root)),
        "skill_md": str(skill_md),
        "_change": {"description": description},
        "hint": (
            "To add reference files, templates, or scripts, use "
            "skill_manage(action='write_file', "
            f"name='{name}', file_path='references/example.md', "
            "file_content='...')"
        ),
    }
    if category:
        result["category"] = category
    _add_description_prompt_preview(result, content)
    return result


async def _edit_skill(name: str, content: str) -> Dict[str, Any]:
    if error := _validate_frontmatter(content):
        return {"success": False, "error": error}
    if error := _validate_content_size(content):
        return {"success": False, "error": error}
    existing = await _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    skill_md = existing["path"] / "SKILL.md"
    await _atomic_write_text(skill_md, content)

    description = ""
    try:
        frontmatter_end = re.search(r"\n---\s*\n", content[3:])
        if frontmatter_end:
            parsed = yaml.safe_load(content[3 : frontmatter_end.start() + 3])
            description = str(parsed.get("description", ""))[:120]
    except Exception:
        pass

    result = {
        "success": True,
        "message": f"Skill '{name}' updated (full rewrite).",
        "path": str(existing["path"]),
        "_change": {"description": description},
    }
    _add_description_prompt_preview(result, content)
    return result


async def _patch_skill(
    name: str,
    old_string: str,
    new_string: str,
    file_path: str = None,
    replace_all: bool = False,
) -> Dict[str, Any]:
    existing = await _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    skill_dir = existing["path"]
    if file_path:
        if error := _validate_file_path(file_path):
            return {"success": False, "error": error}
        target, error = await _resolve_skill_target(skill_dir, file_path)
        if error:
            return {"success": False, "error": error}
        assert target is not None
    else:
        target = skill_dir / "SKILL.md"
    if not await aiofiles.os.path.isfile(target):
        return {
            "success": False,
            "error": f"File not found: {target.relative_to(skill_dir)}",
        }

    content = await _read_text(target)
    from tools.fuzzy_match import format_no_match_hint, fuzzy_find_and_replace

    new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
        content,
        old_string,
        new_string,
        replace_all,
    )
    if match_error:
        return {
            "success": False,
            "error": match_error
            + format_no_match_hint(match_error, match_count, old_string, content),
            "file_preview": content[:500] + ("..." if len(content) > 500 else ""),
        }
    label = file_path or "SKILL.md"
    if error := _validate_content_size(new_content, label=label):
        return {"success": False, "error": error}
    if not file_path and (error := _validate_frontmatter(new_content)):
        return {
            "success": False,
            "error": f"Patch would break SKILL.md structure: {error}",
        }
    await _atomic_write_text(target, new_content)
    return {
        "success": True,
        "message": (
            f"Patched {label} in skill '{name}' "
            f"({match_count} replacement{'s' if match_count > 1 else ''})."
        ),
        "_change": {
            "old": old_string[:200] + ("…" if len(old_string) > 200 else ""),
            "new": new_string[:200] + ("…" if len(new_string) > 200 else ""),
        },
    }


async def _delete_skill(
    name: str,
    absorbed_into: Optional[str] = None,
) -> Dict[str, Any]:
    existing = await _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    absorbed_target = absorbed_into.strip() if isinstance(absorbed_into, str) else ""
    if absorbed_target:
        if absorbed_target == name:
            return {
                "success": False,
                "error": (
                    f"absorbed_into='{absorbed_target}' cannot equal the skill "
                    "being deleted."
                ),
            }
        if not await _find_skill(absorbed_target):
            return {
                "success": False,
                "error": (
                    f"absorbed_into='{absorbed_target}' does not exist. Create or "
                    "patch the umbrella skill first, then retry the delete."
                ),
            }

    skill_dir = existing["path"]
    skills_root, error = await _validate_delete_target(skill_dir)
    if error:
        return {"success": False, "error": error}
    assert skills_root is not None
    tombstone = skill_dir.parent / f".{skill_dir.name}.delete-{uuid.uuid4().hex}"
    await aiofiles.os.replace(skill_dir, tombstone)
    await _remove_tree_fully(tombstone)
    await _cleanup_empty_parent(skill_dir.parent, skills_root)
    message = f"Skill '{name}' deleted."
    if absorbed_target:
        message += f" Content absorbed into '{absorbed_target}'."
    return {"success": True, "message": message}


async def _write_file(
    name: str,
    file_path: str,
    file_content: str,
) -> Dict[str, Any]:
    if error := _validate_file_path(file_path):
        return {"success": False, "error": error}
    content_bytes = len(file_content.encode("utf-8"))
    if content_bytes > MAX_SKILL_FILE_BYTES:
        return {
            "success": False,
            "error": (
                f"File content is {content_bytes:,} bytes "
                f"(limit: {MAX_SKILL_FILE_BYTES:,} bytes / 1 MiB)."
            ),
        }
    if error := _validate_content_size(file_content, label=file_path):
        return {"success": False, "error": error}
    existing = await _find_skill(name)
    if not existing:
        return {
            "success": False,
            "error": _skill_not_found_error(
                name,
                " Create it first with action='create'.",
            ),
        }
    target, error = await _resolve_skill_target(existing["path"], file_path)
    if error:
        return {"success": False, "error": error}
    assert target is not None
    await _atomic_write_text(target, file_content)
    return {
        "success": True,
        "message": f"File '{file_path}' written to skill '{name}'.",
        "path": str(target),
    }


async def _list_supporting_files(skill_dir: Path) -> list[str]:
    files: list[str] = []
    for subdir in sorted(ALLOWED_SUBDIRS):
        directory = skill_dir / subdir
        if not await aiofiles.os.path.isdir(directory):
            continue
        stack = [directory]
        while stack:
            current = stack.pop()
            for name in await aiofiles.os.listdir(current):
                candidate = current / name
                if await aiofiles.os.path.isdir(candidate):
                    stack.append(candidate)
                elif await aiofiles.os.path.isfile(candidate):
                    files.append(str(candidate.relative_to(skill_dir)))
    return sorted(files)


async def _remove_file(name: str, file_path: str) -> Dict[str, Any]:
    if error := _validate_file_path(file_path):
        return {"success": False, "error": error}
    existing = await _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    skill_dir = existing["path"]
    target, error = await _resolve_skill_target(skill_dir, file_path)
    if error:
        return {"success": False, "error": error}
    assert target is not None
    if not await aiofiles.os.path.isfile(target):
        available = await _list_supporting_files(skill_dir)
        return {
            "success": False,
            "error": f"File '{file_path}' not found in skill '{name}'.",
            "available_files": available or None,
        }
    await aiofiles.os.remove(target)
    await _cleanup_empty_parent(target.parent, skill_dir)
    return {
        "success": True,
        "message": f"File '{file_path}' removed from skill '{name}'.",
    }


async def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
) -> str:
    """Manage user skills and return the upstream JSON result shape."""
    async with _skill_write_lock:
        if action == "create":
            if not content:
                return tool_error(
                    "content is required for 'create'. Provide the full "
                    "SKILL.md text (frontmatter + body).",
                    success=False,
                )
            result = await _create_skill(name, content, category)
        elif action == "edit":
            if not content:
                return tool_error(
                    "content is required for 'edit'. Provide the full updated "
                    "SKILL.md text.",
                    success=False,
                )
            result = await _edit_skill(name, content)
        elif action == "patch":
            if not old_string:
                return tool_error(
                    "old_string is required for 'patch'. Provide the text to find.",
                    success=False,
                )
            if new_string is None:
                return tool_error(
                    "new_string is required for 'patch'. Use empty string to "
                    "delete matched text.",
                    success=False,
                )
            result = await _patch_skill(
                name,
                old_string,
                new_string,
                file_path,
                replace_all,
            )
        elif action == "delete":
            result = await _delete_skill(name, absorbed_into=absorbed_into)
        elif action == "write_file":
            if not file_path:
                return tool_error(
                    "file_path is required for 'write_file'. Example: "
                    "'references/api-guide.md'",
                    success=False,
                )
            if file_content is None:
                return tool_error(
                    "file_content is required for 'write_file'.",
                    success=False,
                )
            result = await _write_file(name, file_path, file_content)
        elif action == "remove_file":
            if not file_path:
                return tool_error(
                    "file_path is required for 'remove_file'.",
                    success=False,
                )
            result = await _remove_file(name, file_path)
        else:
            result = {
                "success": False,
                "error": (
                    f"Unknown action '{action}'. Use: create, edit, patch, "
                    "delete, write_file, remove_file"
                ),
            }

        if result.get("success"):
            from agent.prompt_builder import clear_skills_system_prompt_cache
            from tools.skills_tool import _SKILLS_CACHE

            clear_skills_system_prompt_cache(clear_snapshot=True)
            _SKILLS_CACHE.clear()
        return json.dumps(result, ensure_ascii=False)


SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    "description": (
        "Manage skills (create, update, delete). Skills are procedural memory "
        "for reusable approaches. "
        f"New skills go to {display_hermes_home()}/skills/; existing local or "
        "configured external skills are modified in place.\n\n"
        "Actions: create (full SKILL.md + optional category), patch "
        "(old_string/new_string — preferred for fixes), edit (full SKILL.md "
        "rewrite — major overhauls only), delete, write_file, remove_file.\n\n"
        "Create after a difficult workflow succeeds or when the user asks to "
        "save a procedure. Patch a loaded skill when its instructions are "
        "stale or incomplete. Confirm with the user before creating or deleting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create",
                    "patch",
                    "edit",
                    "delete",
                    "write_file",
                    "remove_file",
                ],
                "description": "The action to perform.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Skill name (lowercase, hyphens/underscores, max 64 chars). "
                    "Must match an existing skill except for create."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Full SKILL.md content. Required for create and edit."
                ),
            },
            "old_string": {
                "type": "string",
                "description": "Text to find for patch.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text for patch; may be empty.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every match instead of requiring one.",
            },
            "category": {
                "type": "string",
                "description": "Optional category for create.",
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Supporting file path under references/, templates/, "
                    "scripts/, or assets/. Patch may also target SKILL.md."
                ),
            },
            "file_content": {
                "type": "string",
                "description": "Content required by write_file.",
            },
            "absorbed_into": {
                "type": "string",
                "description": (
                    "For delete, optionally record the existing umbrella skill "
                    "that absorbed this skill's content."
                ),
            },
        },
        "required": ["action", "name"],
    },
}


async def _handle_skill_manage(args: dict, **_kwargs) -> str:
    return await skill_manage(
        action=args.get("action", ""),
        name=args.get("name", ""),
        content=args.get("content"),
        category=args.get("category"),
        file_path=args.get("file_path"),
        file_content=args.get("file_content"),
        old_string=args.get("old_string"),
        new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False),
        absorbed_into=args.get("absorbed_into"),
    )


registry.register(
    name="skill_manage",
    toolset="skills",
    schema=SKILL_MANAGE_SCHEMA,
    handler=_handle_skill_manage,
    emoji="📝",
)
