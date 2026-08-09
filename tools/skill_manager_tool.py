#!/usr/bin/env python3
"""Native-async skill creation and editing.

This is the retained library form of Hermes' ``skill_manage`` tool.  It keeps
the upstream public name, arguments, action names, result shape, and file
location while performing every filesystem operation through an awaited I/O
boundary.  CLI approval and organization-sync workflows remain outside this
module.
"""

from __future__ import annotations

import asyncio
import contextvars
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
_background_review_read_paths: contextvars.ContextVar[frozenset[str]] = (
    contextvars.ContextVar("background_review_read_paths", default=frozenset())
)


async def mark_background_review_skill_read(path: Path) -> None:
    """Record that the active review fork loaded one exact skill file."""
    try:
        from tools.skill_provenance import is_background_review

        if not is_background_review():
            return
    except Exception:
        return
    try:
        resolved = str(await _realpath(path.expanduser()))
    except (OSError, ValueError):
        resolved = str(path)
    paths = set(_background_review_read_paths.get())
    paths.add(resolved)
    _background_review_read_paths.set(frozenset(paths))


async def _background_review_has_read(path: Path) -> bool:
    try:
        resolved = str(await _realpath(path.expanduser()))
    except (OSError, ValueError):
        resolved = str(path)
    return resolved in _background_review_read_paths.get()


def _reset_background_review_read_marks() -> None:
    """Clear read-before-write marks for the current review context."""
    _background_review_read_paths.set(frozenset())


async def _pinned_guard(name: str) -> Optional[str]:
    """Refuse only irreversible foreground deletion of a pinned skill."""
    try:
        from tools import skill_usage

        if (await skill_usage.get_record(name)).get("pinned"):
            return (
                f"Skill '{name}' is pinned and cannot be deleted by "
                "skill_manage. Unpin it explicitly before deletion. Patches "
                "and edits remain allowed; only deletion is blocked."
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("pinned-skill lookup failed for %s", name, exc_info=True)
    return None


async def _background_review_write_guard(
    name: str,
    skill_dir: Path,
    action: str,
) -> Optional[Dict[str, Any]]:
    """Restrict autonomous writes to local, curator-managed skills."""
    try:
        from tools.skill_provenance import is_background_review

        if not is_background_review():
            return None
    except Exception:
        return None

    try:
        from tools import skill_usage

        if (await skill_usage.get_record(name)).get("pinned"):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for pinned skill "
                    f"'{name}': pinned skills are off-limits to autonomous "
                    "maintenance."
                ),
            }
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("pinned skill guard lookup failed for %s", name, exc_info=True)

    try:
        from agent.skill_utils import is_external_skill_path

        if await is_external_skill_path(skill_dir):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for skill '{name}': "
                    "the skill lives in skills.external_dirs, which are "
                    "externally owned and read-only to autonomous curation."
                ),
            }
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("external skill guard lookup failed for %s", name, exc_info=True)

    try:
        from tools import skill_usage

        if skill_usage.is_protected_builtin(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for protected "
                    f"built-in skill '{name}'."
                ),
            }
        if await skill_usage.is_hub_installed(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for hub-installed "
                    f"skill '{name}'."
                ),
            }
        if await skill_usage.is_bundled(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for bundled "
                    f"skill '{name}'."
                ),
            }
        usage_record = (await skill_usage.load_usage()).get(name)
        if not skill_usage._is_curator_managed_record(usage_record):
            detail = (
                f"created_by={usage_record.get('created_by')!r}"
                if isinstance(usage_record, dict)
                else "no usage record"
            )
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for skill '{name}': "
                    f"the skill is not curator-managed ({detail}). User-owned "
                    "skills are off-limits to autonomous curation. Explicitly "
                    "adopt the skill before autonomous maintenance."
                ),
            }
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("owned skill guard lookup failed for %s", name, exc_info=True)
        return {
            "success": False,
            "error": (
                f"Refusing background curator {action} for skill '{name}': "
                "agent ownership could not be verified because the provenance "
                "record is unavailable or unreadable."
            ),
        }
    return None


async def _background_review_read_before_write_guard(
    name: str,
    target: Path,
    action: str,
    file_label: str,
) -> Optional[Dict[str, Any]]:
    try:
        from tools.skill_provenance import is_background_review

        if not is_background_review():
            return None
    except Exception:
        return None
    if await _background_review_has_read(target):
        return None
    return {
        "success": False,
        "error": (
            f"Refusing background curator {action} for skill '{name}': the "
            f"current {file_label} content has not been loaded in this review "
            "turn. Call skill_view(name) for SKILL.md, or skill_view(name, "
            "file_path=...) for a supporting file, then retry the write using "
            "the content just returned."
        ),
        "_read_before_write_required": True,
    }


async def _background_review_preflight(
    action: str,
    name: str,
) -> Optional[Dict[str, Any]]:
    """Run the upstream ownership guard before validating write arguments."""
    if action not in {"edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    existing = await _find_skill(name)
    if not existing:
        return None
    return await _background_review_write_guard(name, existing["path"], action)


def _curator_consolidation_delete_guard(
    name: str,
    absorbed_into: Optional[str],
) -> Optional[Dict[str, Any]]:
    from tools.skill_provenance import is_background_review

    if not is_background_review() or (
        isinstance(absorbed_into, str) and absorbed_into.strip()
    ):
        return None
    return {
        "success": False,
        "error": (
            f"Refusing background curator delete of skill '{name}': the "
            "consolidation pass may only archive a skill it has absorbed into "
            "an umbrella. Pass absorbed_into=<umbrella> (the umbrella must "
            "already exist) to record a verified consolidation. Pruning a "
            "skill with no forwarding target is not permitted here — the "
            "deterministic inactivity prune handles staleness archival "
            f"separately. Keeping '{name}' active."
        ),
        "_fail_closed": True,
    }


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


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one accepted mutation before propagating caller cancellation."""
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


async def _remove_tree_fully(path: Path) -> None:
    """Finish an accepted delete even when the caller is cancelled mid-cleanup."""
    await _finish_owned_task(asyncio.create_task(_remove_tree(path)))


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
    if guard := await _background_review_write_guard(
        name,
        existing["path"],
        "edit",
    ):
        return guard
    skill_md = existing["path"] / "SKILL.md"
    if guard := await _background_review_read_before_write_guard(
        name,
        skill_md,
        "edit",
        "SKILL.md",
    ):
        return guard
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
    if guard := await _background_review_write_guard(name, skill_dir, "patch"):
        return guard
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
    if guard := await _background_review_read_before_write_guard(
        name,
        target,
        "patch",
        file_path or "SKILL.md",
    ):
        return guard

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
    if guard := await _background_review_write_guard(
        name,
        existing["path"],
        "delete",
    ):
        return guard
    if guard := _curator_consolidation_delete_guard(name, absorbed_into):
        return guard
    if pinned_error := await _pinned_guard(name):
        return {"success": False, "error": pinned_error}
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
    from tools.skill_provenance import is_background_review

    if is_background_review():
        from tools.skill_usage import archive_skill

        try:
            archived, archive_message = await archive_skill(name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "success": False,
                "error": f"failed to archive '{name}': {exc}",
            }
        if not archived:
            return {"success": False, "error": archive_message}
        message = f"Skill '{name}' archived ({archive_message})."
        if absorbed_target:
            message += f" Content absorbed into '{absorbed_target}'."
        return {"success": True, "message": message, "_archived": True}

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
    if guard := await _background_review_write_guard(
        name,
        existing["path"],
        "write_file",
    ):
        return guard
    target, error = await _resolve_skill_target(existing["path"], file_path)
    if error:
        return {"success": False, "error": error}
    assert target is not None
    if await aiofiles.os.path.exists(target):
        if guard := await _background_review_read_before_write_guard(
            name,
            target,
            "write_file",
            file_path,
        ):
            return guard
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
    if guard := await _background_review_write_guard(name, skill_dir, "remove_file"):
        return guard
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
    if guard := await _background_review_read_before_write_guard(
        name,
        target,
        "remove_file",
        file_path,
    ):
        return guard
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
    preflight = await _background_review_preflight(action, name)
    if preflight is not None:
        return json.dumps(preflight, ensure_ascii=False)

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

            await clear_skills_system_prompt_cache(clear_snapshot=True)
            _SKILLS_CACHE.clear()
            try:
                from tools import skill_usage
                from tools.skill_provenance import is_background_review

                if action == "create":
                    if is_background_review():
                        await skill_usage.mark_agent_created(name)
                elif action in {"patch", "edit", "write_file", "remove_file"}:
                    await skill_usage.bump_patch(name)
                elif action == "delete" and not result.get("_archived"):
                    await skill_usage.forget(name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "skill usage telemetry failed for %s",
                    name,
                    exc_info=True,
                )
        return json.dumps(result, ensure_ascii=False)


SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    "description": (
        "Manage skills (create, update, delete). Skills are your procedural "
        "memory — reusable approaches for recurring task types. "
        f"New skills go to {display_hermes_home()}/skills/; existing skills "
        "can be modified wherever they live.\n\n"
        "Actions: create (full SKILL.md + optional category), patch "
        "(old_string/new_string — preferred for fixes), edit (full SKILL.md "
        "rewrite — major overhauls only), delete, write_file, remove_file.\n\n"
        "On delete, pass `absorbed_into=<umbrella>` when you're merging this "
        "skill's content into another one, or `absorbed_into=\"\"` when you're "
        "pruning it with no forwarding target. This lets the curator tell "
        "consolidation from pruning without guessing so lifecycle consumers "
        "retain the declared intent. The target you name in `absorbed_into` "
        "must already exist — create/patch the umbrella first, then delete.\n\n"
        "Create when: complex task succeeded (5+ calls), errors overcome, "
        "user-corrected approach worked, non-trivial workflow discovered, "
        "or user asks you to remember a procedure.\n"
        "Update when: instructions stale/wrong, OS-specific failures, "
        "missing steps or pitfalls found during use. If you used a skill and "
        "hit issues not covered by it, patch it immediately.\n\n"
        "After difficult/iterative tasks, offer to save as a skill. Skip for "
        "simple one-offs. Confirm with user before creating/deleting.\n\n"
        "Good skills: trigger conditions, numbered steps with exact commands, "
        "pitfalls section, verification steps. Use skill_view() to see format "
        "examples.\n\n"
        "Description: long descriptions are truncated to the first 57 chars "
        "plus '...' in the system prompt skill index; longer text is visible "
        "via skills_list/skill_view. Keep the trigger self-contained in that "
        "first 57-char window: 'Use when <trigger>. <one-line behavior>.'\n\n"
        "Pinned skills are protected from deletion only — "
        "skill_manage(action='delete') will refuse until the skill is "
        "explicitly unpinned. Patches and edits go through on pinned skills so "
        "you can still improve them as pitfalls come up; pin only guards "
        "against irrecoverable loss."
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
                    "Must match an existing skill for patch/edit/delete/"
                    "write_file/remove_file."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Full SKILL.md content (YAML frontmatter + markdown body). "
                    "Required for 'create' and 'edit'. For 'edit', read the "
                    "skill first with skill_view() and provide the complete "
                    "updated text."
                ),
            },
            "old_string": {
                "type": "string",
                "description": (
                    "Text to find in the file (required for 'patch'). Must be "
                    "unique unless replace_all=true. Include enough "
                    "surrounding context to ensure uniqueness."
                ),
            },
            "new_string": {
                "type": "string",
                "description": (
                    "Replacement text (required for 'patch'). Can be empty "
                    "string to delete the matched text."
                ),
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "For 'patch': replace all occurrences instead of requiring "
                    "a unique match (default: false)."
                ),
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional category/domain for organizing the skill (e.g., "
                    "'devops', 'data-science', 'mlops'). Creates a subdirectory "
                    "grouping. Only used with 'create'."
                ),
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Path to a supporting file within the skill directory. For "
                    "'write_file'/'remove_file': required, must be under "
                    "references/, templates/, scripts/, or assets/. For "
                    "'patch': optional, defaults to SKILL.md if omitted."
                ),
            },
            "file_content": {
                "type": "string",
                "description": "Content for the file. Required for 'write_file'.",
            },
            "absorbed_into": {
                "type": "string",
                "description": (
                    "For 'delete' only — declares intent so the curator can "
                    "tell consolidation from pruning without guessing. Pass "
                    "the umbrella skill name when this skill's content was "
                    "merged into another (the target must already exist). Pass "
                    "an empty string when the skill is truly stale and being "
                    "pruned with no forwarding target. Omitting the arg on "
                    "delete is supported for backward compatibility; lifecycle "
                    "consumers then have to infer intent."
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
