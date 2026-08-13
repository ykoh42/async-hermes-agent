"""Skill bundles — aliases that load multiple skills under one slash command.

A skill bundle is a small YAML file that names a set of skills to load
together. Invoking ``/<bundle-name>`` from the CLI or gateway loads every
referenced skill's full content into a single user message, the same way
``/<skill-name>`` does — but for N skills at once.

Storage
-------
Bundles live in ``~/.hermes/skill-bundles/*.yaml`` (and the equivalent
profile-aware directory under ``HERMES_HOME``). Each file looks like::

    name: backend-dev
    description: Backend feature work — code review, testing, PR workflow.
    skills:
      - github-code-review
      - test-driven-development
      - github-pr-workflow
    instruction: |
      Optional extra guidance to inject above the skill bodies.

The file's stem is treated as a fallback name when ``name:`` is absent, so
dropping a YAML into the directory is enough to register a new bundle.

Conflict resolution
-------------------
If a bundle and a skill share the same slash name, the bundle wins. The
slash command dispatch checks bundles first, then falls back to skills.
This is the intended behavior — a user who names a bundle ``research``
explicitly wants ``/research`` to mean their bundle, not whatever skill
happens to share the slug.

Public API
----------
- :func:`get_skill_bundles` — return ``{"/slug": bundle_info}``
- :func:`resolve_bundle_command_key` — map a user-typed command to its slug
- :func:`build_bundle_invocation_message` — produce the full user message
- :func:`reload_bundles` — re-scan disk and return a diff
- :func:`list_bundles` — return rich info for display (``hermes bundles``)
- :func:`save_bundle` / :func:`delete_bundle` — file-level operations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import os
import re
import threading
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiofiles.os
import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Slug normalization — matches agent/skill_commands.py so a bundle and a
# skill called "Foo Bar" both resolve to "/foo-bar".
_BUNDLE_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_BUNDLE_MULTI_HYPHEN = re.compile(r"-{2,}")

_bundles_cache: Dict[str, Dict[str, Any]] = {}
_bundles_cache_mtime: Optional[float] = None
_bundles_cache_by_dir: Dict[
    str,
    Tuple[Dict[str, Dict[str, Any]], Optional[float]],
] = {}
_bundles_cache_projection_key: str | None = None
_BUNDLE_CACHE_GUARD = threading.RLock()


@dataclass
class _BundleScanClaim:
    finished: bool = False
    waiters: list[
        tuple[
            weakref.ReferenceType[asyncio.AbstractEventLoop],
            weakref.ReferenceType[asyncio.Future[None]],
        ]
    ] = field(default_factory=list)


_BUNDLE_SCAN_CLAIMS: Dict[str, _BundleScanClaim] = {}


def _bundles_dir() -> Path:
    """Return the canonical bundles directory under HERMES_HOME.

    Honors ``HERMES_BUNDLES_DIR`` for tests; falls back to
    ``<HERMES_HOME>/skill-bundles``.
    """
    override = os.environ.get("HERMES_BUNDLES_DIR")
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "skill-bundles"


async def _bundle_cache_key() -> str:
    """Key the cache by the physical bundle directory."""
    realpath = aiofiles.os.wrap(os.path.realpath)
    return os.path.normcase(str(await realpath(os.fspath(_bundles_dir()))))


async def _active_bundle_cache() -> tuple[
    str,
    Dict[str, Dict[str, Any]],
    Optional[float],
]:
    """Project the active profile cache onto the upstream private globals."""
    global _bundles_cache, _bundles_cache_mtime, _bundles_cache_projection_key
    cache_key = await _bundle_cache_key()
    with _BUNDLE_CACHE_GUARD:
        if (
            _bundles_cache_projection_key == cache_key
            and not _bundles_cache
            and _bundles_cache_mtime is None
        ):
            _bundles_cache_by_dir.pop(cache_key, None)
        cached = _bundles_cache_by_dir.get(cache_key)
        if cached is None:
            return cache_key, {}, None
        _bundles_cache, _bundles_cache_mtime = cached
        _bundles_cache_projection_key = cache_key
        return cache_key, _bundles_cache, _bundles_cache_mtime


def _store_bundle_cache(
    key: str,
    bundles: Dict[str, Dict[str, Any]],
    mtime: Optional[float],
) -> Dict[str, Dict[str, Any]]:
    global _bundles_cache, _bundles_cache_mtime, _bundles_cache_projection_key
    with _BUNDLE_CACHE_GUARD:
        _bundles_cache = bundles
        _bundles_cache_mtime = mtime
        _bundles_cache_projection_key = key
        _bundles_cache_by_dir[key] = (bundles, mtime)
    return bundles


def _claim_bundle_scan(
    cache_key: str,
) -> tuple[bool, _BundleScanClaim]:
    """Claim one profile scan without retaining its event loop."""
    with _BUNDLE_CACHE_GUARD:
        claim = _BUNDLE_SCAN_CLAIMS.get(cache_key)
        if claim is None or claim.finished:
            claim = _BundleScanClaim()
            _BUNDLE_SCAN_CLAIMS[cache_key] = claim
            return True, claim
        return False, claim


def _settle_bundle_scan_waiter(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


async def _wait_for_bundle_scan(claim: _BundleScanClaim) -> None:
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()
    waiter_refs = (weakref.ref(loop), weakref.ref(waiter))
    with _BUNDLE_CACHE_GUARD:
        if claim.finished:
            return
        claim.waiters.append(waiter_refs)
    try:
        await waiter
    finally:
        with _BUNDLE_CACHE_GUARD:
            try:
                claim.waiters.remove(waiter_refs)
            except ValueError:
                pass


def _finish_bundle_scan(
    cache_key: str,
    claim: _BundleScanClaim,
) -> None:
    """Release a profile scan and wake cross-loop waiters."""
    with _BUNDLE_CACHE_GUARD:
        if _BUNDLE_SCAN_CLAIMS.get(cache_key) is claim:
            _BUNDLE_SCAN_CLAIMS.pop(cache_key, None)
        claim.finished = True
        waiters = tuple(claim.waiters)
        claim.waiters.clear()
    for loop_ref, waiter_ref in waiters:
        loop = loop_ref()
        waiter = waiter_ref()
        if loop is None or waiter is None:
            continue
        try:
            loop.call_soon_threadsafe(_settle_bundle_scan_waiter, waiter)
        except RuntimeError:
            continue


def _slugify(name: str) -> str:
    cmd = name.lower().replace(" ", "-").replace("_", "-")
    cmd = _BUNDLE_INVALID_CHARS.sub("", cmd)
    cmd = _BUNDLE_MULTI_HYPHEN.sub("-", cmd).strip("-")
    return cmd


async def _iter_bundle_files() -> List[Path]:
    base = _bundles_dir()
    if not await aiofiles.os.path.isdir(base):
        return []
    names = await aiofiles.os.listdir(base)
    files: List[Path] = []
    for suffix in (".yaml", ".yml"):
        files.extend(
            sorted(base / name for name in names if name.endswith(suffix))
        )
    return files


async def _max_mtime(files: List[Path]) -> float:
    """Highest mtime across the bundle files plus the dir itself.

    Watching the directory mtime catches deletions; watching individual
    files catches edits. Together they're a cheap freshness check.
    """
    base = _bundles_dir()
    mtimes = []
    if await aiofiles.os.path.exists(base):
        try:
            mtimes.append((await aiofiles.os.stat(base)).st_mtime)
        except OSError:
            pass
    for f in files:
        try:
            mtimes.append((await aiofiles.os.stat(f)).st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


async def _load_bundle_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a single bundle YAML file. Returns ``None`` on any error.

    Errors are logged at WARNING level. We don't raise — a broken bundle
    shouldn't take down slash command discovery.
    """
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            raw = await handle.read()
    except OSError as exc:
        logger.warning("Could not read bundle %s: %s", path, exc)
        return None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in bundle %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Bundle %s is not a mapping; skipping", path)
        return None

    name = str(data.get("name") or path.stem).strip()
    if not name:
        logger.warning("Bundle %s has no name; skipping", path)
        return None

    skills = data.get("skills") or []
    if not isinstance(skills, list) or not skills:
        logger.warning("Bundle %s has no skills list; skipping", path)
        return None
    skills = [str(s).strip() for s in skills if str(s).strip()]
    if not skills:
        logger.warning("Bundle %s has empty skills list; skipping", path)
        return None

    description = str(data.get("description") or "").strip()
    instruction = str(data.get("instruction") or "").strip()

    slug = _slugify(name)
    if not slug:
        logger.warning("Bundle %s yielded empty slug; skipping", path)
        return None

    return {
        "name": name,
        "slug": slug,
        "description": description or f"Load {len(skills)} skills as a bundle",
        "skills": skills,
        "instruction": instruction,
        "path": str(path),
    }


async def scan_bundles() -> Dict[str, Dict[str, Any]]:
    """Scan the bundles directory and rebuild the cache.

    Returns the same mapping as :func:`get_skill_bundles` — ``"/slug"`` →
    bundle info dict. Later bundles with a duplicate slug are skipped with
    a warning (first wins, alphabetical order).
    """
    cache_key = await _bundle_cache_key()
    owner, claim = _claim_bundle_scan(cache_key)
    if not owner:
        await _wait_for_bundle_scan(claim)
        return await scan_bundles()
    try:
        files = await _iter_bundle_files()
        out: Dict[str, Dict[str, Any]] = {}
        for f in files:
            info = await _load_bundle_file(f)
            if not info:
                continue
            bundle_key = f"/{info['slug']}"
            if bundle_key in out:
                logger.warning(
                    "Duplicate bundle slug %s from %s; keeping %s",
                    bundle_key, f, out[bundle_key]["path"],
                )
                continue
            out[bundle_key] = info
        return _store_bundle_cache(cache_key, out, await _max_mtime(files))
    finally:
        _finish_bundle_scan(cache_key, claim)


async def get_skill_bundles() -> Dict[str, Dict[str, Any]]:
    """Return the current bundle mapping, rescanning when disk changed.

    Cheap to call repeatedly: only rescans when the bundles directory or
    any bundle file's mtime is newer than the cached snapshot.
    """
    cache_key = await _bundle_cache_key()
    with _BUNDLE_CACHE_GUARD:
        active_scan = _BUNDLE_SCAN_CLAIMS.get(cache_key)
    if active_scan is not None:
        await _wait_for_bundle_scan(active_scan)
        return await get_skill_bundles()
    files = await _iter_bundle_files()
    current_mtime = await _max_mtime(files)
    _key, cached, cached_mtime = await _active_bundle_cache()
    if not cached or cached_mtime != current_mtime:
        return await scan_bundles()
    return cached


async def resolve_bundle_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed command to its canonical bundle slash key.

    Hyphens and underscores are treated interchangeably to mirror the
    skill-command behavior (Telegram converts hyphens to underscores in
    bot command names).
    """
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in await get_skill_bundles() else None


async def reload_bundles() -> Dict[str, Any]:
    """Re-scan the bundles directory and return a diff.

    Mirrors :func:`agent.skill_commands.reload_skills` so callers can use
    the same display logic. Returns a dict with ``added``, ``removed``,
    ``unchanged``, and ``total`` keys.
    """
    def _snapshot(cmds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        return {k.lstrip("/"): (v or {}).get("description", "") for k, v in cmds.items()}

    _key, cached, _mtime = await _active_bundle_cache()
    before = _snapshot(cached)
    new = await scan_bundles()
    after = _snapshot(new)

    added_names = sorted(set(after) - set(before))
    removed_names = sorted(set(before) - set(after))
    unchanged = sorted(set(after) & set(before))

    return {
        "added": [{"name": n, "description": after[n]} for n in added_names],
        "removed": [{"name": n, "description": before[n]} for n in removed_names],
        "unchanged": unchanged,
        "total": len(after),
    }


async def list_bundles() -> List[Dict[str, Any]]:
    """Return a sorted list of bundle info dicts for display."""
    bundles = await get_skill_bundles()
    return sorted(bundles.values(), key=lambda b: b["slug"])


async def build_bundle_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    platform: str | None = None,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """Build the user message content for a bundle slash command invocation.

    Returns ``(message, loaded_skill_names, missing_skill_names)`` or
    ``None`` if the bundle wasn't found.

    A bundle that references skills the user doesn't have installed still
    loads — the agent gets a note about which ones were skipped. This is
    the same forgiving stance ``build_preloaded_skills_prompt`` uses for
    ``-s`` CLI preloading.

    Disabled skills are also skipped: bundles load members via
    ``_load_skill_payload`` directly, bypassing the scan-time disabled
    filter in ``get_skill_commands()``, so the disabled list must be
    re-applied here.  ``platform`` scopes the check to a specific
    platform's ``skills.platform_disabled`` config (gateway dispatch
    passes it explicitly because the gateway handles multiple platforms
    in one process); when *None*, the platform resolves from session env
    vars and the global disabled list still applies.  Mirrors the
    stacked-skill gate in gateway dispatch (#58888).
    """
    bundles = await get_skill_bundles()
    info = bundles.get(cmd_key)
    if not info:
        return None

    # Late import to avoid pulling tools/* at module import time and to
    # keep skill_bundles cheap to import in test environments.
    from agent.skill_commands import _load_skill_payload, _build_skill_message

    try:
        from tools.skills_tool import _get_disabled_skill_names

        disabled_names = await _get_disabled_skill_names(platform=platform)
    except Exception:
        disabled_names = set()

    loaded_names: List[str] = []
    missing: List[str] = []
    disabled: List[str] = []
    skill_blocks: List[str] = []
    seen: set[str] = set()

    bundle_name = info["name"]
    skills = info["skills"]
    extra_instruction = info.get("instruction") or ""

    for skill_id in skills:
        identifier = (skill_id or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = await _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue
        loaded_skill, skill_dir, skill_name = loaded

        # Per-platform / global disabled gate. Checked against the loaded
        # skill's canonical name (identifiers may be paths or aliases).
        if skill_name in disabled_names or identifier in disabled_names:
            disabled.append(skill_name or identifier)
            continue

        activation_note = (
            f'[Loaded as part of the "{bundle_name}" skill bundle.]'
        )
        skill_blocks.append(
            await _build_skill_message(
                loaded_skill,
                skill_dir,
                activation_note,
                session_id=task_id,
            )
        )
        loaded_names.append(skill_name)

    if not skill_blocks:
        return None

    # Header — tells the agent this is a bundle, lists the skills, and
    # provides any author-supplied instruction.
    header_lines = [
        f'[IMPORTANT: The user has invoked the "{bundle_name}" skill bundle, '
        f"loading {len(loaded_names)} skills together. Treat every skill below "
        "as active guidance for this turn.]",
        "",
        f"Bundle: {bundle_name}",
        f"Skills loaded: {', '.join(loaded_names)}",
    ]
    if missing:
        header_lines.append(f"Skills missing (skipped): {', '.join(missing)}")
    if disabled:
        header_lines.append(
            f"Skills disabled for this platform (skipped): {', '.join(disabled)}"
        )
    if extra_instruction:
        header_lines.extend(["", f"Bundle instruction: {extra_instruction}"])
    if user_instruction:
        header_lines.extend(
            ["", f"User instruction: {user_instruction}"]
        )

    header = "\n".join(header_lines)
    return ("\n\n".join([header, *skill_blocks]), loaded_names, missing)


# ---------------------------------------------------------------------------
# File-level CRUD helpers — used by `hermes bundles` CLI subcommand.
# ---------------------------------------------------------------------------


def bundle_path_for(name: str) -> Path:
    """Return the canonical filesystem path for a bundle name."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Bundle name {name!r} normalizes to an empty slug")
    return _bundles_dir() / f"{slug}.yaml"


async def save_bundle(
    name: str,
    skills: List[str],
    description: str = "",
    instruction: str = "",
    overwrite: bool = False,
) -> Path:
    """Write a bundle to disk and invalidate the cache.

    Raises ``FileExistsError`` if the target exists and ``overwrite`` is
    False. Raises ``ValueError`` if the inputs are unusable.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Bundle name is required")
    cleaned_skills = [str(s).strip() for s in skills if str(s).strip()]
    if not cleaned_skills:
        raise ValueError("Bundle must reference at least one skill")

    path = bundle_path_for(name)
    if await aiofiles.os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Bundle already exists at {path}")

    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    payload: Dict[str, Any] = {"name": name, "skills": cleaned_skills}
    if description:
        payload["description"] = description
    if instruction:
        payload["instruction"] = instruction

    async with aiofiles.open(path, "w", encoding="utf-8") as handle:
        await handle.write(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        )
    await scan_bundles()  # refresh cache
    return path


async def delete_bundle(name: str) -> Path:
    """Delete a bundle by name. Returns the deleted path.

    Raises ``FileNotFoundError`` if the bundle doesn't exist.
    """
    path = bundle_path_for(name)
    if not await aiofiles.os.path.exists(path):
        raise FileNotFoundError(f"No bundle at {path}")
    await aiofiles.os.unlink(path)
    await scan_bundles()
    return path


async def get_bundle(name: str) -> Optional[Dict[str, Any]]:
    """Look up a bundle by name (slug-normalized)."""
    slug = _slugify(name)
    return (await get_skill_bundles()).get(f"/{slug}")
