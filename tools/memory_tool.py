#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import asyncio
import concurrent.futures.thread as _thread_backend_bootstrap  # noqa: F401
import errno
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any

import aiofiles
import aiofiles.os

msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

# Stable header prefixes for the system-prompt memory blocks rendered by
# MemoryStore._render_block. Exported so compression's prompt-retention check
# (agent/conversation_compression.py) can detect a leftover block for a
# target whose entries have since been emptied — keep in lockstep with
# _render_block below.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)",
    "user": "USER PROFILE (who the user is)",
}

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message


def _scan_memory_content(content: str) -> str | None:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


# Sentinel returned by ``_reload_target`` when the target file EXISTS but could
# not be read. Distinct from a drift-backup path (``str``) and from a clean
# reload (``None``): the caller must abort the mutation rather than persist over
# an unreadable file.
_READ_FAILED = object()


def _read_failed_error(path: "Path") -> dict[str, Any]:
    """Build the error dict returned when the on-disk memory file is unreadable.

    A file that exists but cannot be read is NOT an empty store. Reading it as
    ``[]`` and then persisting would rewrite the whole file from an empty entry
    list — wiping the user's memory. We refuse the write so nothing is lost.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: the file exists on disk but could "
            f"not be read right now (temporarily locked by another program, a "
            f"permission change, invalid/corrupt text encoding, or a filesystem "
            f"error). Treating an unreadable file as empty and saving would wipe "
            f"existing memory, so the write is refused. Nothing was changed — "
            f"retry in a moment."
        ),
    }


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    # After this many failed consolidation attempts (overflow / zero-match) in
    # ONE turn, stop instructing the model to "retry in this turn" and return a
    # terminal "save skipped" result so a fragile replace/add can't loop the
    # turn to budget exhaustion and suppress the user's reply (issue #42405).
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: dict[str, str] = {"memory": "", "user": ""}
        # Per-turn counter of failed at-capacity consolidation attempts; reset
        # at each turn boundary by reset_consolidation_failures() (#42405).
        self._consolidation_failures = 0
        self._write_lock = None

    def _get_write_lock(self) -> asyncio.Lock:
        """Return the per-store lock used by native async mutations."""
        lock = self._write_lock
        if lock is None:
            lock = asyncio.Lock()
            self._write_lock = lock
        return lock

    @staticmethod
    @asynccontextmanager
    async def _file_lock(path: Path):
        """Acquire the upstream per-file lock without blocking the event loop."""
        lock_path = path.with_suffix(path.suffix + ".lock")
        await aiofiles.os.makedirs(lock_path.parent, exist_ok=True)
        async with aiofiles.open(lock_path, "a+b") as handle:
            if fcntl is not None:
                while True:
                    try:
                        fcntl.flock(
                            handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                        break
                    except BlockingIOError:
                        await asyncio.sleep(0.01)
            elif msvcrt is not None:
                await handle.seek(0)
                if not await handle.read(1):
                    await handle.write(b"\0")
                    await handle.flush()
                while True:
                    try:
                        await handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN, 13, 36}:
                            raise
                        await asyncio.sleep(0.01)
            try:
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                elif msvcrt is not None:
                    try:
                        await handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: dict[str, Any]) -> dict[str, Any]:
        """Count an at-capacity consolidation failure and degrade gracefully.

        Under the per-turn cap, return ``response`` unchanged (it already tells
        the model how to self-correct + retry in this turn). Once the cap is
        exceeded, drop the retry instruction and return a TERMINAL result so the
        model stops looping memory calls and proceeds to answer the user — a
        failed memory side effect must never block the turn's reply (#42405).
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times "
                "this turn. Stop retrying memory calls — leave memory unchanged for "
                "now and continue with your reply to the user. The fact can be saved "
                "in a later turn."
            ),
        }

    async def load_from_disk(self) -> None:
        """Load the frozen memory snapshot without blocking an agent turn."""
        mem_dir = get_memory_dir()
        await aiofiles.os.makedirs(mem_dir, exist_ok=True)
        self.memory_entries = await self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = await self._read_file(mem_dir / "USER.md")
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))
        sanitized_memory = self._sanitize_entries_for_snapshot(
            self.memory_entries, "MEMORY.md"
        )
        sanitized_user = self._sanitize_entries_for_snapshot(
            self.user_entries, "USER.md"
        )
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: list[str], filename: str) -> list[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: list[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    async def _reload_target(
        self,
        target: str,
        *,
        skip_drift: bool = False,
    ):
        """Refresh a mutable target from its checked on-disk snapshot."""
        path = self._path_for(target)
        raw, read_ok = await self._read_raw_checked(path)
        if not read_ok:
            return _READ_FAILED
        backup = (
            None
            if skip_drift
            else await self._detect_external_drift(target, raw)
        )
        self._set_entries(target, list(dict.fromkeys(self._parse_entries(raw))))
        return backup

    async def save_to_disk(self, target: str) -> None:
        """Atomically persist one memory target through ``aiofiles``."""
        mem_dir = get_memory_dir()
        await aiofiles.os.makedirs(mem_dir, exist_ok=True)
        await self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> list[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: list[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    async def add(self, target: str, content: str) -> dict[str, Any]:
        """Append one memory entry with native async persistence."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        async with self._get_write_lock(), self._file_lock(self._path_for(target)):
            if await self._reload_target(target, skip_drift=True) is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            entries = self._entries_for(target)
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")
            new_entries = entries + [content]
            limit = self._char_limit(target)
            if len(ENTRY_DELIMITER.join(new_entries)) > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. Adding this entry "
                        f"({len(content)} chars) would exceed the limit. Consolidate now: "
                        "use 'replace' to merge overlapping entries into shorter ones or "
                        "'remove' stale or less important entries (see current_entries below), "
                        "then retry this add — all in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })
            self._set_entries(target, new_entries)
            await self.save_to_disk(target)
        return self._success_response(target, "Entry added.")

    async def replace(
        self,
        target: str,
        old_text: str,
        new_content: str,
    ) -> dict[str, Any]:
        """Native-async replacement with atomic file publication."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {
                "success": False,
                "error": "new_content cannot be empty. Use 'remove' to delete entries.",
            }
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        async with self._get_write_lock(), self._file_lock(self._path_for(target)):
            backup = await self._reload_target(target)
            if backup is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if backup:
                return _drift_error(self._path_for(target), backup)
            entries = self._entries_for(target)
            matches = [(index, entry) for index, entry in enumerate(entries) if old_text in entry]
            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"No entry matched '{old_text}'. Check current_entries below and retry "
                        "with the exact text of the entry you want to replace."
                    ),
                    "current_entries": entries,
                })
            if len({entry for _, entry in matches}) > 1:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": self._previews([entry for _, entry in matches]),
                }
            updated = list(entries)
            updated[matches[0][0]] = new_content
            limit = self._char_limit(target)
            total = len(ENTRY_DELIMITER.join(updated))
            if total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {total:,}/{limit:,} chars. "
                        "Shorten the new content, or 'remove' other stale or less important "
                        "entries to make room (see current_entries below), then retry — all in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })
            self._set_entries(target, updated)
            await self.save_to_disk(target)
        return self._success_response(target, "Entry replaced.")

    async def remove(self, target: str, old_text: str) -> dict[str, Any]:
        """Native-async removal with the existing ambiguity guard."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        async with self._get_write_lock(), self._file_lock(self._path_for(target)):
            backup = await self._reload_target(target)
            if backup is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if backup:
                return _drift_error(self._path_for(target), backup)
            entries = self._entries_for(target)
            matches = [(index, entry) for index, entry in enumerate(entries) if old_text in entry]
            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to remove.",
                    "current_entries": entries,
                })
            if len({entry for _, entry in matches}) > 1:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": self._previews([entry for _, entry in matches]),
                }
            updated = list(entries)
            updated.pop(matches[0][0])
            self._set_entries(target, updated)
            await self.save_to_disk(target)
        return self._success_response(target, "Entry removed.")

    async def apply_batch(
        self,
        target: str,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply a batch atomically through the native async file path."""
        if not operations:
            return {"success": False, "error": "operations list is empty."}
        for index, operation in enumerate(operations):
            action = (operation or {}).get("action")
            content = (operation or {}).get("content") or (operation or {}).get(
                "new_text"
            )
            if action in {"add", "replace"} and content:
                scan_error = _scan_memory_content(content)
                if scan_error:
                    return {"success": False, "error": f"Operation {index + 1}: {scan_error}"}

        async with self._get_write_lock(), self._file_lock(self._path_for(target)):
            backup = await self._reload_target(target)
            if backup is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if backup:
                return _drift_error(self._path_for(target), backup)
            working = list(self._entries_for(target))
            for index, operation in enumerate(operations):
                operation = operation or {}
                action = operation.get("action")
                content = (
                    operation.get("content") or operation.get("new_text") or ""
                ).strip()
                old_text = (operation.get("old_text") or "").strip()
                label = f"Operation {index + 1} ({action or 'unknown'})"
                if action == "add":
                    if not content:
                        return self._batch_error(target, f"{label}: content is required.")
                    if content not in working:
                        working.append(content)
                elif action in {"replace", "remove"}:
                    if not old_text:
                        return self._batch_error(target, f"{label}: old_text is required.")
                    matches = [pos for pos, entry in enumerate(working) if old_text in entry]
                    if not matches:
                        return self._batch_error(target, f"{label}: no entry matched '{old_text}'.")
                    if len({working[pos] for pos in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{label}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    if action == "replace":
                        if not content:
                            return self._batch_error(
                                target,
                                f"{label}: content is required (use action='remove' to delete).",
                            )
                        working[matches[0]] = content
                    else:
                        working.pop(matches[0])
                else:
                    return self._batch_error(
                        target,
                        f"{label}: unknown action. Use add, replace, or remove.",
                    )
            limit = self._char_limit(target)
            total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"After applying all {len(operations)} operations, memory would be at "
                        f"{total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                        "entries in the same batch (see current_entries below), then retry."
                    ),
                    "current_entries": self._entries_for(target),
                    "usage": f"{current:,}/{limit:,}",
                })
            self._set_entries(target, working)
            await self.save_to_disk(target)
        return self._success_response(target, f"Applied {len(operations)} operation(s).")

    def _batch_error(self, target: str, message: str) -> dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return self._consolidation_failure({
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
        })

    def format_for_system_prompt(self, target: str) -> str | None:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    @staticmethod
    def _previews(entries: list[str], width: int = 80) -> list[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str | None = None) -> dict[str, Any]:
        # A successful write means the consolidation loop made progress, so the
        # per-turn failure budget resets (the cap counts consecutive failures,
        # not lifetime ones within a turn) (#42405).
        self._consolidation_failures = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # The success response is intentionally TERMINAL: it confirms the write
        # landed and tells the model to stop. We do NOT echo the full entries
        # list here -- dumping it invites the model to "find more to fix" and
        # re-issue the same operations (observed thrash: the correct batch on
        # call 1, then 5 redundant repeats). Entries are only shown on the
        # error/over-budget paths, where the model genuinely needs them to
        # decide what to consolidate.
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: list[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"{MEMORY_BLOCK_HEADERS['user']} [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"{MEMORY_BLOCK_HEADERS['memory']} [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    async def _read_raw_checked(path: Path) -> tuple[str, bool]:
        """Read a memory file, distinguishing unreadable from an absent file."""
        if not await aiofiles.os.path.exists(path):
            return "", True
        try:
            async with aiofiles.open(path, encoding="utf-8") as handle:
                return await handle.read(), True
        except (OSError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _parse_entries(raw: str) -> list[str]:
        """Split raw memory-file text into stripped, non-empty entries."""
        if not raw.strip():
            return []
        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    async def _read_file(path: Path) -> list[str]:
        """Read entries without risking a later write after a failed read."""
        raw, read_ok = await MemoryStore._read_raw_checked(path)
        return MemoryStore._parse_entries(raw) if read_ok else []
    async def _detect_external_drift(
        self,
        target: str,
        raw: str,
    ) -> str | None:
        """Async drift guard used before a native read-modify-write commit."""
        path = self._path_for(target)
        if not raw.strip():
            return None
        parsed = [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)
        max_entry_len = max((len(entry) for entry in parsed), default=0)
        if raw.strip() == roundtrip and max_entry_len <= self._char_limit(target):
            return None
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            async with aiofiles.open(backup, "w", encoding="utf-8") as handle:
                await handle.write(raw)
        except OSError:
            return str(backup) + " (BACKUP FAILED — file unchanged on disk)"
        return str(backup)

    @staticmethod
    async def _write_file(path: Path, entries: list[str]) -> None:
        """Atomically publish memory with upstream fsync and symlink semantics."""
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        write_path = path
        if await aiofiles.os.wrap(os.path.islink)(path):
            resolved = await aiofiles.os.wrap(os.path.realpath)(path)
            if resolved:
                write_path = Path(resolved)
        temporary = path.with_name(f".mem_{path.name}.{uuid.uuid4().hex}.tmp")
        write_error: OSError | None = None
        replaced = False
        try:
            async with aiofiles.open(temporary, "w", encoding="utf-8") as handle:
                await handle.write(content)
                await handle.flush()
                await aiofiles.os.wrap(os.fsync)(handle.fileno())
            await aiofiles.os.wrap(os.chmod)(temporary, 0o600)
            try:
                await aiofiles.os.replace(temporary, write_path)
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EBUSY}:
                    raise
                async with (
                    aiofiles.open(temporary, "rb") as source,
                    aiofiles.open(write_path, "wb") as destination,
                ):
                    while chunk := await source.read(1024 * 1024):
                        await destination.write(chunk)
                    await destination.flush()
                    await aiofiles.os.wrap(os.fsync)(destination.fileno())
                await aiofiles.os.wrap(os.chmod)(write_path, 0o600)
                await aiofiles.os.remove(temporary)
            replaced = True
        except OSError as exc:
            write_error = exc
        finally:
            if not replaced:
                try:
                    if await aiofiles.os.path.exists(temporary):
                        await aiofiles.os.remove(temporary)
                except OSError:
                    pass
        if write_error is not None:
            raise RuntimeError(
                f"Failed to write memory file {path}: {write_error}"
            ) from write_error


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """Build a recoverable error for a replace/remove call that arrived without
    ``old_text``.

    ``replace``/``remove`` are inherently targeted -- without ``old_text`` there
    is no entry to act on, so we cannot fulfil the call. But returning a bare
    "old_text is required" is a dead-end: some structured-output clients omit the
    optional ``old_text`` field (it isn't, and can't be, schema-required without
    a top-level combinator the Codex backend rejects -- see
    tests/tools/test_memory_tool_schema.py). So instead we return the current
    entry inventory plus an explicit retry instruction, letting the model reissue
    the call with ``old_text`` set to a unique substring of the entry it means.
    Mirrors the batch path's ``_batch_error`` shape. (issues #43412, #49466)
    """
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- a short unique substring of the entry "
                f"to {action}. None was provided. Reissue the {action} with old_text "
                f"set to part of one of the current_entries below."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )




async def memory_tool(
    action: str | None = None,
    target: str = "memory",
    content: str | None = None,
    old_text: str | None = None,
    new_text: str | None = None,
    operations: list[dict[str, Any]] | None = None,
    store: MemoryStore | None = None,
) -> str:
    """Native-async handler for the model-visible memory tool."""
    if store is None:
        return tool_error(
            "Memory is not available. It may be disabled in config or this environment.",
            success=False,
        )
    if content is None and new_text is not None:
        content = new_text
    if target is None:
        target = "memory"
    if target not in {"memory", "user"}:
        return tool_error(
            f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False
        )

    if operations:
        if not isinstance(operations, list):
            return tool_error(
                "operations must be a list of {action, content?, old_text?} objects.",
                success=False,
            )
        return json.dumps(
            await store.apply_batch(target, operations), ensure_ascii=False
        )

    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
    if action == "replace" and (not old_text or not content):
        if not old_text:
            return _missing_old_text_error(store, target, "replace")
        return tool_error("content is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    if action == "add":
        result = await store.add(target, content)
    elif action == "replace":
        result = await store.replace(target, old_text, content)
    elif action == "remove":
        result = await store.remove(target, old_text)
    else:
        return tool_error(
            f"Unknown action '{action}'. Use: add, replace, remove", success=False
        )
    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory is "
        "injected into every future turn, so keep entries compact and high-signal.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that "
        "removes or shortens enough stale entries and adds the new one together.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your "
        "notes (environment, conventions, tool quirks, lessons).\n\n"
        "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, "
        "completed-work logs, temporary TODO state (use session_search for those). Reusable "
        "procedures belong in a skill, not memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform (single-op shape). Omit when using 'operations'."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape)."
            },
            "new_text": {
                "type": "string",
                "description": "Alias for 'content' (single-op shape and batch operations)."
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace. Alias: 'new_text'."},
                        "new_text": {"type": "string", "description": "Alias for 'content' in a batch operation."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error


async def _handle_memory(args: dict, **kwargs) -> str:
    """Adapt the registry's JSON-object contract to ``memory_tool``."""
    return await memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        new_text=args.get("new_text"),
        operations=args.get("operations"),
        store=kwargs.get("store"),
    )


registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=_handle_memory,
    check_fn=check_memory_requirements,
    emoji="🧠",
)
