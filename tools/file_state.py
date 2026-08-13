"""Cross-agent file state coordination.

Prevents mangled edits when concurrent subagents (same process, same
filesystem) touch the same file. Complements the single-agent path-overlap
check in ``run_agent._should_parallelize_tool_batch`` — this module catches
the case where subagent B writes a file that subagent A already read, so
A's next write would overwrite B's changes with stale content.

Design
------
A process-wide singleton ``FileStateRegistry`` tracks, per resolved path:

  * per-profile, per-agent read stamps: {task_id: {path: (mtime, read_ts, partial)}}
  * last writer within a profile: {path: (task_id, write_ts)}
  * per-event-loop path locks for read→modify→write critical sections

Three public hooks are used by the file tools:

  * ``record_read(task_id, path, *, partial)`` — called by read_file
  * ``note_write(task_id, path)`` — called after write_file / patch
  * ``check_stale(task_id, path)`` — called BEFORE write_file / patch

Plus ``lock_path(path)`` — an async context manager providing a per-path lock
around the whole read→modify→write block. And ``writes_since(task_id,
since_ts, paths)`` for the subagent-completion reminder in delegate_tool.

All methods are no-ops when ``HERMES_DISABLE_FILE_STATE_GUARD=1`` is set.

This module is intentionally separate from ``_read_tracker`` in
``file_tools.py`` — that tracker is per-task and handles consecutive-read
loop detection, which is a different concern.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from weakref import WeakKeyDictionary

import aiofiles.os

from hermes_constants import get_hermes_home


# ── Public stamp type ────────────────────────────────────────────────
# (mtime, read_ts, partial).  partial=True when read_file returned a
# windowed view (offset > 1 or limit < total_lines) — writes that happen
# after a partial read should still warn so the model re-reads in full.
ReadStamp = tuple[float, float, bool]

# Number of resolved-path entries retained per agent.  Bounded to keep
# long sessions from accumulating unbounded state.  On overflow we drop
# the oldest entries by insertion order.
_MAX_PATHS_PER_AGENT = 4096

# Global last-writer map cap.  Same policy.
_MAX_GLOBAL_WRITERS = 4096

_file_profile_context: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("file_state_profile_scope", default=None)
)
_file_profile_aliases: dict[str, str] = {}
_file_profile_aliases_lock = threading.RLock()


def _lexical_profile_identity() -> str:
    """Return the environment-only marker for the active Hermes profile."""
    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_profile_identity() -> str:
    """Return the activated canonical profile, or its lexical staging key."""
    lexical = _lexical_profile_identity()
    active = _file_profile_context.get()
    if active is not None and active[0] == lexical:
        return active[1]
    with _file_profile_aliases_lock:
        return _file_profile_aliases.get(lexical, lexical)


async def _resolve_profile_identity(lexical: str) -> str:
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    expanded = str(await expanduser(lexical))
    is_absolute = (
        expanded.startswith(("/", "\\\\"))
        or (len(expanded) >= 3 and expanded[1] == ":" and expanded[2] in "/\\")
    )
    if not is_absolute:
        expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
    realpath = aiofiles.os.wrap(os.path.realpath)
    return os.path.normcase(str(await realpath(expanded)))


@dataclass
class _FileProfileState:
    """Read/write coordination metadata owned by one canonical profile."""

    reads: dict[str, dict[str, ReadStamp]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    last_writer: dict[str, tuple[str, float]] = field(default_factory=dict)
    metadata_lock: threading.RLock = field(default_factory=threading.RLock)


class FileStateRegistry:
    """Process-wide coordinator for cross-agent file edits."""

    def __init__(self) -> None:
        self._profile_states: dict[str, _FileProfileState] = {}
        self._profile_states_lock = threading.RLock()
        self._path_locks: WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = WeakKeyDictionary()
        self._path_locks_lock = threading.RLock()

    def _profile_state(self) -> _FileProfileState:
        profile = _current_profile_identity()
        with self._profile_states_lock:
            return self._profile_states.setdefault(profile, _FileProfileState())

    async def _activate_profile_state(self) -> _FileProfileState:
        lexical = _lexical_profile_identity()
        active = _file_profile_context.get()
        if active is not None and active[0] == lexical:
            canonical = active[1]
        else:
            canonical = await _resolve_profile_identity(lexical)
            with _file_profile_aliases_lock:
                _file_profile_aliases[lexical] = canonical
            _file_profile_context.set((lexical, canonical))
        with self._profile_states_lock:
            state = self._profile_states.setdefault(canonical, _FileProfileState())
            if lexical != canonical:
                staged = self._profile_states.pop(lexical, None)
                if staged is not None and staged is not state:
                    with staged.metadata_lock, state.metadata_lock:
                        for task_id, paths in staged.reads.items():
                            current = state.reads[task_id]
                            for path, stamp in paths.items():
                                prior = current.get(path)
                                if prior is None or stamp[1] > prior[1]:
                                    current[path] = stamp
                        for path, writer in staged.last_writer.items():
                            prior = state.last_writer.get(path)
                            if prior is None or writer[1] > prior[1]:
                                state.last_writer[path] = writer
            return state

    @property
    def _reads(self) -> dict[str, dict[str, ReadStamp]]:
        """Private dict-compatible view for the active profile."""
        return self._profile_state().reads

    @property
    def _last_writer(self) -> dict[str, tuple[str, float]]:
        """Private dict-compatible view for the active profile."""
        return self._profile_state().last_writer

    # ── Path lock management ────────────────────────────────────────
    def _lock_for(self, resolved: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._path_locks_lock:
            for known_loop in tuple(self._path_locks):
                if known_loop.is_closed():
                    self._path_locks.pop(known_loop, None)
            locks = self._path_locks.setdefault(loop, {})
            return locks.setdefault(resolved, asyncio.Lock())

    @asynccontextmanager
    async def lock_path(self, resolved: str):
        """Serialize a read→modify→write section for one resolved path."""
        async with self._lock_for(resolved):
            yield

    # ── Read/write accounting ───────────────────────────────────────
    async def record_read(
        self,
        task_id: str,
        resolved: str,
        *,
        partial: bool = False,
        mtime: float | None = None,
    ) -> None:
        if _disabled():
            return
        state = await self._activate_profile_state()
        if mtime is None:
            try:
                mtime = (await aiofiles.os.stat(resolved)).st_mtime
            except OSError:
                return
        now = time.time()
        with state.metadata_lock:
            agent_reads = state.reads[task_id]
            agent_reads[resolved] = (float(mtime), now, bool(partial))
            _cap_dict(agent_reads, _MAX_PATHS_PER_AGENT)

    async def note_write(
        self,
        task_id: str,
        resolved: str,
        *,
        mtime: float | None = None,
    ) -> None:
        """Record a successful write.

        Updates the global last-writer map AND this agent's own read stamp
        (a write is an implicit read — the agent now knows the current
        content).
        """
        if _disabled():
            return
        state = await self._activate_profile_state()
        if mtime is None:
            try:
                mtime = (await aiofiles.os.stat(resolved)).st_mtime
            except OSError:
                return
        now = time.time()
        with state.metadata_lock:
            state.last_writer[resolved] = (task_id, now)
            _cap_dict(state.last_writer, _MAX_GLOBAL_WRITERS)
            # Writer's own view is now up-to-date.
            state.reads[task_id][resolved] = (float(mtime), now, False)
            _cap_dict(state.reads[task_id], _MAX_PATHS_PER_AGENT)

    async def check_stale(self, task_id: str, resolved: str) -> str | None:
        """Return a model-facing warning if this write would be stale.

        Three staleness classes, in order of severity:

          1. Sibling subagent wrote this file after this agent's last read.
          2. External/unknown change (mtime differs from our last read).
          3. Agent never read the file (write-without-read).

        Returns ``None`` when the write is safe.  Does not raise — callers
        decide whether to block or warn.
        """
        if _disabled():
            return None
        state = await self._activate_profile_state()
        with state.metadata_lock:
            stamp = state.reads.get(task_id, {}).get(resolved)
            last_writer = state.last_writer.get(resolved)

        # Case 3: never read AND we have no write record — net-new file or
        # first touch by this agent.  Let existing _check_sensitive_path
        # and file-exists logic handle it; nothing to warn about here.
        if stamp is None and last_writer is None:
            return None

        try:
            current_mtime = (await aiofiles.os.stat(resolved)).st_mtime
        except OSError:
            # File doesn't exist — write will create it; not stale.
            return None

        # Case 1: sibling subagent modified after our last read.
        if last_writer is not None:
            writer_tid, writer_ts = last_writer
            if writer_tid != task_id:
                if stamp is None:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} but this agent never read it. "
                        "Read the file before writing to avoid overwriting "
                        "the sibling's changes."
                    )
                read_ts = stamp[1]
                if writer_ts > read_ts:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} at {_fmt_ts(writer_ts)} — after "
                        f"this agent's last read at {_fmt_ts(read_ts)}. "
                        "Re-read the file before writing."
                    )

        # Case 2: external / unknown modification (mtime drifted).
        if stamp is not None:
            read_mtime, _read_ts, partial = stamp
            if current_mtime != read_mtime:
                return (
                    f"{resolved} was modified since you last read it "
                    "on disk (external edit or unrecorded writer). "
                    "Re-read the file before writing."
                )
            if partial:
                return (
                    f"{resolved} was last read with offset/limit pagination "
                    "(partial view). Re-read the whole file before "
                    "overwriting it."
                )

        # Case 3b: agent truly never read the file.
        if stamp is None:
            return (
                f"{resolved} was not read by this agent. "
                "Read the file first so you can write an informed edit."
            )

        return None

    # ── Reminder helper for delegate_tool ───────────────────────────
    def writes_since(
        self,
        exclude_task_id: str,
        since_ts: float,
        paths: Iterable[str],
    ) -> dict[str, list[str]]:
        """Return ``{writer_task_id: [paths]}`` for writes done after
        ``since_ts`` by agents OTHER than ``exclude_task_id``.

        Used by delegate_task to append a "subagent modified files the
        parent previously read" reminder to the delegation result.
        """
        if _disabled():
            return {}
        state = self._profile_state()
        paths_set = set(paths)
        out: dict[str, list[str]] = defaultdict(list)
        with state.metadata_lock:
            for p, (writer_tid, ts) in state.last_writer.items():
                if writer_tid == exclude_task_id:
                    continue
                if ts < since_ts:
                    continue
                if p in paths_set:
                    out[writer_tid].append(p)
        return dict(out)

    def known_reads(self, task_id: str) -> list[str]:
        """Return the list of resolved paths this agent has read."""
        if _disabled():
            return []
        state = self._profile_state()
        with state.metadata_lock:
            return list(state.reads.get(task_id, {}).keys())

    # ── Testing hooks ───────────────────────────────────────────────
    def clear(self) -> None:
        """Reset all state.  Intended for tests only."""
        with self._profile_states_lock:
            self._profile_states.clear()
        with self._path_locks_lock:
            self._path_locks.clear()

    def _clear_task(self, task_id: str) -> None:
        """Drop read stamps owned by a task whose environment was closed."""
        state = self._profile_state()
        with state.metadata_lock:
            state.reads.pop(task_id, None)


# ── Module-level singleton + helpers ─────────────────────────────────
_registry = FileStateRegistry()


def get_registry() -> FileStateRegistry:
    return _registry


async def _activate_profile_scope() -> str:
    """Activate canonical file metadata ownership for the current profile."""
    await _registry._activate_profile_state()
    return _current_profile_identity()


def _disabled() -> bool:
    # Re-read each call so tests can toggle via monkeypatch.setenv.
    return os.environ.get("HERMES_DISABLE_FILE_STATE_GUARD", "").strip() == "1"


def _fmt_ts(ts: float) -> str:
    # Short relative wall-clock for error messages; avoids pulling in
    # datetime formatting overhead on the hot path.
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _cap_dict(d: dict, limit: int) -> None:
    """Trim a dict to ``limit`` entries by dropping insertion-order oldest."""
    over = len(d) - limit
    if over <= 0:
        return
    # dict preserves insertion order (PY>=3.7) — pop the oldest keys.
    it = iter(d)
    for _ in range(over):
        try:
            d.pop(next(it))
        except (StopIteration, KeyError):
            break


# ── Convenience wrappers (short names used at call sites) ────────────
async def record_read(
    task_id: str,
    resolved_or_path: str | Path,
    *,
    partial: bool = False,
) -> None:
    await _registry.record_read(
        task_id,
        str(resolved_or_path),
        partial=partial,
    )


async def note_write(
    task_id: str,
    resolved_or_path: str | Path,
) -> None:
    await _registry.note_write(task_id, str(resolved_or_path))


async def check_stale(task_id: str, resolved_or_path: str | Path) -> str | None:
    return await _registry.check_stale(task_id, str(resolved_or_path))


def lock_path(resolved_or_path: str | Path):
    return _registry.lock_path(str(resolved_or_path))


def writes_since(
    exclude_task_id: str,
    since_ts: float,
    paths: Iterable[str | Path],
) -> dict[str, list[str]]:
    return _registry.writes_since(exclude_task_id, since_ts, [str(p) for p in paths])


def known_reads(task_id: str) -> list[str]:
    return _registry.known_reads(task_id)


__all__ = [
    "FileStateRegistry",
    "get_registry",
    "record_read",
    "note_write",
    "check_stale",
    "lock_path",
    "writes_since",
    "known_reads",
]
