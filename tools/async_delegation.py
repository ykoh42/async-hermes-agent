#!/usr/bin/env python3
"""Durable native-async lifecycle for background subagent delegation."""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import inspect
import json
import logging
import os
import sys
import threading
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar
from collections.abc import Awaitable, Callable, Coroutine

import aiofiles
import aiofiles.os
import aiosqlite

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)

logger = logging.getLogger(__name__)

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

_DEFAULT_MAX_ASYNC_CHILDREN = 3
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
_MAX_DELIVERY_ATTEMPTS = 8

_STALE_CHECK_INTERVAL = 30.0
_STALE_IDLE_SECONDS = 450.0
_STALE_IN_TOOL_SECONDS = 1200.0
_STALL_GRACE_SECONDS = 120.0
_OWNER_LEASE_SECONDS = 180.0

@dataclass
class _DelegationScopeState:
    """Runtime state owned by one event loop and canonical Hermes home."""

    profile_home: str
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    monitor_task: asyncio.Task[None] | None = None
    restored_queues: set[tuple[int, str]] = field(default_factory=set)
    db_lock: asyncio.Lock | None = None
    db_lock_users: int = 0
    restore_lock: asyncio.Lock | None = None
    restore_lock_users: int = 0
    # A caller-owned PostgreSQL SessionDB may be bound by an AIAgent's
    # awaited initialization.  SQLite remains the default when this is None.
    session_db_ref: weakref.ReferenceType[Any] | None = None


_scope_states: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _DelegationScopeState]
] = weakref.WeakKeyDictionary()
_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()
_scope_guard = threading.RLock()
_active_scope: contextvars.ContextVar[
    tuple[str, _DelegationScopeState] | None
] = contextvars.ContextVar("async_delegation_profile_scope", default=None)
_SESSION_DB_UNSET = object()
_active_session_db: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "async_delegation_session_db", default=_SESSION_DB_UNSET
)
_legacy_scope = _DelegationScopeState("")

# Private compatibility projections for older tests and integrations. Runtime
# code resolves its scope explicitly; these names only reflect the last scope
# activated in the current process and must not be used for ownership decisions.
_records: dict[str, dict[str, Any]] = _legacy_scope.records
_tasks: dict[str, asyncio.Task[None]] = _legacy_scope.tasks
_monitor_task: asyncio.Task[None] | None = None
_restored_queues: set[tuple[int, str]] = _legacy_scope.restored_queues
_db_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
_restore_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
_process_start_times: dict[int, int | None] = {}
_owner_instance_pid: int | None = None
_owner_instance_token: str | None = None
_DeliveryResult = TypeVar("_DeliveryResult")


def _postgres_session_db(state: _DelegationScopeState | None = None) -> Any:
    """Return the explicitly bound PostgreSQL store, never a guessed fallback."""
    selected_override = _active_session_db.get()
    selected = state or _current_scope_state()
    if selected_override is not _SESSION_DB_UNSET:
        if selected_override is None:
            return None
        reference = selected.session_db_ref
        if reference is None:
            # A task-local binding inherited from another profile must never
            # route this profile's delegation to that database.
            return None
        database = reference()
        if database is None:
            raise RuntimeError(
                "The PostgreSQL SessionDB bound to async delegation is no longer alive"
            )
        if database is not selected_override:
            raise RuntimeError(
                "The PostgreSQL SessionDB binding does not match this delegation profile"
            )
        return database
    reference = selected.session_db_ref
    if reference is None:
        return None
    database = reference()
    if database is None:
        raise RuntimeError(
            "The PostgreSQL SessionDB bound to async delegation is no longer alive"
        )
    return database


async def _bind_session_db(session_db: Any) -> None:
    """Bind an injected PostgreSQL store to the current delegation scope."""
    state = await _activate_scope_state()
    if session_db is None:
        # Explicitly selecting the default SQLite path in this context must
        # not inherit a PostgreSQL binding left by another agent in the same
        # profile/event loop.
        _active_session_db.set(None)
        return
    # SQLite SessionDB deliberately has no _read/_write Core boundaries.  This
    # private capability keeps the public backend surface and the upstream
    # delegation functions unchanged without importing the optional PG extra.
    if not callable(getattr(session_db, "_read", None)) or not callable(
        getattr(session_db, "_write", None)
    ):
        # An explicitly supplied SQLite store must override any PostgreSQL
        # binding inherited by the caller's context.  The scope-level weak
        # reference remains available to sibling PG agents, but this task is
        # fail-closed to the default SQLite path.
        _active_session_db.set(None)
        return
    try:
        reference = weakref.ref(session_db)
    except TypeError as exc:
        raise TypeError(
            "PostgreSQL SessionDB used for async delegation must support weak references"
        ) from exc
    current = state.session_db_ref() if state.session_db_ref is not None else None
    if current is not None and current is not session_db:
        raise RuntimeError(
            "One PostgreSQL SessionDB must be shared by a delegation profile and event loop"
        )
    state.session_db_ref = reference
    _active_session_db.set(session_db)


def _owner_instance_id() -> str:
    """Return a process-local owner token and refresh it after fork."""
    global _owner_instance_pid, _owner_instance_token
    pid = os.getpid()
    if _owner_instance_pid != pid or _owner_instance_token is None:
        _owner_instance_pid = pid
        _owner_instance_token = f"{pid}:{uuid.uuid4().hex}"
    return _owner_instance_token


def _db_path():
    return get_hermes_home() / "state.db"


def _lexical_profile_identity() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _project_compatibility_globals(state: _DelegationScopeState) -> None:
    global _records, _tasks, _monitor_task, _restored_queues
    _records = state.records
    _tasks = state.tasks
    _monitor_task = state.monitor_task
    _restored_queues = state.restored_queues


def _prune_closed_loops() -> None:
    with _scope_guard:
        for loop in tuple(_scope_states):
            if loop.is_closed():
                _scope_states.pop(loop, None)
        for loop in tuple(_scope_aliases):
            if loop.is_closed():
                _scope_aliases.pop(loop, None)


def _current_scope_state() -> _DelegationScopeState:
    """Resolve an already activated scope without filesystem access."""
    lexical = _lexical_profile_identity()
    active = _active_scope.get()
    if active is not None and active[0] == lexical:
        return active[1]
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _legacy_scope
    with _scope_guard:
        aliases = _scope_aliases.get(loop)
        canonical = aliases.get(lexical, lexical) if aliases is not None else lexical
        state = _scope_states.get(loop, {}).get(canonical)
    return state or _legacy_scope


async def _canonical_path_identity(path: str | os.PathLike[str]) -> str:
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    expanded = str(await expanduser(os.fspath(path)))
    is_absolute = (
        expanded.startswith(("/", "\\\\"))
        or (len(expanded) >= 3 and expanded[1] == ":" and expanded[2] in "/\\")
    )
    if not is_absolute:
        expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
    realpath = aiofiles.os.wrap(os.path.realpath)
    return os.path.normcase(str(await realpath(expanded)))


async def _activate_scope_state() -> _DelegationScopeState:
    """Activate the current loop's canonical profile without blocking it."""
    _prune_closed_loops()
    lexical = _lexical_profile_identity()
    active = _active_scope.get()
    if active is not None and active[0] == lexical:
        _project_compatibility_globals(active[1])
        return active[1]
    canonical = await _canonical_path_identity(lexical)
    loop = asyncio.get_running_loop()
    with _scope_guard:
        _scope_aliases.setdefault(loop, {})[lexical] = canonical
        profiles = _scope_states.setdefault(loop, {})
        state = profiles.setdefault(canonical, _DelegationScopeState(canonical))
    _active_scope.set((lexical, state))
    _project_compatibility_globals(state)
    return state


@asynccontextmanager
async def _scope_lock(state: _DelegationScopeState, name: str):
    lock_name = f"{name}_lock"
    users_name = f"{name}_lock_users"
    lock = getattr(state, lock_name)
    if lock is None:
        lock = asyncio.Lock()
        setattr(state, lock_name, lock)
    setattr(state, users_name, getattr(state, users_name) + 1)
    try:
        async with lock:
            yield
    finally:
        remaining = getattr(state, users_name) - 1
        setattr(state, users_name, remaining)
        if remaining == 0:
            setattr(state, lock_name, None)


@asynccontextmanager
async def _db_lock():
    state = await _activate_scope_state()
    async with _scope_lock(state, "db"):
        yield


@asynccontextmanager
async def _restore_lock():
    state = await _activate_scope_state()
    async with _scope_lock(state, "restore"):
        yield


async def _connect() -> aiosqlite.Connection:
    path = _db_path()
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    conn = await aiosqlite.connect(path, timeout=10)
    try:
        await _initialize_schema(conn)
    except BaseException:
        await conn.close()
        raise
    return conn


async def _initialize_schema(conn: aiosqlite.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    await apply_wal_with_fallback(conn, db_label="state.db (async_delegation)")
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            task_json TEXT,
            delivery_claim TEXT,
            delivery_claimed_at REAL,
            origin_session_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    cursor = await conn.execute("PRAGMA table_info(async_delegations)")
    columns = {row[1] for row in await cursor.fetchall()}
    for name, sql_type in (
        ("owner_pid", "INTEGER"),
        ("owner_started_at", "INTEGER"),
        ("task_json", "TEXT"),
        ("delivery_claim", "TEXT"),
        ("delivery_claimed_at", "REAL"),
        ("origin_session_id", "TEXT"),
    ):
        if name not in columns:
            await conn.execute(
                f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}"
            )


@asynccontextmanager
async def _transaction():
    """Commit or roll back and deterministically close every connection."""
    conn = await _connect()
    try:
        yield conn
    except BaseException:
        await conn.rollback()
        raise
    else:
        await conn.commit()
    finally:
        await conn.close()


async def _safe_process_start_time(pid: int) -> int | None:
    if pid in _process_start_times:
        return _process_start_times[pid]
    started_at: int | None = None
    if sys.platform.startswith("linux"):
        try:
            async with aiofiles.open(f"/proc/{pid}/stat", "rb") as handle:
                stat = await handle.read()
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            pass
        else:
            _prefix, separator, fields = stat.rpartition(b") ")
            if separator:
                try:
                    started_at = int(fields.split()[19])
                except (IndexError, TypeError, ValueError):
                    pass
    elif os.name == "posix":
        process = await asyncio.create_subprocess_exec(
            "ps",
            "-o",
            "lstart=",
            "-p",
            str(pid),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        communicate = asyncio.create_task(process.communicate())
        cancellation: asyncio.CancelledError | None = None
        try:
            stdout, _stderr = await asyncio.shield(communicate)
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - reap then re-raise
            cancellation = exc
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            while True:
                try:
                    stdout, _stderr = await asyncio.shield(communicate)
                    break
                except asyncio.CancelledError:  # noqa: ASYNC103 - finish reap
                    if communicate.cancelled():
                        raise
        if cancellation is not None:
            raise cancellation
        if process.returncode == 0:
            raw = stdout.decode(errors="replace").strip()
            try:
                started_at = int(
                    dt.datetime.strptime(raw, "%a %b %d %H:%M:%S %Y")
                    .astimezone()
                    .timestamp()
                )
            except (UnicodeError, ValueError, OverflowError):
                pass
    elif os.name == "nt":
        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if handle:
            try:
                created = wintypes.FILETIME()
                exited = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                if kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    started_at = (created.dwHighDateTime << 32) | created.dwLowDateTime
            finally:
                kernel32.CloseHandle(handle)
    _process_start_times[pid] = started_at
    return started_at


async def _persist_dispatch(record: dict[str, Any]) -> None:
    now = time.time()
    owner_started_at = await _safe_process_start_time(os.getpid())
    state = _current_scope_state()
    postgres = _postgres_session_db(state)
    task_payload = {
        key: record.get(key)
        for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch")
        if key in record
    }
    if postgres is not None:
        from sqlalchemy import text

        owner = record.get("_owner_instance_id") or _owner_instance_id()

        async def _write(connection):
            await connection.execute(
                text(
                    """INSERT INTO async_delegations
                       (delegation_id, origin_session, origin_ui_session_id,
                        parent_session_id, state, dispatched_at, updated_at,
                        delivery_state, delivery_attempts, owner_pid,
                        owner_started_at, task_json, origin_session_id,
                        owner_instance_id, lease_expires_at)
                       VALUES (:delegation_id, :origin_session, :origin_ui,
                               :parent_session, 'running', :dispatched_at,
                               :updated_at, 'pending', 0, :owner_pid,
                               :owner_started_at, :task_json, :origin_session_id,
                               :owner_instance_id,
                               EXTRACT(EPOCH FROM clock_timestamp()) + :lease)
                       ON CONFLICT (delegation_id) DO UPDATE SET
                           origin_session = excluded.origin_session,
                           origin_ui_session_id = excluded.origin_ui_session_id,
                           parent_session_id = excluded.parent_session_id,
                           state = excluded.state,
                           dispatched_at = excluded.dispatched_at,
                           completed_at = NULL,
                           updated_at = excluded.updated_at,
                           event_json = NULL,
                           result_json = NULL,
                           delivery_state = 'pending',
                           delivery_attempts = 0,
                           delivered_at = NULL,
                           owner_pid = excluded.owner_pid,
                           owner_started_at = excluded.owner_started_at,
                           task_json = excluded.task_json,
                           delivery_claim = NULL,
                           delivery_claimed_at = NULL,
                           origin_session_id = excluded.origin_session_id,
                           owner_instance_id = excluded.owner_instance_id,
                           lease_expires_at = excluded.lease_expires_at"""
                ),
                {
                    "delegation_id": record["delegation_id"],
                    "origin_session": record.get("session_key", ""),
                    "origin_ui": record.get("origin_ui_session_id", ""),
                    "parent_session": record.get("parent_session_id"),
                    "dispatched_at": record["dispatched_at"],
                    "updated_at": now,
                    "owner_pid": os.getpid(),
                    "owner_started_at": owner_started_at,
                    "task_json": json.dumps(task_payload),
                    "origin_session_id": record.get("origin_session_id", ""),
                    "owner_instance_id": owner,
                    "lease": _OWNER_LEASE_SECONDS,
                },
            )

        await postgres._write(_write)
        await _prune_durable_records()
        return
    async with _db_lock():
        async with _transaction() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id,
                    parent_session_id, state, dispatched_at, updated_at,
                    delivery_state, delivery_attempts, owner_pid,
                    owner_started_at, task_json, origin_session_id)
                   VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
                (
                    record["delegation_id"],
                    record.get("session_key", ""),
                    record.get("origin_ui_session_id", ""),
                    record.get("parent_session_id"),
                    record["dispatched_at"],
                    now,
                    os.getpid(),
                    owner_started_at,
                    json.dumps(task_payload),
                    record.get("origin_session_id", ""),
                ),
            )
    await _prune_durable_records()


async def _delete_durable_delegation(delegation_id: str) -> None:
    postgres = _postgres_session_db()
    if postgres is not None:
        if bool(getattr(postgres, "_read_only", False)):
            return
        from sqlalchemy import text

        async def _write(connection):
            await connection.execute(
                text("DELETE FROM async_delegations WHERE delegation_id = :id"),
                {"id": delegation_id},
            )

        await postgres._write(_write)
        return
    async with _db_lock():
        async with _transaction() as conn:
            await conn.execute(
                "DELETE FROM async_delegations WHERE delegation_id=?",
                (delegation_id,),
            )


async def _prune_durable_records() -> None:
    cutoff = time.time() - _DURABLE_RETENTION_SECONDS
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            await connection.execute(
                text(
                    "DELETE FROM async_delegations "
                    "WHERE delivery_state = 'delivered' AND updated_at < :cutoff"
                ),
                {"cutoff": cutoff},
            )
            count = await connection.execute(
                text(
                    "SELECT COUNT(*) FROM async_delegations "
                    "WHERE state NOT IN ('running', 'finalizing')"
                )
            )
            excess = max(0, int(count.scalar_one()) - _MAX_RETAINED_COMPLETED)
            if excess:
                await connection.execute(
                    text(
                        "DELETE FROM async_delegations WHERE delegation_id IN ("
                        "SELECT delegation_id FROM async_delegations "
                        "WHERE state NOT IN ('running', 'finalizing') "
                        "ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END, "
                        "updated_at ASC LIMIT :limit)"
                    ),
                    {"limit": excess},
                )
            pending = await connection.execute(
                text(
                    "SELECT COUNT(*) FROM async_delegations "
                    "WHERE state NOT IN ('running', 'finalizing') "
                    "AND delivery_state = 'pending'"
                )
            )
            overflow = max(0, int(pending.scalar_one()) - _MAX_DURABLE_PENDING)
            if overflow:
                await connection.execute(
                    text(
                        "DELETE FROM async_delegations WHERE delegation_id IN ("
                        "SELECT delegation_id FROM async_delegations "
                        "WHERE state NOT IN ('running', 'finalizing') "
                        "AND delivery_state = 'pending' "
                        "ORDER BY updated_at ASC LIMIT :limit)"
                    ),
                    {"limit": overflow},
                )

        await postgres._write(_write)
        return
    async with _db_lock():
        async with _transaction() as conn:
            await conn.execute(
                "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?",
                (cutoff,),
            )
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')"
            )
            row = await cursor.fetchone()
            excess = max(0, int(row[0]) - _MAX_RETAINED_COMPLETED)
            if excess:
                await conn.execute(
                    """DELETE FROM async_delegations WHERE delegation_id IN (
                         SELECT delegation_id FROM async_delegations
                         WHERE state NOT IN ('running','finalizing')
                         ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                                  updated_at ASC LIMIT ?
                       )""",
                    (excess,),
                )
            cursor = await conn.execute(
                """SELECT COUNT(*) FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'"""
            )
            row = await cursor.fetchone()
            overflow = max(0, int(row[0]) - _MAX_DURABLE_PENDING)
            if overflow:
                await conn.execute(
                    """DELETE FROM async_delegations WHERE delegation_id IN (
                         SELECT delegation_id FROM async_delegations
                         WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                         ORDER BY updated_at ASC LIMIT ?
                       )""",
                    (overflow,),
                )


async def _persist_completion(
    event: dict[str, Any], result: dict[str, Any]
) -> bool | None:
    now = time.time()
    state = _current_scope_state()
    postgres = _postgres_session_db(state)
    if postgres is not None:
        from sqlalchemy import text

        owner = state.records.get(event.get("delegation_id"), {}).get(
            "_owner_instance_id"
        ) or _owner_instance_id()

        async def _write(connection):
            result_cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET state = :state,
                       completed_at = :completed_at, updated_at = :updated_at,
                       event_json = :event_json, result_json = :result_json,
                       delivery_state = 'pending', lease_expires_at = NULL
                       WHERE delegation_id = :delegation_id
                         AND owner_instance_id = :owner_instance_id
                         AND state IN ('running', 'finalizing')"""
                ),
                {
                    "state": event.get("status", "completed"),
                    "completed_at": event.get("completed_at", now),
                    "updated_at": now,
                    "event_json": json.dumps(event),
                    "result_json": json.dumps(result),
                    "delegation_id": event["delegation_id"],
                    "owner_instance_id": owner,
                },
            )
            return result_cursor.rowcount == 1

        return await postgres._write(_write)
    async with _db_lock():
        async with _transaction() as conn:
            await conn.execute(
                """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
                   event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (
                    event.get("status", "completed"),
                    event.get("completed_at", now),
                    now,
                    json.dumps(event),
                    json.dumps(result),
                    event["delegation_id"],
                ),
            )
    return None


async def _note_delivery_attempt(delegation_id: str) -> None:
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            await connection.execute(
                text(
                    "UPDATE async_delegations SET delivery_attempts = "
                    "delivery_attempts + 1, updated_at = :updated_at "
                    "WHERE delegation_id = :delegation_id"
                ),
                {"updated_at": time.time(), "delegation_id": delegation_id},
            )

        await postgres._write(_write)
        return
    async with _db_lock():
        async with _transaction() as conn:
            await conn.execute(
                """UPDATE async_delegations SET delivery_attempts=delivery_attempts+1,
                   updated_at=? WHERE delegation_id=?""",
                (time.time(), delegation_id),
            )


async def recover_abandoned_delegations() -> int:
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        now = time.time()

        async def _write(connection):
            cursor = await connection.execute(
                text(
                    """SELECT delegation_id, origin_session,
                              origin_ui_session_id, parent_session_id,
                              dispatched_at, task_json, origin_session_id
                       FROM async_delegations
                       WHERE state IN ('running', 'finalizing')
                         AND (lease_expires_at IS NULL OR
                              lease_expires_at < EXTRACT(EPOCH FROM clock_timestamp()))
                       FOR UPDATE SKIP LOCKED"""
                )
            )
            recovered = 0
            for row in cursor:
                (
                    delegation_id,
                    session_key,
                    origin_ui,
                    parent_id,
                    dispatched_at,
                    task_json,
                    origin_session_id,
                ) = row
                task = json.loads(task_json or "{}")
                error = (
                    "Delegation owner exited before recording a terminal result; "
                    "outcome unknown."
                )
                event = {
                    "type": "async_delegation",
                    "delegation_id": delegation_id,
                    "session_key": session_key,
                    "origin_ui_session_id": origin_ui,
                    "origin_session_id": origin_session_id or "",
                    "parent_session_id": parent_id,
                    "goal": task.get("goal", ""),
                    "goals": task.get("goals"),
                    "context": task.get("context"),
                    "toolsets": task.get("toolsets"),
                    "role": task.get("role"),
                    "model": task.get("model"),
                    "is_batch": bool(task.get("is_batch")),
                    "status": "unknown",
                    "summary": None,
                    "error": error,
                    "dispatched_at": dispatched_at,
                    "completed_at": now,
                }
                result = {"status": "unknown", "summary": None, "error": error}
                await connection.execute(
                    text(
                        """UPDATE async_delegations SET state = 'unknown',
                           completed_at = :completed_at, updated_at = :updated_at,
                           event_json = :event_json, result_json = :result_json,
                           delivery_state = 'pending', lease_expires_at = NULL
                           WHERE delegation_id = :delegation_id
                             AND state IN ('running', 'finalizing')"""
                    ),
                    {
                        "completed_at": now,
                        "updated_at": now,
                        "event_json": json.dumps(event),
                        "result_json": json.dumps(result),
                        "delegation_id": delegation_id,
                    },
                )
                recovered += 1
            return recovered

        return await postgres._write(_write)

    from gateway.status import _pid_exists

    now = time.time()
    recovered = 0
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                """SELECT delegation_id, origin_session, origin_ui_session_id,
                          parent_session_id, dispatched_at, owner_pid,
                          owner_started_at, task_json, origin_session_id
                   FROM async_delegations WHERE state IN ('running','finalizing')"""
            )
            rows = await cursor.fetchall()
            for row in rows:
                (
                    delegation_id,
                    session_key,
                    origin_ui,
                    parent_id,
                    dispatched_at,
                    pid,
                    started,
                    task_json,
                    origin_session_id,
                ) = row
                live = bool(pid) and await _pid_exists(int(pid))
                if live and started is not None:
                    live = await _safe_process_start_time(int(pid)) == int(started)
                if live:
                    continue
                task = json.loads(task_json or "{}")
                error = (
                    "Delegation owner exited before recording a terminal result; "
                    "outcome unknown."
                )
                event = {
                    "type": "async_delegation",
                    "delegation_id": delegation_id,
                    "session_key": session_key,
                    "origin_ui_session_id": origin_ui,
                    "origin_session_id": origin_session_id or "",
                    "parent_session_id": parent_id,
                    "goal": task.get("goal", ""),
                    "goals": task.get("goals"),
                    "context": task.get("context"),
                    "toolsets": task.get("toolsets"),
                    "role": task.get("role"),
                    "model": task.get("model"),
                    "is_batch": bool(task.get("is_batch")),
                    "status": "unknown",
                    "summary": None,
                    "error": error,
                    "dispatched_at": dispatched_at,
                    "completed_at": now,
                }
                result = {"status": "unknown", "summary": None, "error": error}
                await conn.execute(
                    """UPDATE async_delegations SET state='unknown', completed_at=?,
                       updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                       WHERE delegation_id=?""",
                    (now, now, json.dumps(event), json.dumps(result), delegation_id),
                )
                recovered += 1
    return recovered


async def restore_undelivered_completions(target_queue) -> int:
    """Restore durable pending events exactly once for a queue instance."""
    state = await _activate_scope_state()
    postgres = _postgres_session_db(state)
    if postgres is not None:
        if bool(getattr(postgres, "_read_only", False)):
            raise PermissionError(
                "Writable PostgreSQL SessionDB is required for delegation delivery"
            )
        queue_key = (id(target_queue), f"postgres:{id(postgres)}")
        async with _restore_lock():
            if queue_key in state.restored_queues:
                return 0
            await recover_abandoned_delegations()
            from sqlalchemy import text

            async def _read(connection):
                cursor = await connection.execute(
                    text(
                        """SELECT delegation_id, event_json
                           FROM async_delegations
                           WHERE state != 'running'
                             AND delivery_state = 'pending'
                             AND event_json IS NOT NULL
                           ORDER BY completed_at, delegation_id"""
                    )
                )
                return cursor.fetchall()

            rows = await postgres._read(_read)
            restored: list[dict[str, Any]] = []
            for _delegation_id, payload in rows:
                event = json.loads(payload)
                if isinstance(event, dict):
                    event["restored"] = True
                    restored.append(event)
            for event in restored:
                target_queue.put_nowait(event)
            state.restored_queues.add(queue_key)
            return len(restored)
    queue_key = (id(target_queue), await _canonical_path_identity(_db_path()))
    async with _restore_lock():
        if queue_key in state.restored_queues:
            return 0
        await recover_abandoned_delegations()
        restored: list[dict[str, Any]] = []
        async with _db_lock():
            async with _transaction() as conn:
                cursor = await conn.execute(
                    """SELECT delegation_id, event_json FROM async_delegations
                       WHERE state != 'running' AND delivery_state='pending'
                         AND event_json IS NOT NULL
                       ORDER BY completed_at, delegation_id"""
                )
                for _delegation_id, payload in await cursor.fetchall():
                    event = json.loads(payload)
                    if isinstance(event, dict):
                        event["restored"] = True
                        restored.append(event)
        for event in restored:
            target_queue.put_nowait(event)
        state.restored_queues.add(queue_key)
        return len(restored)


async def _finish_delivery_transition(
    operation: Coroutine[Any, Any, _DeliveryResult],
) -> tuple[_DeliveryResult, asyncio.CancelledError | None]:
    """Finish one SQLite delivery transition before propagating cancellation."""
    transition = asyncio.create_task(operation)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(transition)
            return result, cancellation
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - returned below
            if transition.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise


async def _propagate_after_delivery_transition(
    operation: Coroutine[Any, Any, _DeliveryResult],
) -> _DeliveryResult:
    result, cancellation = await _finish_delivery_transition(operation)
    if cancellation is not None:
        raise cancellation
    return result


async def _mark_completion_delivered(delegation_id: str) -> bool:
    now = time.time()
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET delivery_state = 'delivered',
                       delivered_at = :delivered_at, updated_at = :updated_at
                       WHERE delegation_id = :delegation_id
                         AND delivery_state != 'delivered'"""
                ),
                {
                    "delivered_at": now,
                    "updated_at": now,
                    "delegation_id": delegation_id,
                },
            )
            return cursor.rowcount == 1

        return await postgres._write(_write)
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                """UPDATE async_delegations SET delivery_state='delivered',
                   delivered_at=?, updated_at=?
                   WHERE delegation_id=? AND delivery_state!='delivered'""",
                (now, now, delegation_id),
            )
            return cursor.rowcount == 1


async def mark_completion_delivered(delegation_id: str) -> bool:
    return await _propagate_after_delivery_transition(
        _mark_completion_delivered(delegation_id)
    )


async def _claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    now = time.time()
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            exists = await connection.execute(
                text(
                    "SELECT delegation_id FROM async_delegations "
                    "WHERE delegation_id = :delegation_id"
                ),
                {"delegation_id": delegation_id},
            )
            if exists.first() is None:
                return True
            cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET delivery_claim = :claim_id,
                       delivery_claimed_at = :claimed_at,
                       delivery_attempts = delivery_attempts + 1,
                       updated_at = :updated_at
                       WHERE delegation_id = :delegation_id
                         AND delivery_state = 'pending'
                         AND (delivery_claim IS NULL OR
                              delivery_claimed_at < :expired_at)
                       RETURNING delegation_id"""
                ),
                {
                    "claim_id": claim_id,
                    "claimed_at": now,
                    "updated_at": now,
                    "delegation_id": delegation_id,
                    "expired_at": now - 300,
                },
            )
            return cursor.first() is not None

        return await postgres._write(_write)
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                "SELECT delivery_state FROM async_delegations WHERE delegation_id=?",
                (delegation_id,),
            )
            if await cursor.fetchone() is None:
                return True
            cursor = await conn.execute(
                """UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                          delivery_attempts=delivery_attempts+1, updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
                (claim_id, now, now, delegation_id, now - 300),
            )
            return cursor.rowcount == 1


async def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    claimed, cancellation = await _finish_delivery_transition(
        _claim_completion_delivery(delegation_id, claim_id)
    )
    if cancellation is not None:
        if claimed:
            try:
                await _finish_delivery_transition(
                    _release_completion_delivery(delegation_id, claim_id)
                )
            except Exception as exc:
                raise cancellation from exc
        raise cancellation
    return claimed


async def claim_event_delivery(
    evt: dict[str, Any], consumer: str
) -> str | None:
    if evt.get("type") != "async_delegation":
        return ""
    delegation_id = str(evt.get("delegation_id") or "")
    if not delegation_id:
        return ""
    claim_id = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
    if await claim_completion_delivery(delegation_id, claim_id):
        return claim_id
    return None


async def _release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    now = time.time()
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET delivery_state = 'dropped',
                       delivery_claim = NULL, delivery_claimed_at = NULL,
                       updated_at = :updated_at
                       WHERE delegation_id = :delegation_id
                         AND delivery_state = 'pending'
                         AND delivery_claim = :claim_id
                         AND delivery_attempts >= :max_attempts"""
                ),
                {
                    "updated_at": now,
                    "delegation_id": delegation_id,
                    "claim_id": claim_id,
                    "max_attempts": _MAX_DELIVERY_ATTEMPTS,
                },
            )
            if cursor.rowcount == 1:
                logger.warning(
                    "Async delegation %s exhausted its %d delivery attempts; "
                    "marking terminally dropped (result remains queryable).",
                    delegation_id,
                    _MAX_DELIVERY_ATTEMPTS,
                )
                return True
            cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET delivery_claim = NULL,
                       delivery_claimed_at = NULL, updated_at = :updated_at
                       WHERE delegation_id = :delegation_id
                         AND delivery_state = 'pending'
                         AND delivery_claim = :claim_id"""
                ),
                {
                    "updated_at": now,
                    "delegation_id": delegation_id,
                    "claim_id": claim_id,
                },
            )
            return cursor.rowcount == 1

        return await postgres._write(_write)
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                """UPDATE async_delegations SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=? AND delivery_attempts>=?""",
                (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS),
            )
            if cursor.rowcount == 1:
                logger.warning(
                    "Async delegation %s exhausted its %d delivery attempts; "
                    "marking terminally dropped (result remains queryable).",
                    delegation_id,
                    _MAX_DELIVERY_ATTEMPTS,
                )
                return True
            cursor = await conn.execute(
                """UPDATE async_delegations SET delivery_claim=NULL,
                          delivery_claimed_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=?""",
                (now, delegation_id, claim_id),
            )
            return cursor.rowcount == 1


async def release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    return await _propagate_after_delivery_transition(
        _release_completion_delivery(delegation_id, claim_id)
    )


async def _drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET delivery_state = 'dropped',
                       updated_at = :updated_at, delivery_claim = NULL,
                       delivery_claimed_at = NULL
                       WHERE delegation_id = :delegation_id
                         AND delivery_state = 'pending'
                         AND delivery_claim = :claim_id"""
                ),
                {
                    "updated_at": time.time(),
                    "delegation_id": delegation_id,
                    "claim_id": claim_id,
                },
            )
            return cursor.rowcount == 1

        return await postgres._write(_write)
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                """UPDATE async_delegations SET delivery_state='dropped', updated_at=?,
                          delivery_claim=NULL, delivery_claimed_at=NULL
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=?""",
                (time.time(), delegation_id, claim_id),
            )
            return cursor.rowcount == 1


async def drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    return await _propagate_after_delivery_transition(
        _drop_completion_delivery(delegation_id, claim_id)
    )


async def _complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    now = time.time()
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _write(connection):
            cursor = await connection.execute(
                text(
                    """UPDATE async_delegations SET delivery_state = 'delivered',
                       delivered_at = :delivered_at, updated_at = :updated_at,
                       delivery_claim = NULL, delivery_claimed_at = NULL
                       WHERE delegation_id = :delegation_id
                         AND delivery_state = 'pending'
                         AND delivery_claim = :claim_id"""
                ),
                {
                    "delivered_at": now,
                    "updated_at": now,
                    "delegation_id": delegation_id,
                    "claim_id": claim_id,
                },
            )
            return cursor.rowcount == 1

        return await postgres._write(_write)
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                """UPDATE async_delegations SET delivery_state='delivered',
                          delivered_at=?, updated_at=?, delivery_claim=NULL,
                          delivery_claimed_at=NULL
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=?""",
                (now, now, delegation_id, claim_id),
            )
            return cursor.rowcount == 1


async def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    return await _propagate_after_delivery_transition(
        _complete_completion_delivery(delegation_id, claim_id)
    )


async def complete_event_delivery(evt: dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        await complete_completion_delivery(
            str(evt.get("delegation_id") or ""), claim_id
        )


async def release_event_delivery(evt: dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        await release_completion_delivery(
            str(evt.get("delegation_id") or ""), claim_id
        )


async def get_durable_delegation(
    delegation_id: str,
) -> dict[str, Any] | None:
    postgres = _postgres_session_db()
    if postgres is not None:
        from sqlalchemy import text

        async def _read(connection):
            cursor = await connection.execute(
                text(
                    """SELECT origin_session, state, dispatched_at, completed_at,
                              result_json, delivery_state, delivery_attempts,
                              origin_session_id
                       FROM async_delegations WHERE delegation_id = :delegation_id"""
                ),
                {"delegation_id": delegation_id},
            )
            return cursor.first()

        row = await postgres._read(_read)
        if row is None:
            return None
        return {
            "delegation_id": delegation_id,
            "origin_session": row[0],
            "state": row[1],
            "dispatched_at": row[2],
            "completed_at": row[3],
            "result": json.loads(row[4]) if row[4] else None,
            "delivery_state": row[5],
            "delivery_attempts": row[6],
            "origin_session_id": row[7] or "",
        }
    async with _db_lock():
        async with _transaction() as conn:
            cursor = await conn.execute(
                """SELECT origin_session, state, dispatched_at, completed_at,
                          result_json, delivery_state, delivery_attempts,
                          origin_session_id
                   FROM async_delegations WHERE delegation_id=?""",
                (delegation_id,),
            )
            row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "delegation_id": delegation_id,
        "origin_session": row[0],
        "state": row[1],
        "dispatched_at": row[2],
        "completed_at": row[3],
        "result": json.loads(row[4]) if row[4] else None,
        "delivery_state": row[5],
        "delivery_attempts": row[6],
        "origin_session_id": row[7] or "",
    }


def active_count() -> int:
    state = _current_scope_state()
    return sum(
        record.get("status") in {"running", "stalling", "finalizing"}
        for record in state.records.values()
    )


def active_for_session(origin_ui_session_id: str) -> int:
    if not origin_ui_session_id:
        return 0
    state = _current_scope_state()
    return sum(
        record.get("status") in {"running", "stalling", "finalizing"}
        and str(record.get("origin_ui_session_id") or "")
        == origin_ui_session_id
        for record in state.records.values()
    )


def active_task_count() -> int:
    state = _current_scope_state()
    total = 0
    for record in state.records.values():
        if record.get("status") not in {"running", "finalizing"}:
            continue
        goals = record.get("goals") if record.get("is_batch") else None
        total += len(goals) if isinstance(goals, (list, tuple)) and goals else 1
    return total


def _matches_session_selectors(
    record: dict[str, Any],
    *,
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    return (
        bool(origin_ui_session_id)
        and str(record.get("origin_ui_session_id") or "") == origin_ui_session_id
    ) or (
        bool(session_key)
        and str(record.get("session_key") or "") == session_key
    ) or (
        bool(parent_session_id)
        and str(record.get("parent_session_id") or "") == parent_session_id
    )


def has_live_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return False
    state = _current_scope_state()
    return any(
        record.get("status") in {"running", "stalling", "finalizing"}
        and _matches_session_selectors(
            record,
            session_key=session_key,
            origin_ui_session_id=origin_ui_session_id,
            parent_session_id=parent_session_id,
        )
        for record in state.records.values()
    )


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed(state: _DelegationScopeState) -> None:
    completed = [
        (record_id, record)
        for record_id, record in state.records.items()
        if record.get("status") not in {"running", "stalling", "finalizing"}
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    completed.sort(
        key=lambda item: item[1].get("completed_at")
        or item[1].get("dispatched_at")
        or 0
    )
    for record_id, _record in completed[
        : len(completed) - _MAX_RETAINED_COMPLETED
    ]:
        state.records.pop(record_id, None)


def _current_origin_session_id() -> str:
    try:
        from gateway.session_context import get_session_env

        if get_session_env("HERMES_SESSION_PLATFORM", "") != "api_server":
            return ""
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or ""
    except Exception:
        return ""


async def _await_runner(runner: Callable[[], Awaitable[dict[str, Any]]]):
    if not inspect.iscoroutinefunction(runner):
        raise TypeError("Async delegation runner must be a native async callable")
    result = runner()
    if not inspect.isawaitable(result):
        raise TypeError("Async delegation runner must return an awaitable")
    return await result


async def dispatch_async_delegation(
    *,
    goal: str,
    context: str | None,
    toolsets: list[str] | None,
    role: str,
    model: str | None,
    session_key: str,
    parent_session_id: str | None = None,
    runner: Callable[[], dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Callable[[], None] | None = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    progress_fn: Callable[[], tuple] | None = None,
) -> dict[str, Any]:
    return await _dispatch(
        goal=goal,
        goals=None,
        context=context,
        toolsets=toolsets,
        role=role,
        model=model,
        session_key=session_key,
        parent_session_id=parent_session_id,
        runner=runner,
        origin_ui_session_id=origin_ui_session_id,
        origin_session_id=origin_session_id,
        interrupt_fn=interrupt_fn,
        max_async_children=max_async_children,
        progress_fn=progress_fn,
        delegation_id=None,
        is_batch=False,
    )


async def dispatch_async_delegation_batch(
    *,
    goals: list[str],
    context: str | None,
    toolsets: list[str] | None,
    role: str,
    model: str | None,
    session_key: str,
    parent_session_id: str | None = None,
    runner: Callable[[], dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Callable[[], None] | None = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    delegation_id: str | None = None,
    progress_fn: Callable[[], tuple] | None = None,
) -> dict[str, Any]:
    combined_goal = (
        goals[0]
        if len(goals) == 1
        else f"{len(goals)} parallel subagents: "
        + "; ".join(goal[:40] for goal in goals)
    )
    return await _dispatch(
        goal=combined_goal,
        goals=list(goals),
        context=context,
        toolsets=toolsets,
        role=role,
        model=model,
        session_key=session_key,
        parent_session_id=parent_session_id,
        runner=runner,
        origin_ui_session_id=origin_ui_session_id,
        origin_session_id=origin_session_id,
        interrupt_fn=interrupt_fn,
        max_async_children=max_async_children,
        progress_fn=progress_fn,
        delegation_id=delegation_id,
        is_batch=True,
    )


async def _dispatch(
    *,
    goal: str,
    goals: list[str] | None,
    context: str | None,
    toolsets: list[str] | None,
    role: str,
    model: str | None,
    session_key: str,
    parent_session_id: str | None,
    runner: Callable[[], Awaitable[dict[str, Any]]],
    origin_ui_session_id: str,
    origin_session_id: str,
    interrupt_fn: Callable[[], None] | None,
    max_async_children: int,
    progress_fn: Callable[[], tuple] | None,
    delegation_id: str | None,
    is_batch: bool,
) -> dict[str, Any]:
    if not inspect.iscoroutinefunction(runner):
        return {
            "status": "rejected",
            "error": "Async delegation runner must be a native async callable",
        }
    state = await _activate_scope_state()
    delegation_id = delegation_id or _new_delegation_id()
    dispatched_at = time.time()
    running = sum(
        record.get("status") in {"running", "stalling"}
        for record in state.records.values()
    )
    if running >= max_async_children:
        suffix = (
            "or raise delegation.max_concurrent_children in config.yaml to allow "
            "more concurrent background units."
            if is_batch
            else "or run this task synchronously (background=false). Raise "
            "delegation.max_concurrent_children in config.yaml to allow more "
            "concurrent background subagents."
        )
        return {
            "status": "rejected",
            "error": (
                f"Async delegation capacity reached ({max_async_children} running). "
                f"Wait for one to finish (its result will re-enter the chat), {suffix}"
            ),
        }
    record: dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "progress_fn": progress_fn,
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
        "_owner_instance_id": _owner_instance_id(),
    }
    if is_batch:
        record["goals"] = goals
        record["is_batch"] = True
    state.records[delegation_id] = record
    try:
        await _persist_dispatch(record)
        task = asyncio.create_task(
            _run_dispatch(state, delegation_id, runner, is_batch),
            name=f"async-delegate-{delegation_id}",
        )
    except BaseException:
        state.records.pop(delegation_id, None)
        await _delete_durable_delegation(delegation_id)
        raise
    state.tasks[delegation_id] = task
    task.add_done_callback(
        lambda _task, rid=delegation_id, owner=state: owner.tasks.pop(rid, None)
    )
    if progress_fn is not None or _postgres_session_db(state) is not None:
        _ensure_stale_monitor(state)
    return {"status": "dispatched", "delegation_id": delegation_id}


async def _run_dispatch(
    state: _DelegationScopeState,
    delegation_id: str,
    runner: Callable[[], Awaitable[dict[str, Any]]],
    is_batch: bool,
) -> None:
    record = state.records.get(delegation_id) or {}
    dispatched_at = record.get("dispatched_at") or time.time()
    result: dict[str, Any]
    status = "error"
    cancellation: asyncio.CancelledError | None = None
    try:
        result = await _await_runner(runner) or {}
        if is_batch:
            child_results = result.get("results") or []
            status = (
                "error"
                if child_results
                and all(
                    child.get("status") not in {"completed", "success"}
                    for child in child_results
                )
                else "completed"
            )
        else:
            status = result.get("status") or "completed"
    except asyncio.CancelledError as exc:  # noqa: ASYNC103 - persist before re-raise
        cancellation = exc
        result = (
            {
                "results": [],
                "error": "Async delegation cancelled",
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            if is_batch
            else {
                "status": "interrupted",
                "summary": None,
                "error": "Async delegation cancelled",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
                "exit_reason": "interrupted",
            }
        )
        status = "interrupted"
    except Exception as exc:
        logger.exception("Async delegation %s crashed", delegation_id)
        result = (
            {
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            if is_batch
            else {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
            }
        )
        status = "error"
    if cancellation is not None:
        try:
            await _finish_delivery_transition(
                _finalize(state, delegation_id, result, status, is_batch=is_batch)
            )
        except Exception as exc:
            raise cancellation from exc
        raise cancellation
    await _finalize(state, delegation_id, result, status, is_batch=is_batch)


def _begin_finalization(
    state: _DelegationScopeState, delegation_id: str
) -> dict[str, Any] | None:
    record = state.records.get(delegation_id)
    if record is None or record.get("status") not in {"running", "stalling"}:
        return None
    record["status"] = "finalizing"
    record["completed_at"] = time.time()
    record["interrupt_fn"] = None
    record["progress_fn"] = None
    record["_progress_token"] = None
    return dict(record)


async def _finalize(
    state: _DelegationScopeState,
    delegation_id: str,
    result: dict[str, Any],
    status: str,
    *,
    is_batch: bool,
) -> None:
    event_record = _begin_finalization(state, delegation_id)
    if event_record is None:
        return
    if is_batch:
        await _push_batch_completion_event(state, event_record, result, status)
    else:
        await _push_completion_event(state, event_record, result, status)
    record = state.records.get(delegation_id)
    if record is not None:
        record["status"] = status
    _prune_completed(state)
    await _stop_stale_monitor_if_idle(state)


async def _push_completion_event(
    state: _DelegationScopeState,
    record: dict[str, Any],
    result: dict[str, Any],
    status: str,
) -> None:
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()
    event = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal": record.get("goal", ""),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": result.get("model") or record.get("model"),
        "status": status,
        "summary": result.get("summary"),
        "error": result.get("error"),
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get(
            "duration_seconds", round(completed_at - dispatched_at, 2)
        ),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
    }
    _copy_stall_metadata(result, event)
    await _publish_completion(state, event, result)


async def _push_batch_completion_event(
    state: _DelegationScopeState,
    record: dict[str, Any],
    combined: dict[str, Any],
    status: str,
) -> None:
    event = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal": record.get("goal", ""),
        "goals": record.get("goals"),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": record.get("model"),
        "status": status,
        "is_batch": True,
        "results": combined.get("results") or [],
        "live_transcripts": combined.get("live_transcripts"),
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": record.get("dispatched_at") or time.time(),
        "completed_at": record.get("completed_at") or time.time(),
    }
    _copy_stall_metadata(combined, event)
    await _publish_completion(state, event, combined)


def _copy_stall_metadata(source: dict[str, Any], event: dict[str, Any]) -> None:
    for key in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if key in source:
            event[key] = source[key]


async def _publish_completion(
    state: _DelegationScopeState,
    event: dict[str, Any],
    result: dict[str, Any],
) -> None:
    home_token = set_hermes_home_override(state.profile_home)
    scope_token = _active_scope.set((_lexical_profile_identity(), state))
    try:
        persisted = await _persist_completion(event, result)
        # A PostgreSQL owner that lost its lease must not re-publish a
        # completion after another worker fenced the row and recovered it.
        if persisted is False:
            return
        from tools.process_registry import process_registry

        await process_registry._activate_profile_state()
        process_registry.completion_queue.put_nowait(event)
    finally:
        _active_scope.reset(scope_token)
        reset_hermes_home_override(home_token)


def _ensure_stale_monitor(state: _DelegationScopeState) -> None:
    global _monitor_task
    if state.monitor_task is not None and not state.monitor_task.done():
        return
    state.monitor_task = asyncio.create_task(
        _stale_monitor_loop(state), name="async-delegate-stale-monitor"
    )
    _monitor_task = state.monitor_task


async def _stop_stale_monitor_if_idle(state: _DelegationScopeState) -> None:
    task = state.monitor_task
    if task is None or task.done() or task is asyncio.current_task():
        return
    if any(
        record.get("status") in {"running", "stalling"}
        and callable(record.get("progress_fn"))
        for record in state.records.values()
    ) or (
        _postgres_session_db(state) is not None
        and any(
            record.get("status") in {"running", "finalizing"}
            for record in state.records.values()
        )
    ):
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _renew_postgres_leases(state: _DelegationScopeState) -> None:
    try:
        postgres = _postgres_session_db(state)
        if postgres is None:
            return
        from sqlalchemy import text

        async def _write(connection):
            await connection.execute(
                text(
                    """UPDATE async_delegations
                       SET lease_expires_at =
                           EXTRACT(EPOCH FROM clock_timestamp()) + :lease
                       WHERE owner_instance_id = :owner
                         AND state IN ('running', 'finalizing')"""
                ),
                {"lease": _OWNER_LEASE_SECONDS, "owner": _owner_instance_id()},
            )

        await postgres._write(_write)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "Could not renew PostgreSQL async delegation lease", exc_info=True
        )


async def _stale_monitor_loop(state: _DelegationScopeState) -> None:
    global _monitor_task
    current = asyncio.current_task()
    try:
        while True:
            await asyncio.sleep(_STALE_CHECK_INTERVAL)
            await _renew_postgres_leases(state)
            now = time.time()
            stalled: list[str] = []
            expired: list[str] = []
            for record in list(state.records.values()):
                status = record.get("status")
                if status == "stalling":
                    if (
                        now - (record.get("_interrupted_at") or now)
                        >= _STALL_GRACE_SECONDS
                    ):
                        expired.append(record["delegation_id"])
                    continue
                if status != "running" or not callable(record.get("progress_fn")):
                    continue
                try:
                    token, in_tool = record["progress_fn"]()
                except Exception:
                    token, in_tool = record.get("_progress_token"), False
                if token != record.get("_progress_token"):
                    record["_progress_token"] = token
                    record["_progress_ts"] = now
                    continue
                quiet_for = now - (record.get("_progress_ts") or now)
                threshold = (
                    _STALE_IN_TOOL_SECONDS if in_tool else _STALE_IDLE_SECONDS
                )
                if quiet_for >= threshold:
                    record["status"] = "stalling"
                    record["_interrupted_at"] = now
                    record["_stall_quiet_seconds"] = round(quiet_for, 2)
                    record["_stall_threshold_seconds"] = threshold
                    record["_stall_in_tool"] = bool(in_tool)
                    stalled.append(record["delegation_id"])
            for delegation_id in stalled:
                record = state.records.get(delegation_id)
                try:
                    callback = record.get("interrupt_fn") if record else None
                    if callable(callback):
                        callback()
                except Exception:
                    logger.debug(
                        "Async delegation stall interrupt failed", exc_info=True
                    )
            for delegation_id in expired:
                await _finalize_stalled(state, delegation_id)
            postgres_active = _postgres_session_db(state) is not None and any(
                record.get("status") in {"running", "finalizing"}
                for record in state.records.values()
            )
            if not postgres_active and not any(
                record.get("status") in {"running", "stalling"}
                and callable(record.get("progress_fn"))
                for record in state.records.values()
            ):
                return
    finally:
        if state.monitor_task is current:
            state.monitor_task = None
        if _monitor_task is current:
            _monitor_task = None


async def _finalize_stalled(
    state: _DelegationScopeState, delegation_id: str
) -> None:
    event_record = _begin_finalization(state, delegation_id)
    if event_record is None:
        return
    completed_at = event_record.get("completed_at") or time.time()
    duration = round(
        completed_at - (event_record.get("dispatched_at") or completed_at), 2
    )
    error = (
        f"Async delegation {delegation_id} stalled: the detached subagent "
        "stopped making progress, did not respond to interruption, and never "
        "produced a completion event. Re-dispatch the task if it is still needed."
    )
    in_tool = event_record.get("_stall_in_tool")
    metadata = {
        "stalled_after_quiet_seconds": event_record.get("_stall_quiet_seconds"),
        "stall_threshold_seconds": event_record.get("_stall_threshold_seconds"),
        "stall_phase": "in_tool" if in_tool else "idle" if in_tool is not None else None,
        "stall_grace_seconds": _STALL_GRACE_SECONDS,
    }
    if event_record.get("is_batch"):
        await _push_batch_completion_event(
            state,
            event_record,
            {
                "results": [],
                "error": error,
                "total_duration_seconds": duration,
                **metadata,
            },
            "stalled",
        )
    else:
        await _push_completion_event(
            state,
            event_record,
            {
                "status": "stalled",
                "summary": None,
                "error": error,
                "api_calls": 0,
                "duration_seconds": duration,
                "exit_reason": "stalled",
                **metadata,
            },
            "stalled",
        )
    record = state.records.get(delegation_id)
    if record is not None:
        record["status"] = "stalled"
    task = state.tasks.get(delegation_id)
    if task is not None and not task.done():
        task.cancel()


def _children_activity_from_token(
    token: Any, now: float
) -> list[dict[str, Any] | None] | None:
    try:
        parts = list(token)
    except TypeError:
        return None
    output: list[dict[str, Any] | None] = []
    for part in parts:
        if isinstance(part, (list, tuple)) and len(part) >= 2:
            entry: dict[str, Any] = {
                "api_calls": part[0],
                "current_tool": part[1],
            }
            if len(part) >= 3 and isinstance(part[2], (int, float)):
                entry["seconds_since_activity"] = round(
                    max(0.0, now - float(part[2])), 1
                )
            output.append(entry)
        else:
            output.append(None)
    return output


def list_async_delegations() -> list[dict[str, Any]]:
    state = _current_scope_state()
    now = time.time()
    items: list[dict[str, Any]] = []
    for record in list(state.records.values()):
        item = {
            key: value
            for key, value in record.items()
            if key not in {"interrupt_fn", "progress_fn"}
            and not key.startswith("_")
        }
        status = record.get("status")
        if status in {"running", "stalling"}:
            timestamp = record.get("_progress_ts")
            if timestamp:
                item["seconds_since_progress"] = round(now - timestamp, 1)
            progress_fn = record.get("progress_fn")
            if callable(progress_fn):
                try:
                    sampled = progress_fn()
                    token, in_tool = sampled
                    activity = _children_activity_from_token(token, now)
                    if activity is not None:
                        item["children_activity"] = activity
                    item["in_tool"] = bool(in_tool)
                except Exception:
                    pass
        if status in {"stalling", "stalled"}:
            for source, target in (
                ("_stall_quiet_seconds", "stalled_after_quiet_seconds"),
                ("_stall_threshold_seconds", "stall_threshold_seconds"),
                ("_stall_in_tool", "stall_in_tool"),
            ):
                if record.get(source) is not None:
                    item[target] = record[source]
        items.append(item)
    return items


async def _interrupt_records(
    state: _DelegationScopeState,
    records: list[dict[str, Any]],
    reason: str,
) -> int:
    if not records:
        return 0
    for record in records:
        try:
            callback = record.get("interrupt_fn")
            if callable(callback):
                callback()
        except Exception:
            logger.debug("Async delegation interrupt failed", exc_info=True)
        task = state.tasks.get(record["delegation_id"])
        if task is not None and not task.done():
            task.cancel()
    tasks = [
        state.tasks[record["delegation_id"]]
        for record in records
        if record["delegation_id"] in state.tasks
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Interrupted %d async delegation(s) (%s)", len(records), reason)
    return len(records)


async def interrupt_all(reason: str = "shutdown") -> int:
    state = await _activate_scope_state()
    return await _interrupt_records(
        state,
        [
            record
            for record in list(state.records.values())
            if record.get("status") in {"running", "stalling"}
        ],
        reason,
    )


async def interrupt_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
    reason: str = "session_end",
) -> int:
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return 0
    state = await _activate_scope_state()
    return await _interrupt_records(
        state,
        [
            record
            for record in list(state.records.values())
            if record.get("status") in {"running", "stalling"}
            and _matches_session_selectors(
                record,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
        ],
        reason,
    )


async def _reset_for_tests() -> None:
    global _monitor_task
    loop = asyncio.get_running_loop()
    with _scope_guard:
        states = list(_scope_states.get(loop, {}).values())
    if all(state is not _legacy_scope for state in states):
        states.append(_legacy_scope)
    tasks = [
        task
        for state in states
        for task in state.tasks.values()
        if not task.done()
    ]
    monitors = [
        state.monitor_task
        for state in states
        if state.monitor_task is not None and not state.monitor_task.done()
    ]
    for task in [*tasks, *monitors]:
        task.cancel()
    if tasks or monitors:
        await asyncio.gather(*tasks, *monitors, return_exceptions=True)
    for state in states:
        state.monitor_task = None
        state.tasks.clear()
        state.records.clear()
        state.restored_queues.clear()
        state.db_lock = None
        state.db_lock_users = 0
        state.restore_lock = None
        state.restore_lock_users = 0
        state.session_db_ref = None
    with _scope_guard:
        _scope_states.pop(loop, None)
        _scope_aliases.pop(loop, None)
    _active_scope.set(None)
    _active_session_db.set(_SESSION_DB_UNSET)
    _project_compatibility_globals(_legacy_scope)
    _monitor_task = None
    _process_start_times.clear()
    _db_locks.pop(loop, None)
    _restore_locks.pop(loop, None)
