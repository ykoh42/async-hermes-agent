"""Native-async runtime components for the retained Mem0 OSS backend.

The upstream ``mem0ai`` package implements its SQLite history store with
``sqlite3`` and a ``threading.Lock``.  This module keeps that schema and result
contract while moving database lifecycle and queries behind ``aiosqlite``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any
import uuid

import aiosqlite

logger = logging.getLogger(__name__)

_HISTORY_COLUMNS = (
    "id",
    "memory_id",
    "old_memory",
    "new_memory",
    "event",
    "created_at",
    "updated_at",
    "is_deleted",
    "actor_id",
    "role",
)

_CREATE_HISTORY = """
    CREATE TABLE IF NOT EXISTS history (
        id           TEXT PRIMARY KEY,
        memory_id    TEXT,
        old_memory   TEXT,
        new_memory   TEXT,
        event        TEXT,
        created_at   DATETIME,
        updated_at   DATETIME,
        is_deleted   INTEGER,
        actor_id     TEXT,
        role         TEXT
    )
"""

_CREATE_MESSAGES = """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_scope TEXT,
        role TEXT,
        content TEXT,
        name TEXT,
        created_at DATETIME
    )
"""


class SQLiteManager:
    """Async equivalent of ``mem0.memory.storage.SQLiteManager``.

    Construction is state-only.  The connection and schema are opened lazily
    at the first awaited operation so creating an ``OSSBackend`` never performs
    file I/O in its constructor.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def _connection_locked(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("Cannot use a closed SQLiteManager")
        if self.connection is not None:
            return self.connection

        connection = await aiosqlite.connect(self.db_path)
        try:
            self.connection = connection
            await self._migrate_history_table_locked(connection)
            await connection.execute(_CREATE_HISTORY)
            await connection.execute(_CREATE_MESSAGES)
            await connection.commit()
        except BaseException:
            self.connection = None
            await connection.close()
            raise
        return connection

    async def _migrate_history_table_locked(
        self, connection: aiosqlite.Connection
    ) -> None:
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
        )
        try:
            exists = await cursor.fetchone()
        finally:
            await cursor.close()
        if exists is None:
            return

        cursor = await connection.execute("PRAGMA table_info(history)")
        try:
            old_columns = {row[1] for row in await cursor.fetchall()}
        finally:
            await cursor.close()
        if old_columns == set(_HISTORY_COLUMNS):
            return

        logger.info("Migrating history table to new schema (no convo columns).")
        try:
            await connection.execute("BEGIN")
            await connection.execute("DROP TABLE IF EXISTS history_old")
            await connection.execute("ALTER TABLE history RENAME TO history_old")
            await connection.execute(_CREATE_HISTORY.replace("IF NOT EXISTS ", ""))
            intersecting = [
                column for column in _HISTORY_COLUMNS if column in old_columns
            ]
            if intersecting:
                columns = ", ".join(intersecting)
                await connection.execute(
                    f"INSERT INTO history ({columns}) "
                    f"SELECT {columns} FROM history_old"
                )
            await connection.execute("DROP TABLE history_old")
            await connection.commit()
            logger.info("History table migration completed successfully.")
        except BaseException:
            await connection.rollback()
            logger.exception("History table migration failed")
            raise

    async def add_history(
        self,
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        *,
        created_at: str | None = None,
        updated_at: str | None = None,
        is_deleted: int = 0,
        actor_id: str | None = None,
        role: str | None = None,
    ) -> None:
        async with self._lock:
            connection = await self._connection_locked()
            try:
                await connection.execute("BEGIN")
                await connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def batch_add_history(self, records: list[dict[str, Any]]) -> None:
        values = [
            (
                str(uuid.uuid4()),
                record.get("memory_id"),
                record.get("old_memory"),
                record.get("new_memory"),
                record.get("event"),
                record.get("created_at"),
                record.get("updated_at"),
                record.get("is_deleted", 0),
                record.get("actor_id"),
                record.get("role"),
            )
            for record in records
        ]
        async with self._lock:
            connection = await self._connection_locked()
            try:
                await connection.execute("BEGIN")
                await connection.executemany(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            connection = await self._connection_locked()
            cursor = await connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at ASC, DATETIME(updated_at) ASC
                """,
                (memory_id,),
            )
            try:
                rows = await cursor.fetchall()
            finally:
                await cursor.close()

        return [
            {
                "id": row[0],
                "memory_id": row[1],
                "old_memory": row[2],
                "new_memory": row[3],
                "event": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "is_deleted": bool(row[7]),
                "actor_id": row[8],
                "role": row[9],
            }
            for row in rows
        ]

    async def save_messages(
        self, messages: list[dict[str, Any]], session_scope: str
    ) -> None:
        if not messages:
            return
        now = datetime.now(timezone.utc).isoformat()
        values = [
            (
                str(uuid.uuid4()),
                session_scope,
                message.get("role"),
                message.get("content"),
                message.get("name"),
                now,
            )
            for message in messages
        ]
        async with self._lock:
            connection = await self._connection_locked()
            try:
                await connection.execute("BEGIN")
                await connection.executemany(
                    """
                    INSERT INTO messages (
                        id, session_scope, role, content, name, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                await connection.execute(
                    """
                    DELETE FROM messages
                    WHERE session_scope = ? AND id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM messages
                            WHERE session_scope = ?
                            ORDER BY created_at DESC
                            LIMIT 10
                        )
                    )
                    """,
                    (session_scope, session_scope),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def get_last_messages(
        self, session_scope: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        async with self._lock:
            connection = await self._connection_locked()
            cursor = await connection.execute(
                """
                SELECT role, content, name, created_at FROM (
                    SELECT role, content, name, created_at
                    FROM messages
                    WHERE session_scope = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (session_scope, limit),
            )
            try:
                rows = await cursor.fetchall()
            finally:
                await cursor.close()

        return [
            {
                "role": row[0],
                "content": row[1],
                "name": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    async def reset(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed SQLiteManager")
            connection = await self._connection_locked()
            try:
                await connection.execute("BEGIN")
                await connection.execute("DROP TABLE IF EXISTS history")
                await connection.execute("DROP TABLE IF EXISTS messages")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            connection = self.connection
            self.connection = None
            if connection is not None:
                close_task = asyncio.create_task(connection.close())
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    try:
                        await close_task
                    except Exception:
                        logger.exception(
                            "Mem0 SQLite cleanup failed during cancellation"
                        )
                    raise
