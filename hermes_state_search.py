"""Native-async search and FTS maintenance for ``SessionDB``.

This restores the upstream search-mixin path while retaining the original
search routing, result shapes, and FTS maintenance behavior. The mixin owns no
runtime state and must not import ``hermes_state``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from typing import Any
from collections.abc import Callable, Collection

from agent.skill_commands import describe_skill_invocation
from hermes_state_common import (
    FTS_CJK_STALE_KEY,
    FTS_STALE_KEY,
    FTS_SQL,
    FTS_STORAGE_VERSION,
    FTS_TRIGRAM_SQL,
    MAX_FTS5_QUERY_CHARS,
    SCHEMA_VERSION,
    _FTS_CJK_TRIGGERS,
    _session_runtime_config_value,
    escape_like as _escape_like,
)


logger = logging.getLogger("hermes_state")


class SessionSearchMixin:
    """Native-async retained search and FTS behavior for ``SessionDB``."""

    _SEARCH_MESSAGE_RESULT_FIELDS = (
        "id",
        "session_id",
        "role",
        "snippet",
        "timestamp",
        "tool_name",
        "source",
        "model",
        "session_started",
        "context",
    )

    @classmethod
    def _search_message_fields(
        cls, fields: Collection[str] | None
    ) -> tuple[str, ...] | None:
        """Validate and canonically order an optional search projection."""
        if fields is None:
            return None
        if isinstance(fields, str):
            raise TypeError(
                "search fields must be a collection of field names, not a string"
            )
        requested = set(fields)
        unknown = requested.difference(cls._SEARCH_MESSAGE_RESULT_FIELDS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown search result field(s): {names}")
        return tuple(
            field
            for field in cls._SEARCH_MESSAGE_RESULT_FIELDS
            if field in requested
        )

    async def _try_incremental_merge_fts(self) -> None:
        """Run one bounded FTS5 merge pass after a completed write."""
        if not self._fts_enabled:
            return
        try:
            await self._merge_fts_incrementally(
                max_pages=self._FTS_MERGE_MAX_PAGES_PER_INDEX
            )
        except sqlite3.Error as exc:
            logger.warning("FTS incremental merge failed: %s", exc)

    async def fts_rebuild_status(self) -> dict[str, Any] | None:
        """Return deferred FTS rebuild progress without sync metadata access."""
        rows = await self._read_fetchall(
            "SELECT key, value FROM state_meta WHERE key IN (?, ?)",
            ("fts_rebuild_high_water", "fts_rebuild_progress"),
        )
        meta = {row["key"]: row["value"] for row in rows}
        high_water = meta.get("fts_rebuild_high_water")
        if high_water is None:
            return None
        try:
            total = int(high_water)
            indexed = int(meta.get("fts_rebuild_progress") or 0)
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        return {
            "pending": True,
            "total": total,
            "indexed": indexed,
            "percent": min(100, int(100 * indexed / total)),
        }

    async def _fts_rebuild_finish(self) -> None:
        """Finalize the deferred rebuild with a boundary sweep."""
        include_trigram = self._trigram_available

        async def _finish(connection):
            async with connection.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water'"
            ) as cursor:
                high_water_row = await cursor.fetchone()
            if high_water_row is not None:
                high_water = int(high_water_row[0])
                lower, upper = high_water - 1000, high_water + 1000
                cursor = await connection.execute(
                    "INSERT INTO messages_fts"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m WHERE m.id > ? AND m.id <= ? "
                    "AND NOT EXISTS (SELECT 1 FROM messages_fts_docsize d "
                    "WHERE d.id = m.id)",
                    (lower, upper),
                )
                await cursor.close()
                if include_trigram:
                    cursor = await connection.execute(
                        "INSERT INTO messages_fts_trigram"
                        "(rowid, content, tool_name, tool_calls) "
                        "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                        "FROM messages m WHERE m.id > ? AND m.id <= ? "
                        "AND m.role <> 'tool' AND NOT EXISTS "
                        "(SELECT 1 FROM messages_fts_trigram_docsize d "
                        "WHERE d.id = m.id)",
                        (lower, upper),
                    )
                    await cursor.close()
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key IN "
                "('fts_rebuild_high_water', 'fts_rebuild_progress')"
            )
            await cursor.close()

        await self._execute_write(_finish)
        logger.info("Deferred FTS rebuild complete — all messages indexed.")

    async def _fts_teardown_trash_step(self) -> bool:
        """Tear down one bounded chunk of a demoted v22 shadow table."""
        connection = await self._get_connection()
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE ? ESCAPE '\\'",
            (_escape_like(self._FTS_TRASH_PREFIX) + "%",),
        )
        try:
            trash = [row[0] for row in await cursor.fetchall()]
        finally:
            await cursor.close()
        if not trash:
            return False
        table = trash[0]

        async def _teardown(connection):
            info_cursor = await connection.execute(f"PRAGMA table_info({table})")
            try:
                primary_keys = [
                    row[1] for row in await info_cursor.fetchall() if row[5] > 0
                ]
            finally:
                await info_cursor.close()
            key = ", ".join(primary_keys) if primary_keys else "rowid"
            delete_cursor = await connection.execute(
                f"DELETE FROM {table} WHERE ({key}) IN "
                f"(SELECT {key} FROM {table} LIMIT "
                f"{self._FTS_REBUILD_CHUNK_ROWS})"
            )
            try:
                deleted = delete_cursor.rowcount
            finally:
                await delete_cursor.close()
            if deleted == 0:
                cursor = await connection.execute(f"DROP TABLE IF EXISTS {table}")
                await cursor.close()
                logger.info("Old FTS shadow table %s torn down.", table)
            return True

        try:
            return bool(await self._execute_write(_teardown))
        except sqlite3.OperationalError as exc:
            logger.debug("FTS trash teardown chunk failed (will retry): %s", exc)
            return True

    async def fts_rebuild_step(self) -> bool:
        """Backfill one chunk of the deferred FTS rebuild."""
        await self._get_connection()
        if not self._fts_enabled:
            return False
        high_water_raw = await self.get_meta("fts_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)
        include_trigram = self._trigram_available

        async def _backfill(connection):
            async with connection.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_rebuild_progress'"
            ) as cursor:
                progress_row = await cursor.fetchone()
            if progress_row is None:
                return False
            progress = int(progress_row[0])
            if progress >= high_water:
                return False
            upper = min(progress + self._FTS_REBUILD_CHUNK_ROWS, high_water)
            cursor = await connection.execute(
                "INSERT INTO messages_fts"
                "(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id > ? AND id <= ?",
                (progress, upper),
            )
            await cursor.close()
            if include_trigram:
                cursor = await connection.execute(
                    "INSERT INTO messages_fts_trigram"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT id, content, tool_name, tool_calls FROM messages "
                    "WHERE id > ? AND id <= ? AND role <> 'tool'",
                    (progress, upper),
                )
                await cursor.close()
            cursor = await connection.execute(
                "UPDATE state_meta SET value = ? "
                "WHERE key = 'fts_rebuild_progress'",
                (str(upper),),
            )
            await cursor.close()
            return upper < high_water

        try:
            more = await self._execute_write(_backfill)
        except sqlite3.OperationalError as exc:
            logger.debug("FTS rebuild chunk failed (will retry): %s", exc)
            return True
        if more is False:
            status = await self.fts_rebuild_status()
            if status is not None and status["indexed"] >= status["total"]:
                await self._fts_rebuild_finish()
            return False
        return bool(more)

    async def fts_cjk_rebuild_status(self) -> dict[str, Any] | None:
        """CJK-index backfill progress, or None when none is pending."""
        rows = await self._read_fetchall(
            "SELECT key, value FROM state_meta WHERE key IN (?, ?)",
            ("fts_cjk_rebuild_high_water", "fts_cjk_rebuild_progress"),
        )
        meta = {row["key"]: row["value"] for row in rows}
        high_water = meta.get("fts_cjk_rebuild_high_water")
        if high_water is None:
            return None
        total = int(high_water)
        indexed = int(meta.get("fts_cjk_rebuild_progress") or 0)
        if total <= 0:
            return None
        return {
            "pending": True,
            "total": total,
            "indexed": indexed,
            "percent": min(100, int(100 * indexed / total)),
        }

    async def fts_cjk_rebuild_step(self) -> bool:
        """Backfill one chunk of the CJK index. True while work remains."""
        await self._get_connection()
        if not self._fts_enabled or not self._fts_cjk_loaded:
            return False
        high_water_raw = await self.get_meta("fts_cjk_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)

        async def _backfill(connection):
            async with connection.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_cjk_rebuild_progress'"
            ) as cursor:
                progress_row = await cursor.fetchone()
            if progress_row is None:
                return False
            progress = int(progress_row[0])
            if progress >= high_water:
                return False
            upper = min(progress + self._FTS_REBUILD_CHUNK_ROWS, high_water)
            cursor = await connection.execute(
                "INSERT INTO messages_fts_cjk"
                "(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id > ? AND id <= ? AND role <> 'tool'",
                (progress, upper),
            )
            await cursor.close()
            cursor = await connection.execute(
                "UPDATE state_meta SET value = ? "
                "WHERE key = 'fts_cjk_rebuild_progress'",
                (str(upper),),
            )
            await cursor.close()
            return upper < high_water

        try:
            more = await self._execute_write(_backfill)
        except sqlite3.OperationalError as exc:
            logger.debug("CJK FTS rebuild chunk failed (will retry): %s", exc)
            return True
        if more is False:
            status = await self.fts_cjk_rebuild_status()
            if status is not None and status["indexed"] >= status["total"]:
                await self._fts_cjk_rebuild_finish()
            return False
        return bool(more)

    async def _fts_cjk_rebuild_finish(self) -> None:
        """Boundary sweep, clear CJK markers, and enable CJK reads."""

        async def _finish(connection):
            async with connection.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_cjk_rebuild_high_water'"
            ) as cursor:
                high_water_row = await cursor.fetchone()
            if high_water_row is not None:
                high_water = int(high_water_row[0])
                lower, upper = high_water - 1000, high_water + 1000
                cursor = await connection.execute(
                    "INSERT INTO messages_fts_cjk"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m WHERE m.id > ? AND m.id <= ? "
                    "AND m.role <> 'tool' AND NOT EXISTS "
                    "(SELECT 1 FROM messages_fts_cjk_docsize d "
                    "WHERE d.id = m.id)",
                    (lower, upper),
                )
                await cursor.close()
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', "
                "'fts_cjk_rebuild_progress')"
            )
            await cursor.close()

        await self._execute_write(_finish)
        self._fts_cjk_available = True
        logger.info("CJK FTS index backfill complete — serving CJK search.")

    async def _fts_cjk_reset_if_stale(self) -> None:
        """Reset a stale CJK index so it can be rebuilt from scratch."""
        if not self._fts_cjk_loaded:
            return

        async def _reset(connection):
            async with connection.execute(
                "SELECT 1 FROM state_meta WHERE key = ?",
                (FTS_CJK_STALE_KEY,),
            ) as cursor:
                stale = await cursor.fetchone()
            if stale is None:
                return False
            for trigger in _FTS_CJK_TRIGGERS:
                cursor = await connection.execute(
                    f"DROP TRIGGER IF EXISTS {trigger}"
                )
                await cursor.close()
            cursor = await connection.execute(
                "DROP TABLE IF EXISTS messages_fts_cjk"
            )
            await cursor.close()
            cursor = await connection.execute(
                "DROP VIEW IF EXISTS messages_fts_cjk_src"
            )
            await cursor.close()
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key IN "
                f"('{FTS_CJK_STALE_KEY}', 'fts_cjk_rebuild_high_water', "
                "'fts_cjk_rebuild_progress')"
            )
            await cursor.close()
            return True

        if await self._execute_write(_reset):
            async with self._get_write_lock():
                connection = await self._get_connection()
                await self._ensure_fts_cjk_schema(connection)
                await connection.commit()

    async def _fts_external_index_empty_with_messages(self, conn) -> bool:
        """Return whether a populated DB has an empty external FTS index."""
        try:
            async with conn.execute(
                "SELECT EXISTS(SELECT 1 FROM messages)"
            ) as cursor:
                has_message = await cursor.fetchone()
            if not has_message[0]:
                return False
            async with conn.execute(
                "SELECT EXISTS(SELECT 1 FROM messages_fts_docsize)"
            ) as cursor:
                has_fts = await cursor.fetchone()
            return not has_fts[0]
        except sqlite3.OperationalError:
            return False

    async def _fts_index_known_empty(self, conn) -> bool:
        try:
            async with conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ) as cursor:
                row = await cursor.fetchone()
            return int(row[0]) == 0
        except sqlite3.OperationalError:
            return True

    async def _reset_fts_index_to_empty(self, conn) -> None:
        for table in ("messages_fts", "messages_fts_trigram"):
            try:
                cursor = await conn.execute(
                    f"INSERT INTO {table}({table}) VALUES('delete-all')"
                )
                await cursor.close()
            except sqlite3.OperationalError:
                pass

    async def _seed_fts_rebuild_markers(
        self, conn, *, force: bool = False
    ) -> int:
        async with conn.execute(
            "SELECT value FROM state_meta "
            "WHERE key = 'fts_rebuild_high_water'"
        ) as cursor:
            existing = await cursor.fetchone()
        if existing is not None and not force:
            high_water = int(existing[0])
            async with conn.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_rebuild_progress'"
            ) as cursor:
                progress = await cursor.fetchone()
            if progress is None:
                if not await self._fts_index_known_empty(conn):
                    await self._reset_fts_index_to_empty(conn)
                cursor = await conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES "
                    "('fts_rebuild_progress', '0') "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                )
                await cursor.close()
            return high_water

        async with conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages"
        ) as cursor:
            row = await cursor.fetchone()
        high_water = int(row[0])
        for key, value in (
            ("fts_rebuild_high_water", str(high_water)),
            ("fts_rebuild_progress", "0"),
        ):
            cursor = await conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await cursor.close()
        return high_water

    async def _repair_optimize_bookkeeping(self) -> None:
        """Heal interrupted demote/backfill bookkeeping before optimize."""

        async def _repair(connection):
            async with connection.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water'"
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None:
                async with connection.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_progress'"
                ) as cursor:
                    progress = await cursor.fetchone()
                if progress is None:
                    if not await self._fts_index_known_empty(connection):
                        await self._reset_fts_index_to_empty(connection)
                    cursor = await connection.execute(
                        "INSERT INTO state_meta (key, value) VALUES "
                        "('fts_rebuild_progress', '0') "
                        "ON CONFLICT(key) DO UPDATE SET value = '0'"
                    )
                    await cursor.close()
                return
            if await self._db_has_legacy_inline_fts(connection):
                return
            if await self._fts_external_index_empty_with_messages(connection):
                cursor = await connection.execute(
                    "DELETE FROM state_meta WHERE key = 'fts_storage_version'"
                )
                await cursor.close()
                await self._seed_fts_rebuild_markers(connection, force=True)

        await self._execute_write(_repair)

    async def fts_optimize_available(self) -> bool:
        """Return whether storage migration, backfill, or teardown is pending."""
        connection = await self._get_connection()
        if not self._fts_enabled or self.read_only:
            return False
        async with self._get_write_lock():
            if await self._db_has_legacy_inline_fts(connection):
                return True
            async with connection.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ) as cursor:
                pending = await cursor.fetchone()
            if pending is not None:
                return True
            if self._fts_cjk_loaded:
                async with connection.execute(
                    "SELECT 1 FROM state_meta WHERE key IN "
                    f"('fts_cjk_rebuild_high_water', "
                    f"'{FTS_CJK_STALE_KEY}') LIMIT 1"
                ) as cursor:
                    cjk_pending = await cursor.fetchone()
                if cjk_pending is not None:
                    return True
            if await self._has_fts_trash(connection):
                return True
            return await self._fts_external_index_empty_with_messages(connection)

    async def _demote_legacy_fts_to_trash(self) -> int:
        """Demote legacy inline FTS tables and seed a resumable rebuild."""

        async def _stage(connection):
            await self._drop_fts_triggers(connection)
            cursor = await connection.execute(
                "DROP VIEW IF EXISTS messages_fts_trigram_src"
            )
            await cursor.close()
            async with connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('messages_fts', 'messages_fts_trigram') "
                "AND sql LIKE 'CREATE VIRTUAL TABLE%' LIMIT 1"
            ) as cursor:
                had_row = await cursor.fetchone()
            if had_row is not None:
                cursor = await connection.execute("PRAGMA writable_schema=ON")
                await cursor.close()
                cursor = await connection.execute(
                    "DELETE FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('messages_fts', 'messages_fts_trigram') "
                    "AND sql LIKE 'CREATE VIRTUAL TABLE%'"
                )
                await cursor.close()
                cursor = await connection.execute("PRAGMA writable_schema=RESET")
                await cursor.close()
                cursor = await connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND (name LIKE 'messages_fts_%' ESCAPE '\\' "
                    "OR name LIKE 'messages_fts_trigram_%' ESCAPE '\\')"
                )
                try:
                    shadows = [row[0] for row in await cursor.fetchall()]
                finally:
                    await cursor.close()
                for shadow in shadows:
                    cursor = await connection.execute(
                        f"ALTER TABLE {shadow} "
                        f"RENAME TO fts_v22_trash_{shadow}"
                    )
                    await cursor.close()
            high_water = await self._seed_fts_rebuild_markers(
                connection, force=True
            )
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key = 'fts_optimize_available'"
            )
            await cursor.close()
            return high_water

        high_water = int(await self._execute_write(_stage))
        async with self._get_write_lock():
            connection = await self._get_connection()
            base_ok = await self._ensure_fts_schema(
                connection, "messages_fts", FTS_SQL
            )
            trigram_ok = await self._ensure_fts_schema(
                connection, "messages_fts_trigram", FTS_TRIGRAM_SQL
            )
            self._trigram_available = bool(trigram_ok)
            if not base_ok:
                raise sqlite3.OperationalError(
                    "failed to create v23 messages_fts during "
                    "optimize-storage demote"
                )
            await connection.commit()
        return high_water

    async def optimize_fts_storage(
        self,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
        vacuum: bool = True,
    ) -> dict[str, Any]:
        """Migrate legacy FTS storage and finish all resumable backfills."""
        connection = await self._get_connection()
        if not self._fts_enabled:
            return {"ok": False, "reason": "fts5_unavailable"}
        if self.read_only:
            return {"ok": False, "reason": "read_only"}

        await self._repair_optimize_bookkeeping()
        async with self._get_write_lock():
            legacy = await self._db_has_legacy_inline_fts(connection)
        pending = await self.get_meta("fts_rebuild_high_water") is not None
        if legacy and not pending:
            await self._demote_legacy_fts_to_trash()
        elif pending and not legacy:
            async with self._get_write_lock():
                base_ok = await self._ensure_fts_schema(
                    connection, "messages_fts", FTS_SQL
                )
                trigram_ok = await self._ensure_fts_schema(
                    connection, "messages_fts_trigram", FTS_TRIGRAM_SQL
                )
                self._trigram_available = bool(trigram_ok)
                if not base_ok:
                    raise sqlite3.OperationalError(
                        "failed to re-create v23 messages_fts "
                        "on optimize-storage resume"
                    )
                await connection.commit()

        await self._fts_cjk_reset_if_stale()
        if self._fts_cjk_loaded:
            async with self._get_write_lock():
                await self._ensure_fts_cjk_schema(connection)
                await connection.commit()

        async def _emit(phase: str) -> None:
            if progress_cb is None:
                return
            status = await self.fts_rebuild_status()
            if status is None:
                status = await self.fts_cjk_rebuild_status()
            progress_cb(
                {
                    "phase": phase,
                    "percent": status["percent"] if status else 100,
                    "indexed": status["indexed"] if status else 0,
                    "total": status["total"] if status else 0,
                }
            )

        async def _pause(chunk_seconds: float) -> None:
            await asyncio.sleep(
                max(
                    self._FTS_REBUILD_MIN_PAUSE,
                    chunk_seconds * self._FTS_REBUILD_DUTY_FACTOR,
                )
            )

        await _emit("backfill")
        while True:
            started = time.monotonic()
            if not await self.fts_rebuild_step():
                break
            await _emit("backfill")
            await _pause(time.monotonic() - started)
        await _emit("backfill")

        while True:
            started = time.monotonic()
            if not await self.fts_cjk_rebuild_step():
                break
            await _emit("backfill")
            await _pause(time.monotonic() - started)

        await _emit("teardown")
        while True:
            started = time.monotonic()
            if not await self._fts_teardown_trash_step():
                break
            await _emit("teardown")
            await _pause(time.monotonic() - started)

        async with self._get_write_lock():
            async with connection.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ) as cursor:
                still_pending = await cursor.fetchone()
            still_trash = await self._has_fts_trash(connection)
            empty_index = await self._fts_external_index_empty_with_messages(
                connection
            )
        if still_pending is not None or still_trash or empty_index:
            reason = (
                "backfill_incomplete"
                if still_pending is not None or empty_index
                else "teardown_incomplete"
            )
            logger.warning(
                "FTS storage optimization did not settle (%s): "
                "pending=%s trash=%s empty_index=%s",
                reason,
                still_pending is not None,
                still_trash,
                empty_index,
            )
            return {"ok": False, "reason": reason, "vacuumed": None}

        vacuum_ok = None
        if vacuum:
            await _emit("vacuum")
            try:
                async with self._get_write_lock():
                    cursor = await connection.execute("VACUUM")
                    await cursor.close()
                vacuum_ok = True
            except sqlite3.OperationalError as exc:
                logger.warning("VACUUM after FTS optimize failed: %s", exc)
                vacuum_ok = False
            try:
                async with self._get_write_lock():
                    cursor = await connection.execute(
                        "PRAGMA wal_checkpoint(TRUNCATE)"
                    )
                    await cursor.close()
            except Exception as exc:
                logger.debug(
                    "WAL checkpoint (TRUNCATE) after optimize VACUUM "
                    "failed: %s",
                    exc,
                )

        async def _settle(connection):
            async with connection.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ) as cursor:
                pending_row = await cursor.fetchone()
            if pending_row is not None:
                return "backfill_incomplete"
            if await self._has_fts_trash(connection):
                return "teardown_incomplete"
            if await self._fts_external_index_empty_with_messages(connection):
                return "backfill_incomplete"
            cursor = await connection.execute(
                "INSERT INTO state_meta (key, value) VALUES "
                "('fts_storage_version', ?) ON CONFLICT(key) "
                "DO UPDATE SET value = excluded.value",
                (str(FTS_STORAGE_VERSION),),
            )
            await cursor.close()
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key = 'fts_optimize_available'"
            )
            await cursor.close()
            cursor = await connection.execute(
                "UPDATE schema_version SET version = ? WHERE version < ?",
                (SCHEMA_VERSION, SCHEMA_VERSION),
            )
            await cursor.close()
            return None

        refusal = await self._execute_write(_settle)
        if refusal is not None:
            logger.warning("FTS storage optimization settle refused (%s)", refusal)
            return {"ok": False, "reason": refusal, "vacuumed": vacuum_ok}
        await _emit("done")
        logger.info(
            "FTS storage optimization complete (layout v%d).",
            FTS_STORAGE_VERSION,
        )
        return {"ok": True, "vacuumed": vacuum_ok}

    async def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: tuple[str, ...] | None = ("user", "assistant"),
    ) -> dict[str, Any]:
        """Return an anchored window and bookends using only aiosqlite."""
        primitive = await self.get_messages_around(
            session_id, around_message_id, window=window
        )
        rows = primitive["window"]
        if not rows:
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }
        if keep_roles is None:
            filtered = rows
        else:
            allowed = set(keep_roles)
            filtered = [
                row for row in rows
                if row.get("id") == around_message_id or row.get("role") in allowed
            ]
        bookend = max(0, int(bookend))
        starts: list[dict[str, Any]] = []
        ends: list[dict[str, Any]] = []
        if bookend:
            role_sql = ""
            role_params: list[Any] = []
            if keep_roles is not None:
                role_sql = " AND role IN (" + ",".join("?" for _ in keep_roles) + ")"
                role_params = list(keep_roles)
            async with self._read_ctx() as connection:
                async with connection.execute(
                    "SELECT * FROM messages "
                    "WHERE session_id = ? AND id < ?"
                    f"{role_sql} AND length(content) > 0 "
                    "ORDER BY id ASC LIMIT ?",
                    (session_id, rows[0]["id"], *role_params, bookend),
                ) as cursor:
                    start_rows = await cursor.fetchall()
                async with connection.execute(
                    "SELECT * FROM messages "
                    "WHERE session_id = ? AND id > ?"
                    f"{role_sql} AND length(content) > 0 "
                    "ORDER BY id DESC LIMIT ?",
                    (session_id, rows[-1]["id"], *role_params, bookend),
                ) as cursor:
                    end_rows = await cursor.fetchall()
            starts = [self._decode_message_row(row) for row in start_rows]
            ends = [self._decode_message_row(row) for row in reversed(end_rows)]
        return {
            "window": filtered,
            "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": starts,
            "bookend_end": ends,
        }

    async def list_recent_user_messages(
        self,
        session_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the most-recent real user turns, newest first."""
        active_clause = "" if include_inactive else " AND active = 1"
        connection = await self._get_connection()
        cursor = await connection.execute(
            "SELECT id, timestamp, content FROM messages "
            "WHERE session_id = ? AND role = 'user'"
            f"{active_clause} AND (display_kind IS NULL OR display_kind = '') "
            "ORDER BY id DESC LIMIT ?",
            (session_id, int(limit)),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()

        result: list[dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_content(row["content"])
            if isinstance(decoded, list):
                text_parts = [
                    part.get("text", "")
                    for part in decoded
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                preview = " ".join(text for text in text_parts if text).strip()
                if not preview:
                    preview = "[multimodal content]"
            elif isinstance(decoded, str):
                preview = describe_skill_invocation(decoded) or decoded
            else:
                preview = ""
            preview = " ".join(preview.split())
            if len(preview) > 80:
                preview = preview[:77] + "..."
            result.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "preview": preview,
                }
            )
        return result

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}``, the column-filter operator ``:`` and bare
        boolean operators (``AND``, ``OR``, ``NOT``) have special meaning.
        Passing raw user input directly to MATCH can cause
        ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Cap user-controlled FTS input before any regex processing. Search
        # queries do not need to be arbitrarily large, and bounding them keeps
        # sanitizer/runtime behavior predictable under adversarial input.
        query = query[:MAX_FTS5_QUERY_CHARS]

        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders. Do this with a
        # single linear scan rather than a regex so pathological quote runs
        # cannot induce backtracking.
        _quoted_parts: list = []
        pieces: list[str] = []
        i = 0
        while i < len(query):
            ch = query[i]
            if ch != '"':
                pieces.append(ch)
                i += 1
                continue
            end = query.find('"', i + 1)
            if end == -1:
                # Unmatched quote: replace with whitespace like the old
                # sanitizer's special-char stripping step.
                pieces.append(" ")
                i += 1
                continue
            _quoted_parts.append(query[i:end + 1])
            pieces.append(f"\x00Q{len(_quoted_parts) - 1}\x00")
            i = end + 1

        sanitized = "".join(pieces)

        # Step 2: Strip remaining (unmatched) FTS5-special characters.  ``:`` is
        # FTS5's column-filter operator (``col:term``); since the FTS table has a
        # single ``content`` column, an unquoted colon query like ``TODO: fix``
        # parses as ``column:term`` and raises "no such column" — swallowed at
        # the execute site into zero results.  Strip it like the others.
        sanitized = re.sub(r'[+{}():\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted, hyphenated and underscored
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()

    @staticmethod
    def _is_cjk_codepoint(cp: int) -> bool:
        return (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0x3000 <= cp <= 0x303F
            or 0x3040 <= cp <= 0x309F
            or 0x30A0 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7AF
        )

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Return whether *text* contains Chinese, Japanese, or Korean text."""
        for character in text:
            cp = ord(character)
            if (
                0x4E00 <= cp <= 0x9FFF
                or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF
                or 0x3000 <= cp <= 0x303F
                or 0x3040 <= cp <= 0x309F
                or 0x30A0 <= cp <= 0x30FF
                or 0xAC00 <= cp <= 0xD7AF
            ):
                return True
        return False

    @classmethod
    def _count_cjk(cls, text: str) -> int:
        return sum(
            cls._is_cjk_codepoint(ord(character)) for character in text
        )

    @classmethod
    def _has_lone_cjk_run(cls, query: str) -> bool:
        run = 0
        for character in query:
            if cls._is_cjk_codepoint(ord(character)):
                run += 1
            else:
                if run == 1:
                    return True
                run = 0
        return run == 1

    @staticmethod
    def _trigram_eligible_tokens(query: str) -> bool:
        tokens = [
            token
            for token in query.strip('"').strip().split()
            if token.upper() not in {"AND", "OR", "NOT"}
        ]
        return bool(tokens) and all(len(token) >= 3 for token in tokens)

    async def _run_trigram_search(
        self,
        raw_query: str,
        *,
        table: str = "messages_fts_trigram",
        order_by_sql: str,
        include_inactive: bool,
        source_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]] | None:
        """Search one of the two upstream substring-capable FTS indexes."""
        if table not in {"messages_fts_trigram", "messages_fts_cjk"}:
            raise ValueError(f"unsupported FTS table: {table}")
        parts = [
            token
            if token.upper() in {"AND", "OR", "NOT"}
            else '"' + token.replace('"', '""') + '"'
            for token in raw_query.split()
        ]
        where = [f"{table} MATCH ?"]
        params: list[Any] = [" ".join(parts)]
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   snippet({table}, -1, '>>>', '<<<', '...', 40) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM {table}
            JOIN messages m ON m.id = {table}.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        try:
            return [
                dict(row) for row in await self._read_fetchall(sql, params)
            ]
        except sqlite3.DatabaseError as exc:
            if not await self._try_runtime_fts_rebuild(exc):
                return None
            try:
                return [
                    dict(row)
                    for row in await self._read_fetchall(sql, params)
                ]
            except sqlite3.DatabaseError:
                logger.warning(
                    "%s search still failing after in-place rebuild; "
                    "falling back to LIKE",
                    table,
                )
                return None

    async def search_messages(
        self,
        query: str,
        source_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        include_inactive: bool = False,
        fields: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Instrumented native-async message search.

        This preserves the v2026.8.3 routing and result contract while keeping
        every SQLite operation awaitable.
        """
        started = time.time()
        rows = None
        try:
            rows = await self._search_messages_impl(
                query,
                source_filter=source_filter,
                exclude_sources=exclude_sources,
                role_filter=role_filter,
                limit=limit,
                offset=offset,
                sort=sort,
                include_inactive=include_inactive,
                fields=fields,
            )
            return rows
        finally:
            try:
                threshold = float(
                    _session_runtime_config_value(
                        "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
                    )
                )
            except (TypeError, ValueError):
                threshold = 1000.0
            elapsed_ms = (time.time() - started) * 1000.0
            if elapsed_ms >= threshold:
                logger.info(
                    "slow session search: path=%s elapsed=%.0fms rows=%s query=%r",
                    self._describe_search_path(query),
                    elapsed_ms,
                    len(rows) if rows is not None else "err",
                    query[:200],
                )

    def _describe_search_path(self, query: str) -> str:
        """Return the upstream route name used for slow-search diagnostics."""
        try:
            if self._fts_stale:
                return "like_scan_fts_stale"
            sanitized = self._sanitize_fts5_query(query or "")
            if not sanitized:
                return "empty"
            if not self._contains_cjk(sanitized):
                return "fts5"
            raw = sanitized.strip('"').strip()
            if self._fts_cjk_available and not self._has_lone_cjk_run(raw):
                return "fts_cjk"
            tokens = [
                token
                for token in raw.split()
                if token.upper() not in {"AND", "OR", "NOT"}
                and self._contains_cjk(token)
            ]
            short = any(self._count_cjk(token) < 3 for token in tokens)
            if self._count_cjk(raw) >= 3 and not short and self._trigram_available:
                return "trigram"
            return "like_scan"
        except Exception:
            return "unknown"

    @staticmethod
    def _compile_like_boolean_query(
        query: str,
    ) -> tuple[str, list[Any], str | None]:
        """Compile the supported FTS boolean subset into LIKE predicates."""
        groups: list[list[tuple[str, bool]]] = [[]]
        negate_next = False
        for raw_token in re.findall(r'"[^"]+"|\S+', query):
            operator = raw_token.upper()
            if operator == "OR":
                if groups[-1]:
                    groups.append([])
                negate_next = False
                continue
            if operator in {"AND", "NEAR"}:
                continue
            if operator == "NOT":
                negate_next = True
                continue
            term = raw_token.strip('"').strip("*").strip()
            if term:
                groups[-1].append((term, negate_next))
                negate_next = False

        compiled_groups: list[str] = []
        params: list[Any] = []
        snippet_term: str | None = None
        for group in groups:
            if not group or not any(not negated for _, negated in group):
                continue
            clauses: list[str] = []
            for term, negated in group:
                escaped = _escape_like(term)
                clause = (
                    "(COALESCE(m.content, '') LIKE ? ESCAPE '\\' OR "
                    "COALESCE(m.tool_name, '') LIKE ? ESCAPE '\\' OR "
                    "COALESCE(m.tool_calls, '') LIKE ? ESCAPE '\\')"
                )
                clauses.append(f"NOT {clause}" if negated else clause)
                params.extend([f"%{escaped}%"] * 3)
                if snippet_term is None and not negated:
                    snippet_term = term
            compiled_groups.append(f"({' AND '.join(clauses)})")
        return " OR ".join(compiled_groups), params, snippet_term

    async def _search_messages_like_fallback(
        self,
        query: str,
        *,
        source_filter: list[str] | None,
        exclude_sources: list[str] | None,
        role_filter: list[str] | None,
        limit: int,
        offset: int,
        sort: str | None,
        include_inactive: bool,
    ) -> list[dict[str, Any]]:
        """Search canonical rows while durable FTS state is stale."""
        predicate, params, snippet_term = self._compile_like_boolean_query(query)
        if not predicate or snippet_term is None:
            return []
        where = [f"({predicate})"]
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        order = (
            "ASC"
            if isinstance(sort, str) and sort.strip().lower() == "oldest"
            else "DESC"
        )
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   substr(m.content, max(1, instr(m.content, ?) - 40), 120) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY m.timestamp {order}, m.id {order}
            LIMIT ? OFFSET ?
        """
        rows = await self._read_fetchall(
            sql, [snippet_term, *params, limit, offset]
        )
        return [dict(row) for row in rows]

    async def _refresh_fts_stale_state(self) -> None:
        """Observe a fail-open marker written by another process."""
        if self._fts_stale or not self._fts_enabled:
            return
        try:
            rows = await self._read_fetchall(
                "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1",
                (FTS_STALE_KEY,),
            )
        except sqlite3.Error:
            return
        if rows:
            self._fts_stale = True
            self._fts_enabled = False
            self._trigram_available = False
            self._fts_cjk_available = False

    async def _search_messages_impl(
        self,
        query: str,
        source_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        include_inactive: bool = False,
        fields: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search with v2026.8.3 routing and result semantics."""
        result_fields = self._search_message_fields(fields)
        await self._get_connection()
        if not query or not query.strip():
            return []
        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        await self._refresh_fts_stale_state()
        if self._fts_stale:
            return await self._search_messages_like_fallback(
                query,
                source_filter=source_filter,
                exclude_sources=exclude_sources,
                role_filter=role_filter,
                limit=limit,
                offset=offset,
                sort=sort,
                include_inactive=include_inactive,
            )
        if not self._fts_enabled:
            return []

        sort_norm = sort.strip().lower() if isinstance(sort, str) else None
        if sort_norm not in {"newest", "oldest"}:
            sort_norm = None
        if sort_norm == "newest":
            order_by_sql = "ORDER BY m.timestamp DESC, rank"
        elif sort_norm == "oldest":
            order_by_sql = "ORDER BY m.timestamp ASC, rank"
        else:
            order_by_sql = "ORDER BY rank"

        where = ["messages_fts MATCH ?"]
        params: list[Any] = [query]
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   snippet(messages_fts, -1, '>>>', '<<<', '...', 40) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        matches: list[dict[str, Any]] = []
        is_cjk = self._contains_cjk(query)
        used_like = False
        if is_cjk:
            raw_query = query.strip('"').strip()
            cjk_tokens = [
                token
                for token in raw_query.split()
                if token.upper() not in {"AND", "OR", "NOT"}
                and self._contains_cjk(token)
            ]
            any_short_cjk = any(
                self._count_cjk(token) < 3 for token in cjk_tokens
            )
            wants_tool_rows = bool(role_filter) and "tool" in role_filter
            substring_matches: list[dict[str, Any]] | None = None
            if (
                self._fts_cjk_available
                and not wants_tool_rows
                and not self._has_lone_cjk_run(raw_query)
            ):
                substring_matches = await self._run_trigram_search(
                    raw_query,
                    table="messages_fts_cjk",
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
            if (
                substring_matches is None
                and self._count_cjk(raw_query) >= 3
                and not any_short_cjk
                and self._trigram_available
                and not wants_tool_rows
            ):
                substring_matches = await self._run_trigram_search(
                    raw_query,
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
            if substring_matches is not None:
                matches = substring_matches
            else:
                used_like = True
                tokens = [
                    token
                    for token in raw_query.split()
                    if token.upper() not in {"AND", "OR", "NOT"}
                ] or [raw_query]
                token_clauses = []
                like_params: list[Any] = []
                for token in tokens:
                    escaped = _escape_like(token)
                    token_clauses.append(
                        "(m.content LIKE ? ESCAPE '\\' "
                        "OR m.tool_name LIKE ? ESCAPE '\\' "
                        "OR m.tool_calls LIKE ? ESCAPE '\\')"
                    )
                    like_params.extend([f"%{escaped}%"] * 3)
                like_where = ["(" + " OR ".join(token_clauses) + ")"]
                if not include_inactive:
                    like_where.append("(m.active = 1 OR m.compacted = 1)")
                if source_filter is not None:
                    like_where.append(
                        "s.source IN ("
                        + ",".join("?" for _ in source_filter)
                        + ")"
                    )
                    like_params.extend(source_filter)
                if exclude_sources is not None:
                    like_where.append(
                        "s.source NOT IN ("
                        + ",".join("?" for _ in exclude_sources)
                        + ")"
                    )
                    like_params.extend(exclude_sources)
                if role_filter:
                    like_where.append(
                        "m.role IN ("
                        + ",".join("?" for _ in role_filter)
                        + ")"
                    )
                    like_params.extend(role_filter)
                like_sql = f"""
                    SELECT m.id, m.session_id, m.role,
                           substr(m.content,
                                  max(1, instr(m.content, ?) - 40),
                                  120) AS snippet,
                           m.content, m.timestamp, m.tool_name,
                           s.source, s.model,
                           s.started_at AS session_started
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(like_where)}
                    ORDER BY m.timestamp DESC
                    LIMIT ? OFFSET ?
                """
                matches = [
                    dict(row)
                    for row in await self._read_fetchall(
                        like_sql,
                        [tokens[0], *like_params, limit, offset],
                    )
                ]
        else:
            try:
                matches = [
                    dict(row)
                    for row in await self._read_fetchall(sql, params)
                ]
            except sqlite3.DatabaseError as exc:
                if (
                    isinstance(exc, sqlite3.OperationalError)
                    and not self._is_fts_write_corruption_error(exc)
                ):
                    return []
                # A new caller cancellation must supersede this stale FTS error.
                if not await self._try_runtime_fts_rebuild(exc):  # noqa: ASYNC120
                    raise
                matches = [
                    dict(row)
                    for row in await self._read_fetchall(sql, params)
                ]

        rebuild_status = await self.fts_rebuild_status()
        if not used_like and rebuild_status is not None and len(matches) < limit:
            try:
                gap_matches = await self._search_unindexed_gap(
                    query,
                    limit - len(matches),
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                )
                seen_ids = {match["id"] for match in matches}
                matches.extend(
                    match for match in gap_matches if match["id"] not in seen_ids
                )
            except sqlite3.OperationalError as exc:
                logger.debug("Unindexed-gap supplement skipped: %s", exc)

        if (
            not matches
            and not is_cjk
            and not (bool(role_filter) and "tool" in role_filter)
        ):
            fallback_query = query.strip('"').strip()
            if self._fts_cjk_available:
                cjk_matches = await self._run_trigram_search(
                    fallback_query,
                    table="messages_fts_cjk",
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
                if cjk_matches:
                    matches = cjk_matches
            if (
                not matches
                and self._trigram_available
                and self._trigram_eligible_tokens(query)
            ):
                trigram_matches = await self._run_trigram_search(
                    fallback_query,
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
                if trigram_matches:
                    matches = trigram_matches

        context_matches = (
            matches if result_fields is None or "context" in result_fields else ()
        )
        for match in context_matches:
            try:
                context_rows = await self._read_fetchall(
                    """WITH target AS (
                           SELECT session_id, timestamp, id
                           FROM messages WHERE id = ?
                       )
                       SELECT role, content FROM (
                           SELECT m.id, m.timestamp, m.role, m.content
                           FROM messages m JOIN target t
                             ON t.session_id = m.session_id
                           WHERE m.timestamp < t.timestamp
                              OR (m.timestamp = t.timestamp AND m.id < t.id)
                           ORDER BY m.timestamp DESC, m.id DESC LIMIT 1
                       )
                       UNION ALL
                       SELECT role, content FROM messages WHERE id = ?
                       UNION ALL
                       SELECT role, content FROM (
                           SELECT m.id, m.timestamp, m.role, m.content
                           FROM messages m JOIN target t
                             ON t.session_id = m.session_id
                           WHERE m.timestamp > t.timestamp
                              OR (m.timestamp = t.timestamp AND m.id > t.id)
                           ORDER BY m.timestamp ASC, m.id ASC LIMIT 1
                       )""",
                    (match["id"], match["id"]),
                )
                context = []
                for row in context_rows:
                    decoded = self._decode_content(row["content"])
                    if isinstance(decoded, list):
                        texts = [
                            part.get("text", "")
                            for part in decoded
                            if isinstance(part, dict)
                            and part.get("type") == "text"
                        ]
                        preview = " ".join(
                            text for text in texts if text
                        ).strip()
                        preview = preview or "[multimodal content]"
                    elif isinstance(decoded, str):
                        preview = decoded
                    else:
                        preview = ""
                    context.append(
                        {"role": row["role"], "content": preview[:200]}
                    )
                match["context"] = context
            except Exception:
                match["context"] = []

        for match in matches:
            match.pop("content", None)
        if result_fields is not None:
            matches = [
                {field: match[field] for field in result_fields if field in match}
                for match in matches
            ]
        return matches

    async def _search_unindexed_gap(
        self,
        fts_query: str,
        limit: int,
        *,
        include_inactive: bool = False,
        source_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """LIKE-scan only the deferred FTS rebuild's unindexed id range."""
        status = await self.fts_rebuild_status()
        if status is None or limit <= 0:
            return []
        terms = []
        for raw_token in re.findall(r'"[^"]+"|\S+', fts_query):
            token = raw_token.strip('"').strip("*").strip()
            if token and token.upper() not in {"AND", "OR", "NOT", "NEAR"}:
                terms.append(token)
        if not terms:
            return []
        where = ["m.id > ? AND m.id <= ?"]
        params: list[Any] = [status["indexed"], status["total"]]
        for term in terms:
            escaped = _escape_like(term)
            where.append(
                "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' "
                "OR m.tool_calls LIKE ? ESCAPE '\\')"
            )
            params.extend([f"%{escaped}%"] * 3)
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   substr(m.content,
                          max(1, instr(m.content, ?) - 40),
                          120) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY m.timestamp DESC
            LIMIT ?
        """
        return [
            dict(row)
            for row in await self._read_fetchall(
                sql, [terms[0], *params, limit]
            )
        ]

    async def search_sessions_by_id(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = True,
        source: str | None = None,
        sources: list[str] | None = None,
        exclude_sources: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search surfaced sessions by exact, prefix, or substring id."""
        needle = (query or "").strip().lower()
        if not needle or limit <= 0:
            return []
        candidates = await self.list_sessions_rich(
            source=source,
            sources=sources,
            exclude_sources=exclude_sources,
            limit=max(limit * 4, limit),
            offset=0,
            include_archived=include_archived,
            order_by_last_active=True,
            id_query=needle,
        )

        def score(row: dict[str, Any]) -> int:
            ids = [
                str(row.get("id") or ""),
                str(row.get("_lineage_root_id") or ""),
            ]
            normalized = [value.lower() for value in ids if value]
            if any(value == needle for value in normalized):
                return 0
            if any(value.startswith(needle) for value in normalized):
                return 1
            return 2

        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (score(item[1]), item[0]),
        )
        return [row for _, row in ranked[:limit]]

    async def _fts_table_exists(self, name: str) -> bool:
        """Return whether an FTS5 virtual table is queryable."""
        connection = await self._get_connection()
        try:
            cursor = await connection.execute(f"SELECT 1 FROM {name} LIMIT 0")
            await cursor.close()
            return True
        except sqlite3.DatabaseError:
            return False

    async def optimize_fts(self) -> int:
        """Merge fragmented FTS5 b-tree segments into one per index."""
        optimized = 0
        async with self._get_write_lock():
            connection = await self._get_connection()
            for table_name in self._FTS_TABLES:
                if not await self._fts_table_exists(table_name):
                    continue
                try:
                    cursor = await connection.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('optimize')"
                    )
                    await cursor.close()
                    optimized += 1
                except sqlite3.OperationalError as exc:
                    logger.warning(
                        "FTS optimize failed for %s: %s", table_name, exc
                    )
        return optimized

    async def rebuild_fts(self) -> int:
        """Rebuild present FTS5 indexes from their canonical content tables."""
        rebuilt = 0
        async with self._get_write_lock():
            connection = await self._get_connection()
            for table_name in self._FTS_TABLES:
                if not await self._fts_table_exists(table_name):
                    continue
                try:
                    cursor = await connection.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                    )
                    await cursor.close()
                    await connection.commit()
                    rebuilt += 1
                except sqlite3.OperationalError as exc:
                    await connection.rollback()
                    logger.warning("FTS rebuild failed for %s: %s", table_name, exc)
        return rebuilt

    async def _merge_fts_incrementally(
        self,
        *,
        max_pages: int,
        max_commands: int | None = None,
    ) -> int:
        """Run upstream's bounded FTS5 ``'merge'`` protocol natively."""
        if isinstance(max_pages, bool) or not isinstance(max_pages, int):
            raise TypeError("max_pages must be an integer")
        if max_pages <= 0:
            raise ValueError("max_pages must be greater than zero")
        if max_commands is None:
            max_commands = self._FTS_MERGE_COMMANDS_PER_PASS
        if isinstance(max_commands, bool) or not isinstance(max_commands, int):
            raise TypeError("max_commands must be an integer")
        if max_commands <= 0:
            raise ValueError("max_commands must be greater than zero")

        executed = 0
        async with self._get_write_lock():
            connection = await self._get_connection()
            for table_name in self._FTS_TABLES:
                if not await self._fts_table_exists(table_name):
                    continue
                if not self._fts_usermerge_floor_applied:
                    cursor = await connection.execute(
                        f"INSERT INTO {table_name}({table_name}, rank) "
                        "VALUES('usermerge', 2)"
                    )
                    await cursor.close()
                for _ in range(max_commands):
                    before = connection.total_changes
                    cursor = await connection.execute(
                        f"INSERT INTO {table_name}({table_name}, rank) "
                        "VALUES('merge', ?)",
                        (max_pages,),
                    )
                    await cursor.close()
                    executed += 1
                    if connection.total_changes - before < 2:
                        break
            self._fts_usermerge_floor_applied = True
        return executed
