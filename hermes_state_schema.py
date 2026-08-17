"""Schema migration helpers for the native-async ``SessionDB``.

This restores the upstream schema-mixin path for the retained library. Gateway
routing-PK repair and the ``sessions.json`` gateway backfill remain outside the
reduced package. This module must not import ``hermes_state``.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import aiosqlite

from hermes_state_common import (
    DEFERRED_INDEX_SQL,
    FTS_CJK_STALE_KEY,
    FTS_STALE_KEY,
    FTS_SQL,
    FTS_STORAGE_VERSION,
    FTS_TRIGRAM_SQL,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    _FTS_CJK_TRIGGERS,
    _FTS_TRIGGERS,
    _ephemeral_child_sql,
)


logger = logging.getLogger("hermes_state")


async def _rebuild_corrupt_fts_schema(conn: aiosqlite.Connection) -> None:
    """Replace corrupt FTS catalog entries using canonical ``messages`` rows.

    SQLite can refuse to drop a damaged FTS5 virtual table because constructing
    the table itself fails.  The shadow tables are derived state, so a bounded
    ``writable_schema`` surgery is safe here: only ``messages_fts*`` catalog
    rows are removed, then the retained legacy or external-content schema is
    recreated and backfilled.  Canonical session/message tables are untouched.
    """
    base_cursor = await conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'messages_fts'"
    )
    try:
        base_row = await base_cursor.fetchone()
    finally:
        await base_cursor.close()
    trigram_cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'messages_fts_trigram'"
    )
    try:
        include_trigram = await trigram_cursor.fetchone() is not None
    finally:
        await trigram_cursor.close()
    if base_row is None:
        raise sqlite3.DatabaseError("messages_fts schema is missing")
    base_sql = str(base_row[0] or "")
    legacy = "content='messages'" not in "".join(base_sql.split()).lower()

    await conn.execute("PRAGMA writable_schema=ON")
    try:
        await conn.execute(
            "DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'"
        )
        await conn.execute("PRAGMA writable_schema=OFF")
        await conn.commit()
    except BaseException:
        try:
            await conn.execute("PRAGMA writable_schema=OFF")
        except sqlite3.Error:
            pass
        await conn.rollback()
        raise
    await conn.execute("VACUUM")

    if legacy:
        rebuild_sql = LEGACY_FTS_SQL
        if include_trigram:
            rebuild_sql += LEGACY_FTS_TRIGRAM_SQL
        rebuild_sql += """
            INSERT INTO messages_fts(rowid, content)
            SELECT id,
                   COALESCE(content, '') || ' ' ||
                   COALESCE(tool_name, '') || ' ' ||
                   COALESCE(tool_calls, '')
            FROM messages;
        """
        if include_trigram:
            rebuild_sql += """
                INSERT INTO messages_fts_trigram(rowid, content)
                SELECT id,
                       COALESCE(content, '') || ' ' ||
                       COALESCE(tool_name, '') || ' ' ||
                       COALESCE(tool_calls, '')
                FROM messages;
            """
    else:
        rebuild_sql = FTS_SQL
        if include_trigram:
            rebuild_sql += FTS_TRIGRAM_SQL
        rebuild_sql += "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
        if include_trigram:
            rebuild_sql += (
                "INSERT INTO messages_fts_trigram(messages_fts_trigram) "
                "VALUES('rebuild');"
            )
    await conn.executescript(
        "BEGIN IMMEDIATE;"
        + rebuild_sql
        + f"DELETE FROM state_meta WHERE key = '{FTS_STALE_KEY}';"
        + "COMMIT;"
    )


class SessionSchemaMixin:
    """Native-async retained schema migrations for ``SessionDB``."""

    async def _dedupe_legacy_system_prompts(self, cursor: Any) -> None:
        """Move inline prompt snapshots into the shared addressed table."""
        try:
            async with cursor.execute(
                "SELECT id, system_prompt FROM sessions "
                "WHERE system_prompt IS NOT NULL"
            ) as query_cursor:
                rows = await query_cursor.fetchall()
        except sqlite3.OperationalError:
            return

        for row in rows:
            session_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
            prompt = (
                row["system_prompt"] if isinstance(row, sqlite3.Row) else row[1]
            )
            prompt_hash = await self._store_system_prompt(  # type: ignore[unresolved-attribute]
                cursor,
                prompt,
            )
            update_cursor = await cursor.execute(
                "UPDATE sessions "
                "SET system_prompt_hash = ?, system_prompt = NULL "
                "WHERE id = ?",
                (prompt_hash, session_id),
            )
            await update_cursor.close()

    async def _sqlite_supports_fts5(self, cursor: Any) -> bool:
        try:
            probe_cursor = await cursor.execute(
                "CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)"
            )
            await probe_cursor.close()
            drop_cursor = await cursor.execute(
                "DROP TABLE temp._hermes_fts5_probe"
            )
            await drop_cursor.close()
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(  # type: ignore[unresolved-attribute]
                exc
            ):
                raise
            self._warn_fts5_unavailable(exc)  # type: ignore[unresolved-attribute]
            return False

    @staticmethod
    async def _fts_trigger_count(cursor: Any) -> int:
        placeholders = ",".join("?" for _ in _FTS_TRIGGERS)
        async with cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            _FTS_TRIGGERS,
        ) as query_cursor:
            row = await query_cursor.fetchone()
        return int(row[0])

    @staticmethod
    def _fts_update_trigger_needs_narrowing(sql: str | None) -> bool:
        """True when trigger SQL is missing AFTER UPDATE OF (still broad)."""
        if not sql:
            return False
        compact = " ".join(sql.split()).upper()
        if "AFTER UPDATE OF " in compact:
            return False
        return "AFTER UPDATE ON " in compact

    async def _cjk_update_trigger_is_narrowed(self, cursor: Any) -> bool:
        """True when messages_fts_cjk_update exists with AFTER UPDATE OF."""
        async with cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name = ?",
            ("messages_fts_cjk_update",),
        ) as query_cursor:
            row = await query_cursor.fetchone()
        if not row:
            return False
        sql = row["sql"] if isinstance(row, sqlite3.Row) else row[0]
        return not self._fts_update_trigger_needs_narrowing(sql)

    async def _quarantine_cjk_after_update_of_migration(
        self,
        cursor: Any,
    ) -> None:
        """Fail closed after dropping CJK UPDATE during OF migration."""
        self._fts_cjk_available = False
        try:
            await self.set_meta(  # type: ignore[unresolved-attribute]
                FTS_CJK_STALE_KEY,
                "1",
                cursor=cursor,
            )
        except Exception:
            logger.debug(
                "Could not persist CJK FTS stale breadcrumb",
                exc_info=True,
            )
        try:
            drop_cursor = await cursor.execute(
                "DROP TRIGGER IF EXISTS messages_fts_cjk_update"
            )
            await drop_cursor.close()
        except Exception:
            logger.debug(
                "Could not drop residual CJK UPDATE trigger after quarantine",
                exc_info=True,
            )

    async def _migrate_broad_fts_update_triggers(self, cursor: Any) -> int:
        """Replace broad AFTER UPDATE FTS triggers with AFTER UPDATE OF variants.

        ``CREATE TRIGGER IF NOT EXISTS`` cannot replace triggers installed by
        older Hermes versions. The migration only drops names from a fixed
        allowlist, recreates the matching FTS layout, and never rebuilds data.
        """
        legacy_layout = await self._db_has_legacy_inline_fts(  # type: ignore[unresolved-attribute]
            cursor
        )
        update_names = (
            "messages_fts_update",
            "messages_fts_trigram_update",
        )
        if not legacy_layout and hasattr(self, "_ensure_fts_cjk_schema"):
            update_names += ("messages_fts_cjk_update",)
        placeholders = ", ".join("?" for _ in update_names)
        async with cursor.execute(
            "SELECT name, sql FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            update_names,
        ) as query_cursor:
            rows = await query_cursor.fetchall()
        to_drop = []
        for row in rows:
            name = row["name"] if isinstance(row, sqlite3.Row) else row[0]
            sql = row["sql"] if isinstance(row, sqlite3.Row) else row[1]
            if self._fts_update_trigger_needs_narrowing(sql):
                to_drop.append(name)
        if not to_drop:
            return 0

        for name in to_drop:
            drop_cursor = await cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
            await drop_cursor.close()

        if legacy_layout:
            await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                cursor,
                "messages_fts",
                LEGACY_FTS_SQL,
            )
            await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                cursor,
                "messages_fts_trigram",
                LEGACY_FTS_TRIGRAM_SQL,
            )
        else:
            await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                cursor,
                "messages_fts",
                FTS_SQL,
            )
            await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                cursor,
                "messages_fts_trigram",
                FTS_TRIGRAM_SQL,
            )
            if "messages_fts_cjk_update" in to_drop:
                ensure_error: Exception | None = None
                try:
                    await self._ensure_fts_cjk_schema(  # type: ignore[unresolved-attribute]
                        cursor
                    )
                except Exception as exc:
                    ensure_error = exc
                if ensure_error is not None:
                    await self._quarantine_cjk_after_update_of_migration(cursor)
                    logger.error(
                        "CJK FTS re-ensure after UPDATE OF migration failed",
                        exc_info=(
                            type(ensure_error),
                            ensure_error,
                            ensure_error.__traceback__,
                        ),
                    )
                    raise ensure_error
                if not await self._cjk_update_trigger_is_narrowed(cursor):
                    await self._quarantine_cjk_after_update_of_migration(cursor)
                    logger.warning(
                        "CJK FTS UPDATE trigger missing or still broad after "
                        "UPDATE OF migration; marked stale and unavailable"
                    )

        logger.info(
            "Migrated %d broad FTS UPDATE trigger(s) to AFTER UPDATE OF "
            "(no rebuild required)",
            len(to_drop),
        )
        return len(to_drop)

    @staticmethod
    async def _rebuild_fts_indexes(
        cursor: Any,
        *,
        include_trigram: bool = True,
    ) -> None:
        rebuild_cursor = await cursor.execute(
            "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
        )
        await rebuild_cursor.close()
        if include_trigram:
            trigram_cursor = await cursor.execute(
                "INSERT INTO messages_fts_trigram(messages_fts_trigram) "
                "VALUES('rebuild')"
            )
            await trigram_cursor.close()
        marker_cursor = await cursor.execute(
            "DELETE FROM state_meta WHERE key IN "
            "('fts_rebuild_high_water', 'fts_rebuild_progress')"
        )
        await marker_cursor.close()

    @staticmethod
    async def _rebuild_legacy_fts_indexes(
        cursor: Any,
        *,
        include_trigram: bool = True,
    ) -> None:
        delete_cursor = await cursor.execute("DELETE FROM messages_fts")
        await delete_cursor.close()
        insert_cursor = await cursor.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') "
            "FROM messages"
        )
        await insert_cursor.close()
        if not include_trigram:
            return
        delete_trigram_cursor = await cursor.execute(
            "DELETE FROM messages_fts_trigram"
        )
        await delete_trigram_cursor.close()
        insert_trigram_cursor = await cursor.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) "
            "SELECT id, COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') "
            "FROM messages"
        )
        await insert_trigram_cursor.close()

    async def _fts_table_probe(
        self,
        cursor: Any,
        table_name: str,
    ) -> bool | None:
        try:
            probe_cursor = await cursor.execute(
                f"SELECT * FROM {table_name} LIMIT 0"
            )
            await probe_cursor.close()
            return True
        except sqlite3.OperationalError as exc:
            if self._is_fts5_unavailable_error(  # type: ignore[unresolved-attribute]
                exc
            ):
                if self._is_trigram_unavailable_error(  # type: ignore[unresolved-attribute]
                    exc
                ):
                    self._warn_trigram_unavailable(  # type: ignore[unresolved-attribute]
                        exc
                    )
                else:
                    self._warn_fts5_unavailable(  # type: ignore[unresolved-attribute]
                        exc
                    )
                return None
            if "no such table" in str(exc).lower():
                return False
            raise

    async def _recover_stale_fts(self, cursor: Any, *, legacy: bool) -> bool:
        """Atomically rebuild stale FTS indexes and restore their triggers."""
        try:
            trigram_status = await self._fts_table_probe(
                cursor, "messages_fts_trigram"
            )
        except sqlite3.DatabaseError:
            # A corrupt virtual table may fail even a LIMIT 0 probe. Include
            # it in the drop-and-recreate recovery in that case.
            trigram_status = True
        include_trigram = trigram_status is True

        drop_sql = "".join(
            f"DROP TRIGGER IF EXISTS {trigger};" for trigger in _FTS_TRIGGERS
        )
        drop_sql += "".join(
            f"DROP TRIGGER IF EXISTS {trigger};"
            for trigger in _FTS_CJK_TRIGGERS
        )
        if include_trigram:
            drop_sql += "DROP TABLE IF EXISTS messages_fts_trigram;"
        drop_sql += (
            "DROP VIEW IF EXISTS messages_fts_trigram_src;"
            "DROP TABLE IF EXISTS messages_fts;"
        )

        if legacy:
            schema_sql = LEGACY_FTS_SQL
            if include_trigram:
                schema_sql += LEGACY_FTS_TRIGRAM_SQL
            rebuild_sql = schema_sql + """
                INSERT INTO messages_fts(rowid, content)
                SELECT id,
                       COALESCE(content, '') || ' ' ||
                       COALESCE(tool_name, '') || ' ' ||
                       COALESCE(tool_calls, '')
                FROM messages;
            """
            if include_trigram:
                rebuild_sql += """
                    DELETE FROM messages_fts_trigram;
                    INSERT INTO messages_fts_trigram(rowid, content)
                    SELECT id,
                           COALESCE(content, '') || ' ' ||
                           COALESCE(tool_name, '') || ' ' ||
                           COALESCE(tool_calls, '')
                    FROM messages;
                """
        else:
            schema_sql = FTS_SQL
            if include_trigram:
                schema_sql += FTS_TRIGRAM_SQL
            rebuild_sql = schema_sql + (
                "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
            )
            if include_trigram:
                rebuild_sql += (
                    "INSERT INTO messages_fts_trigram(messages_fts_trigram) "
                    "VALUES('rebuild');"
                )
            rebuild_sql += (
                "DELETE FROM state_meta WHERE key IN "
                "('fts_rebuild_high_water', 'fts_rebuild_progress');"
            )

        recovery_sql = (
            "BEGIN IMMEDIATE;"
            + drop_sql
            + rebuild_sql
            + f"DELETE FROM state_meta WHERE key = '{FTS_STALE_KEY}';"
            + "COMMIT;"
        )
        try:
            await cursor.executescript(recovery_sql)
        except sqlite3.DatabaseError as exc:
            try:
                await cursor.rollback()
            except sqlite3.Error:
                pass
            try:
                await self._drop_all_fts_triggers(cursor)  # type: ignore[attr-defined]
                await cursor.commit()
            except sqlite3.Error:
                pass
            # Some SQLite builds reject DROP TABLE for a corrupt FTS5 virtual
            # table (``vtable constructor failed``).  The canonical
            # ``messages`` table is still healthy, so remove only the derived
            # FTS catalog entries and rebuild them from canonical rows.  This
            # is the same bounded recovery used by the offline repair path;
            # it must not touch user/session tables.
            try:
                await _rebuild_corrupt_fts_schema(cursor)
            except sqlite3.DatabaseError as fallback_exc:
                try:
                    await cursor.execute("PRAGMA writable_schema=OFF")
                except sqlite3.Error:
                    pass
                try:
                    await cursor.rollback()
                except sqlite3.Error:
                    pass
                try:
                    await self._drop_all_fts_triggers(cursor)  # type: ignore[attr-defined]
                    await cursor.commit()
                except sqlite3.Error:
                    pass
                logger.error(
                    "Automatic rebuild of stale FTS indexes failed (%s; "
                    "catalog recovery: %s); canonical writes remain enabled "
                    "with FTS detached.",
                    exc,
                    fallback_exc,
                )
                return False

        self._fts_stale = False
        self._fts_enabled = True
        self._trigram_available = include_trigram
        logger.warning(
            "Rebuilt stale state.db FTS indexes from canonical messages and "
            "restored async sync triggers."
        )
        return True

    @staticmethod
    async def _parse_schema_columns(
        schema_sql: str,
    ) -> dict[str, dict[str, str]]:
        """Extract expected columns using SQLite itself as the DDL parser."""
        reference = await aiosqlite.connect(":memory:")
        try:
            script_cursor = await reference.executescript(schema_sql)
            await script_cursor.close()
            async with reference.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ) as table_cursor:
                tables = await table_cursor.fetchall()
            table_columns: dict[str, dict[str, str]] = {}
            for (table_name,) in tables:
                async with reference.execute(
                    f'PRAGMA table_info("{table_name}")'
                ) as column_cursor:
                    rows = await column_cursor.fetchall()
                columns: dict[str, str] = {}
                for row in rows:
                    column_name = row[1]
                    column_type = row[2] or ""
                    not_null = row[3]
                    default = row[4]
                    primary_key = row[5]
                    parts = [column_type] if column_type else []
                    if not_null and not primary_key:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    columns[column_name] = " ".join(parts)
                table_columns[table_name] = columns
            return table_columns
        finally:
            await reference.close()

    async def _reconcile_columns(self, cursor: Any) -> None:
        """Ensure live tables have every column declared in ``SCHEMA_SQL``."""
        expected = await self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_columns in expected.items():
            try:
                async with cursor.execute(
                    f'PRAGMA table_info("{table_name}")'
                ) as pragma_cursor:
                    rows = await pragma_cursor.fetchall()
            except sqlite3.OperationalError:
                continue
            live_columns = {
                row["name"] if isinstance(row, sqlite3.Row) else row[1]
                for row in rows
            }
            for column_name, column_type in declared_columns.items():
                if column_name in live_columns:
                    continue
                safe_name = column_name.replace('"', '""')
                try:
                    alter_cursor = await cursor.execute(
                        f'ALTER TABLE "{table_name}" '
                        f'ADD COLUMN "{safe_name}" {column_type}'
                    )
                    await alter_cursor.close()
                except sqlite3.OperationalError as exc:
                    message = str(exc).lower()
                    if "duplicate column" in message:
                        logger.debug(
                            "reconcile %s.%s: %s",
                            table_name,
                            column_name,
                            exc,
                        )
                        continue
                    if "locked" in message or "busy" in message:
                        raise
                    logger.warning(
                        "reconcile %s.%s failed; store remains behind "
                        "SCHEMA_SQL: %s",
                        table_name,
                        column_name,
                        exc,
                    )

    async def _heal_session_model_usage_pk(self, cursor: Any) -> None:
        """Rebuild legacy usage tables whose composite key omits ``task``."""
        async with cursor.execute(
            'PRAGMA table_info("session_model_usage")'
        ) as pragma_cursor:
            rows = await pragma_cursor.fetchall()
        if not rows:
            return
        pk_columns = {row["name"] for row in rows if row["pk"]}
        if "task" in pk_columns:
            return

        logger.info(
            "session_model_usage has legacy primary key %r; rebuilding",
            sorted(pk_columns),
        )
        foreign_keys_cursor = await cursor.execute("PRAGMA foreign_keys=OFF")
        await foreign_keys_cursor.close()
        try:
            begin_cursor = await cursor.execute("BEGIN IMMEDIATE")
            await begin_cursor.close()
            rename_cursor = await cursor.execute(
                "ALTER TABLE session_model_usage "
                "RENAME TO session_model_usage_legacy_pk"
            )
            await rename_cursor.close()
            create_cursor = await cursor.execute(
                """CREATE TABLE session_model_usage (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    billing_provider TEXT NOT NULL DEFAULT '',
                    billing_base_url TEXT NOT NULL DEFAULT '',
                    billing_mode TEXT NOT NULL DEFAULT '',
                    task TEXT NOT NULL DEFAULT '',
                    api_call_count INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    actual_cost_usd REAL NOT NULL DEFAULT 0,
                    cost_status TEXT,
                    cost_source TEXT,
                    first_seen REAL,
                    last_seen REAL,
                    PRIMARY KEY (
                        session_id, model, billing_provider,
                        billing_base_url, billing_mode, task
                    )
                )"""
            )
            await create_cursor.close()
            copy_cursor = await cursor.execute(
                """INSERT OR IGNORE INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   )
                   SELECT session_id, model,
                          COALESCE(billing_provider, ''),
                          COALESCE(billing_base_url, ''),
                          COALESCE(billing_mode, ''), COALESCE(task, ''),
                          api_call_count, input_tokens, output_tokens,
                          cache_read_tokens, cache_write_tokens,
                          reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                          cost_status, cost_source, first_seen, last_seen
                   FROM session_model_usage_legacy_pk"""
            )
            await copy_cursor.close()
            drop_cursor = await cursor.execute(
                "DROP TABLE session_model_usage_legacy_pk"
            )
            await drop_cursor.close()
            session_index_cursor = await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
                "ON session_model_usage(session_id)"
            )
            await session_index_cursor.close()
            model_index_cursor = await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
                "ON session_model_usage(model)"
            )
            await model_index_cursor.close()
            await cursor.commit()
        except sqlite3.OperationalError as exc:
            await cursor.rollback()
            logger.debug("session_model_usage PK heal skipped: %s", exc)
        finally:
            restore_foreign_keys_cursor = await cursor.execute(
                "PRAGMA foreign_keys=ON"
            )
            await restore_foreign_keys_cursor.close()

    async def _init_schema(self) -> None:
        """Create tables and FTS if absent and run retained data migrations."""
        if self._schema_ready:
            return
        cursor = self._initializing_connection  # type: ignore[unresolved-attribute]
        if cursor is None:
            cursor = self._connection  # type: ignore[unresolved-attribute]
        if cursor is None:
            await self._get_connection()  # type: ignore[unresolved-attribute]
            return

        script_cursor = await cursor.executescript(SCHEMA_SQL)
        await script_cursor.close()
        await self._reconcile_columns(cursor)
        # Upstream also heals ``gateway_routing`` here. That table belongs to
        # the removed messaging gateway rather than the retained library API.
        await self._heal_session_model_usage_pk(cursor)

        try:
            index_cursor = await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
                "ON messages(session_id, platform_message_id) "
                "WHERE platform_message_id IS NOT NULL"
            )
            await index_cursor.close()
        except sqlite3.OperationalError as exc:
            logger.debug("idx_messages_platform_msg_id create skipped: %s", exc)

        deferred_cursor = await cursor.executescript(DEFERRED_INDEX_SQL)
        await deferred_cursor.close()
        try:
            active_cursor = await cursor.execute(
                "UPDATE messages SET active = 1 WHERE active IS NULL"
            )
            await active_cursor.close()
        except sqlite3.OperationalError:
            pass

        fts5_available = await self._sqlite_supports_fts5(cursor)
        fts_migrations_complete = True
        async with cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1",
            (FTS_STALE_KEY,),
        ) as stale_cursor:
            self._fts_stale = await stale_cursor.fetchone() is not None
        if self._fts_stale:
            # A prior process detached FTS after corruption. Never recreate
            # triggers opportunistically; only a complete rebuild may clear
            # the durable marker and restore sync.
            await self._drop_all_fts_triggers(cursor)  # type: ignore[attr-defined]
        if not fts5_available:
            await self._drop_fts_triggers(cursor)  # type: ignore[unresolved-attribute]

        async with cursor.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ) as version_cursor:
            version_row = await version_cursor.fetchone()
        if version_row is None:
            insert_cursor = await cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await insert_cursor.close()
        else:
            current_version = (
                version_row["version"]
                if isinstance(version_row, sqlite3.Row)
                else version_row[0]
            )
            if current_version < 10 and SCHEMA_VERSION == 10:
                if fts5_available:
                    trigram_exists = await self._fts_table_probe(
                        cursor,
                        "messages_fts_trigram",
                    )
                    if trigram_exists is False:
                        if await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                            cursor,
                            "messages_fts_trigram",
                            FTS_TRIGRAM_SQL,
                        ):
                            backfill_cursor = await cursor.execute(
                                "INSERT INTO messages_fts_trigram(rowid, content) "
                                "SELECT id, content FROM messages "
                                "WHERE content IS NOT NULL"
                            )
                            await backfill_cursor.close()
                        else:
                            fts_migrations_complete = False
                    elif trigram_exists is None:
                        fts_migrations_complete = False
                else:
                    fts_migrations_complete = False
            if current_version < 11 and SCHEMA_VERSION < 23:
                pass
            if current_version < 16:
                try:
                    tagged_cursor = await cursor.execute(
                        "UPDATE sessions SET model_config = json_set("
                        "COALESCE(model_config, '{}'), '$._delegate_from', "
                        "parent_session_id) "
                        "WHERE parent_session_id IS NOT NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), "
                        "'$._delegate_from') IS NULL "
                        f"AND {_ephemeral_child_sql('sessions')}"
                    )
                    await tagged_cursor.close()
                    orphan_cursor = await cursor.execute(
                        "UPDATE sessions SET model_config = json_set("
                        "COALESCE(model_config, '{}'), '$._delegate_from', "
                        "'__orphaned__') "
                        "WHERE parent_session_id IS NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), "
                        "'$._delegate_from') IS NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), "
                        "'$._branched_from') IS NULL "
                        "AND title IS NULL AND message_count <= 25 "
                        "AND EXISTS (SELECT 1 FROM messages m "
                        "WHERE m.session_id = sessions.id AND m.role = 'tool') "
                        "AND NOT EXISTS (SELECT 1 FROM sessions ch "
                        "WHERE ch.parent_session_id = sessions.id)"
                    )
                    await orphan_cursor.close()
                except sqlite3.OperationalError:
                    pass
            # Upstream v18 backfills gateway metadata from sessions.json.
            # That product-owned file and migration are outside this package.
            if current_version < 20:
                try:
                    usage_cursor = await cursor.execute(
                        """INSERT OR IGNORE INTO session_model_usage (
                               session_id, model, billing_provider,
                               billing_base_url, billing_mode,
                               api_call_count, input_tokens,
                               output_tokens, cache_read_tokens,
                               cache_write_tokens, reasoning_tokens,
                               estimated_cost_usd, actual_cost_usd,
                               cost_status, cost_source, first_seen, last_seen
                           )
                           SELECT id, COALESCE(model, 'unknown'),
                                  COALESCE(billing_provider, ''),
                                  COALESCE(billing_base_url, ''),
                                  COALESCE(billing_mode, ''),
                                  COALESCE(api_call_count, 0),
                                  COALESCE(input_tokens, 0),
                                  COALESCE(output_tokens, 0),
                                  COALESCE(cache_read_tokens, 0),
                                  COALESCE(cache_write_tokens, 0),
                                  COALESCE(reasoning_tokens, 0),
                                  COALESCE(estimated_cost_usd, 0),
                                  COALESCE(actual_cost_usd, 0),
                                  cost_status, cost_source,
                                  started_at, COALESCE(ended_at, started_at)
                           FROM sessions
                           WHERE COALESCE(input_tokens, 0)
                                 + COALESCE(output_tokens, 0)
                                 + COALESCE(cache_read_tokens, 0)
                                 + COALESCE(cache_write_tokens, 0)
                                 + COALESCE(reasoning_tokens, 0) > 0"""
                    )
                    await usage_cursor.close()
                except sqlite3.OperationalError:
                    pass
            if current_version < 22:
                try:
                    async with cursor.execute(
                        "SELECT COUNT(*) FROM "
                        "pragma_table_info('session_model_usage') "
                        "WHERE name = 'task' AND pk > 0"
                    ) as pk_cursor:
                        legacy_pk = (await pk_cursor.fetchone())[0]
                    if not legacy_pk:
                        await self._heal_session_model_usage_pk(cursor)
                except sqlite3.OperationalError as exc:
                    logger.debug(
                        "v22 session_model_usage rebuild skipped: %s",
                        exc,
                    )
            if current_version < 23 and fts5_available:
                if await self._db_has_legacy_inline_fts(  # type: ignore[unresolved-attribute]
                    cursor
                ):
                    await self.set_meta(  # type: ignore[unresolved-attribute]
                        "fts_optimize_available",
                        "1",
                        cursor=cursor,
                    )
            if current_version < 25:
                await self._dedupe_legacy_system_prompts(cursor)

            pending = None
            if fts5_available and not await self._db_has_legacy_inline_fts(  # type: ignore[unresolved-attribute]
                cursor
            ):
                async with cursor.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                ) as pending_cursor:
                    pending = await pending_cursor.fetchone()
                if (
                    pending is None
                    and not await self._has_fts_trash(cursor)  # type: ignore[unresolved-attribute]
                    and not await self._fts_external_index_empty_with_messages(  # type: ignore[unresolved-attribute]
                        cursor
                    )
                ):
                    await self.set_meta(  # type: ignore[unresolved-attribute]
                        "fts_storage_version",
                        str(FTS_STORAGE_VERSION),
                        cursor=cursor,
                    )

            if (
                current_version < SCHEMA_VERSION
                and fts_migrations_complete
                and fts5_available
            ):
                update_version_cursor = await cursor.execute(
                    "UPDATE schema_version SET version = ?",
                    (SCHEMA_VERSION,),
                )
                await update_version_cursor.close()

        title_index_sql = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
            "ON sessions(title) WHERE title IS NOT NULL"
        )
        try:
            title_cursor = await cursor.execute(title_index_sql)
            await title_cursor.close()
        except sqlite3.IntegrityError:
            try:
                repair_cursor = await cursor.execute(
                    """UPDATE sessions AS older
                       SET title = NULL
                       WHERE title IS NOT NULL
                         AND EXISTS (
                             SELECT 1 FROM sessions AS newer
                             WHERE newer.title = older.title
                               AND newer.rowid > older.rowid
                         )"""
                )
                cleared = repair_cursor.rowcount
                await repair_cursor.close()
                logger.warning(
                    "Cleared %d duplicate session title(s) while restoring "
                    "the unique index",
                    cleared,
                )
                title_cursor = await cursor.execute(title_index_sql)
                await title_cursor.close()
            except sqlite3.Error:
                logger.exception(
                    "Could not repair duplicate session titles; "
                    "unique title index not created"
                )
        except sqlite3.OperationalError:
            pass

        if fts5_available and self._fts_stale:
            legacy_fts = await self._db_has_legacy_inline_fts(  # type: ignore[unresolved-attribute]
                cursor
            )
            if await self._recover_stale_fts(cursor, legacy=legacy_fts):
                await self._ensure_fts_cjk_schema(  # type: ignore[unresolved-attribute]
                    cursor
                )
            else:
                self._fts_enabled = False
                self._trigram_available = False
                self._fts_cjk_available = False
        elif fts5_available:
            if await self._db_has_legacy_inline_fts(  # type: ignore[unresolved-attribute]
                cursor
            ):
                triggers_need_repair = (
                    await self._fts_trigger_count(cursor)
                    < len(_FTS_TRIGGERS)
                )
                self._fts_enabled = await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                    cursor,
                    "messages_fts",
                    LEGACY_FTS_SQL,
                )
                if self._fts_enabled:
                    self._trigram_available = await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                        cursor,
                        "messages_fts_trigram",
                        LEGACY_FTS_TRIGRAM_SQL,
                    )
                    if triggers_need_repair:
                        await self._rebuild_legacy_fts_indexes(
                            cursor,
                            include_trigram=self._trigram_available,
                        )
            else:
                triggers_need_repair = (
                    await self._fts_trigger_count(cursor)
                    < len(_FTS_TRIGGERS)
                )
                self._fts_enabled = await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                    cursor,
                    "messages_fts",
                    FTS_SQL,
                )
                if self._fts_enabled:
                    self._trigram_available = await self._ensure_fts_schema(  # type: ignore[unresolved-attribute]
                        cursor,
                        "messages_fts_trigram",
                        FTS_TRIGRAM_SQL,
                    )
                    if triggers_need_repair:
                        await self._rebuild_fts_indexes(
                            cursor,
                            include_trigram=self._trigram_available,
                        )
                    await self._ensure_fts_cjk_schema(  # type: ignore[unresolved-attribute]
                        cursor
                    )

            if self._fts_enabled:
                await self._migrate_broad_fts_update_triggers(cursor)

        await cursor.commit()
        self._schema_ready = True
