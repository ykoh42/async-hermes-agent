"""Schema migration helpers for the native-async ``SessionDB``.

This restores the upstream schema-mixin path incrementally. The retained
trigger-migration methods use the async connection and FTS helpers supplied by
``hermes_state.SessionDB``; this module must not import ``hermes_state``.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

from hermes_state_common import (
    FTS_CJK_STALE_KEY,
    FTS_SQL,
    FTS_TRIGRAM_SQL,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
)


logger = logging.getLogger("hermes_state")


class SessionSchemaMixin:
    """Native-async retained schema migrations for ``SessionDB``."""

    @staticmethod
    def _fts_update_trigger_needs_narrowing(sql: Optional[str]) -> bool:
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
