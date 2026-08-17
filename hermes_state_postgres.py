"""SQLAlchemy Core PostgreSQL backend for the retained SessionDB contract.

The SQLite implementation in hermes_state remains the upstream-shaped
default. This module exposes the same class name and public method signatures
so an embedding application changes only its import and explicit PostgreSQL
DSN.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import sqlite3
import time
import typing
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

from agent.session_activity import ActivityProvenance, build_activity_snapshot
from agent.message_sanitization import _sanitize_surrogates
from agent.skill_commands import (
    SKILL_SCAFFOLD_SQL_LIKE,
    describe_skill_invocation,
)
from hermes_state_common import (
    _RESET_END_REASONS,
    _shape_preview,
    SCHEMA_VERSION,
    escape_like as _escape_like,
)
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)

try:
    import sqlalchemy as _sa
    from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine
except ImportError:  # pragma: no cover - exercised by the missing-extra test
    _sa = None
    _create_async_engine = None

_CONTENT_JSON_PREFIX = "\x00json:"
_POSTGRES_STORAGE_VERSION_KEY = "postgres_storage_version"
_POSTGRES_STORAGE_VERSION = 5
_POSTGRES_DELEGATION_INDEX = "idx_async_delegations_delivery"
_POSTGRES_MIGRATION_LOCK = 731948251
_POSTGRES_MIGRATION_TABLES = (
    "schema_version",
    "state_meta",
    "system_prompts",
    "sessions",
    "messages",
    "session_model_usage",
    "compression_locks",
    "session_turn_leases",
)

_POSTGRES_POOL_DEFAULTS = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30.0,
    "pool_recycle": -1.0,
    "pool_pre_ping": True,
    "pool_use_lifo": False,
}
_POSTGRES_POOL_KEYS = frozenset(_POSTGRES_POOL_DEFAULTS)
_POSTGRES_CONNECT_KEYS = frozenset(
    {
        "connect_timeout",
        "prepare_threshold",
        "application_name",
        "options",
    }
)

_POSTGRES_SEARCH_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS messages_hermes_search_idx "
    "ON messages USING GIN (to_tsvector('simple', "
    "coalesce(content, '') || ' ' || coalesce(tool_name, '') || ' ' || "
    "coalesce(tool_calls, '')))"
)

# Keep the retained SQLite access paths indexed on PostgreSQL as well.  These
# statements use fixed identifiers only; no caller-controlled SQL is interpolated.
_POSTGRES_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_source "
    "ON sessions(source)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_source_id "
    "ON sessions(source, id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_parent "
    "ON sessions(parent_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started "
    "ON sessions(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_messages_session "
    "ON messages(session_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_session_id "
    "ON messages(session_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_assistant_calls_by_session "
    "ON messages(session_id) "
    "WHERE role = 'assistant' AND tool_calls IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_compression_locks_expires "
    "ON compression_locks(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_turn_leases_expires "
    "ON session_turn_leases(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
    "ON session_model_usage(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
    "ON session_model_usage(model)",
    "CREATE INDEX IF NOT EXISTS idx_messages_session_active "
    "ON messages(session_id, active, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_active_null "
    "ON messages(active) WHERE active IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_sessions_session_key "
    "ON sessions(session_key, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer "
    "ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state "
    "ON sessions(handoff_state, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash "
    "ON sessions(system_prompt_hash)",
    "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
    "ON messages(session_id, platform_message_id) "
    "WHERE platform_message_id IS NOT NULL",
    _POSTGRES_SEARCH_INDEX_DDL,
)

_POSTGRES_DELEGATION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at DOUBLE PRECISION NOT NULL,
    completed_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivered_at DOUBLE PRECISION,
    owner_pid INTEGER,
    owner_started_at BIGINT,
    task_json TEXT,
    delivery_claim TEXT,
    delivery_claimed_at DOUBLE PRECISION,
    origin_session_id TEXT NOT NULL DEFAULT '',
    owner_instance_id TEXT,
    lease_expires_at DOUBLE PRECISION
)
"""
_POSTGRES_DELEGATION_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery "
    "ON async_delegations(delivery_state, completed_at)"
)
_POSTGRES_DELEGATION_COLUMNS = frozenset(
    {
        "delegation_id",
        "origin_session",
        "origin_ui_session_id",
        "parent_session_id",
        "state",
        "dispatched_at",
        "completed_at",
        "updated_at",
        "event_json",
        "result_json",
        "delivery_state",
        "delivery_attempts",
        "delivered_at",
        "owner_pid",
        "owner_started_at",
        "task_json",
        "delivery_claim",
        "delivery_claimed_at",
        "origin_session_id",
        "owner_instance_id",
        "lease_expires_at",
    }
)
_POSTGRES_DELEGATION_COLUMN_TYPES = {
    "delegation_id": "text",
    "origin_session": "text",
    "origin_ui_session_id": "text",
    "parent_session_id": "text",
    "state": "text",
    "dispatched_at": "double precision",
    "completed_at": "double precision",
    "updated_at": "double precision",
    "event_json": "text",
    "result_json": "text",
    "delivery_state": "text",
    "delivery_attempts": "integer",
    "delivered_at": "double precision",
    "owner_pid": "integer",
    "owner_started_at": "bigint",
    "task_json": "text",
    "delivery_claim": "text",
    "delivery_claimed_at": "double precision",
    "origin_session_id": "text",
    "owner_instance_id": "text",
    "lease_expires_at": "double precision",
}

_POSTGRES_TURN_LEASE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
)
"""

_POSTGRES_INDEX_NAMES = frozenset(
    {
        "idx_sessions_title_unique",
        "idx_sessions_source",
        "idx_sessions_source_id",
        "idx_sessions_parent",
        "idx_sessions_started",
        "idx_messages_session",
        "idx_messages_session_id",
        "idx_messages_assistant_calls_by_session",
        "idx_compression_locks_expires",
        "idx_session_turn_leases_expires",
        "idx_session_model_usage_session",
        "idx_session_model_usage_model",
        "idx_messages_session_active",
        "idx_messages_active_null",
        "idx_sessions_session_key",
        "idx_sessions_gateway_peer",
        "idx_sessions_handoff_state",
        "idx_sessions_system_prompt_hash",
        "idx_messages_platform_msg_id",
        "messages_hermes_search_idx",
    }
)

_POSTGRES_FOREIGN_KEYS = (
    (
        "fk_sessions_parent_session_id",
        "sessions",
        "FOREIGN KEY (parent_session_id) REFERENCES sessions(id)",
    ),
    (
        "fk_sessions_system_prompt_hash",
        "sessions",
        "FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)",
    ),
    (
        "fk_messages_session_id",
        "messages",
        "FOREIGN KEY (session_id) REFERENCES sessions(id)",
    ),
    (
        "fk_session_model_usage_session_id",
        "session_model_usage",
        "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE",
    ),
)
_POSTGRES_FOREIGN_KEY_NAMES = frozenset(
    name for name, _table, _definition in _POSTGRES_FOREIGN_KEYS
)


def _config_number(
    value: Any,
    name: str,
    *,
    integer: bool = False,
    minimum: float | int | None = None,
    allow_none: bool = False,
) -> int | float | None:
    """Validate one serializable SQLAlchemy/psycopg numeric option."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        expected = "an integer" if integer else "a number"
        raise ValueError(f"database.postgres.{name} must be {expected}")
    if integer and not isinstance(value, int):
        raise ValueError(f"database.postgres.{name} must be an integer")
    if not math.isfinite(float(value)):
        raise ValueError(f"database.postgres.{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"database.postgres.{name} must be >= {minimum}"
        )
    return value


def _postgres_engine_defaults() -> dict[str, Any]:
    """Return a fresh copy of SQLAlchemy's supported pool defaults."""
    return dict(_POSTGRES_POOL_DEFAULTS)


def _json_text(value: Any) -> str:
    # Keep the upstream SQLite JSON text representation (including its
    # spacing/escaping) so exported strings and model_config values do not
    # acquire a backend-specific serialization shape.
    return json.dumps(value, default=str)


def _json_value(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _scrub_surrogates(value: str) -> str:
    return _sanitize_surrogates(value)


def _system_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _contains_cjk(text: str) -> bool:
    """Match SQLite's retained CJK route selection for search."""
    return any(
        0x4E00 <= ord(character) <= 0x9FFF
        or 0x3400 <= ord(character) <= 0x4DBF
        or 0x20000 <= ord(character) <= 0x2A6DF
        or 0x3000 <= ord(character) <= 0x303F
        or 0x3040 <= ord(character) <= 0x30FF
        or 0xAC00 <= ord(character) <= 0xD7AF
        for character in text
    )


class SessionDB:
    """Native-async PostgreSQL implementation of retained SessionDB methods."""

    MAX_TITLE_LENGTH = 100
    TITLE_SOURCE_DERIVED = "derived"
    TITLE_SOURCE_LLM = "llm"
    TITLE_SOURCE_USER = "user"
    _TITLE_SOURCE_RANK = {
        TITLE_SOURCE_DERIVED: 0,
        TITLE_SOURCE_LLM: 1,
        TITLE_SOURCE_USER: 2,
    }

    @classmethod
    def _title_rank(cls, source: str | None) -> int:
        """Return the upstream title-provenance authority rank."""
        if source is None:
            return cls._TITLE_SOURCE_RANK[cls.TITLE_SOURCE_USER]
        return cls._TITLE_SOURCE_RANK.get(str(source), 0)

    def __init__(
        self,
        db_path: os.PathLike[str] | str | None = None,
        read_only: bool = False,
    ) -> None:
        if db_path is None:
            raise ValueError(
                "PostgreSQL SessionDB requires an explicit "
                "postgresql+psycopg:// URL"
            )
        if not isinstance(db_path, str) or not db_path.startswith(
            "postgresql+psycopg://"
        ):
            raise ValueError(
                "db_path must be a postgresql+psycopg:// URL; "
                "environment/database-file fallback is disabled"
            )
        parsed = urlsplit(db_path)
        if not parsed.hostname or not parsed.path.strip("/"):
            raise ValueError(
                "db_path must include a PostgreSQL host and database name"
            )
        self._db_path = db_path
        self._read_only = bool(read_only)
        # Match upstream's creation-time profile ownership while keeping this
        # async constructor state-only: the config file is read only when the
        # first awaited operation initializes the engine.
        self._config_home = get_hermes_home()
        self._engine_options: dict[str, Any] | None = None
        self._engine: Any = None
        self._tables: dict[str, Any] = {}
        self._ready_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def _resolve_engine_options(self) -> dict[str, Any]:
        """Resolve and validate profile-scoped PostgreSQL engine settings."""
        if self._engine_options is not None:
            return dict(self._engine_options)

        from hermes_cli.config import load_config_readonly

        # ``load_config_readonly`` follows the active Hermes home. Install the
        # home captured by __init__ only for this ContextVar-scoped read so a
        # shared worker pool cannot inherit whichever profile made the first
        # request.
        home_token = set_hermes_home_override(self._config_home)
        try:
            config = await load_config_readonly()
        finally:
            reset_hermes_home_override(home_token)

        database = config.get("database") if isinstance(config, dict) else None
        postgres = database.get("postgres") if isinstance(database, dict) else None
        if postgres is None:
            postgres = {}
        if not isinstance(postgres, dict):
            raise ValueError("database.postgres must be a mapping")

        unknown_pool = set(postgres) - _POSTGRES_POOL_KEYS - {"connect_args"}
        if unknown_pool:
            names = ", ".join(sorted(str(key) for key in unknown_pool))
            raise ValueError(
                f"database.postgres contains unsupported key(s): {names}"
            )

        options = _postgres_engine_defaults()
        for name in _POSTGRES_POOL_KEYS:
            if name not in postgres:
                continue
            value = postgres[name]
            if name in {"pool_pre_ping", "pool_use_lifo"}:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"database.postgres.{name} must be a boolean"
                    )
                options[name] = value
            elif name in {"pool_size", "max_overflow"}:
                minimum = 0 if name == "pool_size" else -1
                options[name] = _config_number(
                    value,
                    name,
                    integer=True,
                    minimum=minimum,
                )
            else:
                minimum = -1 if name == "pool_recycle" else 0
                options[name] = _config_number(
                    value,
                    name,
                    minimum=minimum,
                )

        raw_connect_args = postgres.get("connect_args", {})
        if raw_connect_args is None:
            raw_connect_args = {}
        if not isinstance(raw_connect_args, dict):
            raise ValueError("database.postgres.connect_args must be a mapping")
        unknown_connect = set(raw_connect_args) - _POSTGRES_CONNECT_KEYS
        if unknown_connect:
            names = ", ".join(sorted(str(key) for key in unknown_connect))
            raise ValueError(
                "database.postgres.connect_args contains unsupported "
                f"key(s): {names}"
            )

        connect_args: dict[str, Any] = {}
        numeric_connect = {
            "connect_timeout": (True, 1, False),
            "prepare_threshold": (True, 0, True),
        }
        for name, (integer, minimum, allow_none) in numeric_connect.items():
            if name not in raw_connect_args:
                continue
            value = _config_number(
                raw_connect_args[name],
                f"connect_args.{name}",
                integer=integer,
                minimum=minimum,
                allow_none=allow_none,
            )
            if value is not None or name == "prepare_threshold":
                connect_args[name] = value

        for name in ("application_name", "options"):
            if name not in raw_connect_args:
                continue
            value = raw_connect_args[name]
            if not isinstance(value, str):
                raise ValueError(
                    f"database.postgres.connect_args.{name} must be a string"
                )
            connect_args[name] = value

        if connect_args:
            options["connect_args"] = connect_args
        self._engine_options = dict(options)
        return dict(options)

    async def _ensure_postgres_constraints(self, connection: Any) -> None:
        """Add retained foreign-key constraints to an older PG schema.

        ``MetaData.create_all`` creates these constraints for a fresh database,
        but it deliberately does not alter existing tables.  Keep the additive
        migration explicit and serialized by the surrounding advisory lock.
        Invalid legacy rows are surfaced to the caller rather than silently
        weakening referential integrity.
        """
        for name, table, definition in _POSTGRES_FOREIGN_KEYS:
            present = await connection.execute(
                _sa.text(
                    "SELECT 1 FROM pg_constraint "
                    "WHERE conname = :name "
                    "AND conrelid = to_regclass(:table)"
                ),
                {"name": name, "table": table},
            )
            if present.scalar_one_or_none() is not None:
                continue
            await connection.execute(
                _sa.text(
                    f'ALTER TABLE "{table}" ADD CONSTRAINT "{name}" '
                    f"{definition}"
                )
            )

    async def _ensure_postgres_indexes(self, connection: Any) -> None:
        """Create the fixed indexes used by retained SQLite query paths."""
        fixed_names = ", ".join(
            f"'{name}'" for name in sorted(_POSTGRES_INDEX_NAMES)
        )
        invalid_indexes = await connection.execute(
            _sa.text(
                "SELECT c.relname FROM pg_class AS c "
                "JOIN pg_index AS i ON i.indexrelid = c.oid "
                "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = current_schema() "
                f"AND c.relname IN ({fixed_names}) "
                "AND (NOT i.indisvalid OR NOT i.indisready)"
            )
        )
        for row in invalid_indexes:
            name = str(row[0])
            if name in _POSTGRES_INDEX_NAMES:
                await connection.execute(
                    _sa.text(f'DROP INDEX IF EXISTS "{name}"')
                )
        for statement in _POSTGRES_INDEX_DDL:
            await connection.execute(_sa.text(statement))

        title_index = (
            await connection.execute(
                _sa.text(
                    "SELECT pg_get_indexdef(indexrelid) "
                    "FROM pg_index "
                    "WHERE indexrelid = to_regclass(:index_name)"
                ),
                {"index_name": "idx_sessions_title_unique"},
            )
        ).scalar_one_or_none()
        normalized = str(title_index or "").lower()
        if title_index is not None and (
            "create unique index" not in normalized
            or "where (title is not null)" not in normalized
        ):
            await connection.execute(
                _sa.text('DROP INDEX IF EXISTS "idx_sessions_title_unique"')
            )
            title_index = None
        if title_index is None:
            # Older PostgreSQL schemas did not enforce the same title
            # uniqueness as SQLite.  Preserve the newest row's title before
            # creating the fixed partial unique index so repair is idempotent
            # and remains inside the advisory-lock transaction.
            await connection.execute(
                _sa.text(
                    "WITH ranked AS ("
                    "SELECT id, row_number() OVER ("
                    "PARTITION BY title ORDER BY started_at DESC NULLS LAST, id DESC"
                    ") AS position FROM sessions WHERE title IS NOT NULL) "
                    "UPDATE sessions AS sessions_to_repair SET title = NULL, "
                    "title_source = NULL FROM ranked "
                    "WHERE sessions_to_repair.id = ranked.id "
                    "AND ranked.position > 1"
                )
            )
            await connection.execute(
                _sa.text(
                    "CREATE UNIQUE INDEX idx_sessions_title_unique "
                    "ON sessions(title) WHERE title IS NOT NULL"
                )
            )

        search_index = (
            await connection.execute(
                _sa.text(
                    "SELECT pg_get_indexdef(indexrelid) "
                    "FROM pg_index "
                    "WHERE indexrelid = to_regclass(:index_name)"
                ),
                {"index_name": "messages_hermes_search_idx"},
            )
        ).scalar_one_or_none()
        if "tool_calls" not in str(search_index or "").lower():
            await connection.execute(
                _sa.text("DROP INDEX IF EXISTS messages_hermes_search_idx")
            )
            await connection.execute(
                _sa.text(_POSTGRES_SEARCH_INDEX_DDL)
            )

    async def _ensure_postgres_delegation_storage(
        self, connection: Any
    ) -> None:
        """Create the upstream async-delegation table during PG storage setup."""
        await connection.execute(_sa.text(_POSTGRES_DELEGATION_TABLE_DDL))
        columns = await connection.execute(
            _sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'async_delegations'"
            )
        )
        present = {str(row[0]) for row in columns}
        for name, sql_type in (
            ("origin_session_id", "TEXT NOT NULL DEFAULT ''"),
            ("owner_instance_id", "TEXT"),
            ("lease_expires_at", "DOUBLE PRECISION"),
        ):
            if name not in present:
                await connection.execute(
                    _sa.text(
                        f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}"
                    )
                )
        await connection.execute(_sa.text(_POSTGRES_DELEGATION_INDEX_DDL))

    async def _ensure_postgres_turn_lease_storage(
        self, connection: Any
    ) -> None:
        """Create the durable turn-lease table for storage version five."""
        await connection.execute(_sa.text(_POSTGRES_TURN_LEASE_TABLE_DDL))
        await connection.execute(
            _sa.text(
                "CREATE INDEX IF NOT EXISTS idx_session_turn_leases_expires "
                "ON session_turn_leases(expires_at)"
            )
        )

    async def _ensure_postgres_hidden_session_column(
        self, connection: Any
    ) -> None:
        """Apply the upstream additive ``sessions.hidden`` column."""
        present = await connection.execute(
            _sa.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'sessions' AND column_name = 'hidden'"
            )
        )
        if present.scalar_one_or_none() is None:
            await connection.execute(
                _sa.text(
                    "ALTER TABLE sessions ADD COLUMN hidden INTEGER "
                    "NOT NULL DEFAULT 0"
                )
            )

    async def _postgres_delegation_catalog_is_current(
        self, connection: Any
    ) -> bool:
        table = await connection.execute(
            _sa.text("SELECT to_regclass(:table_name)"),
            {"table_name": "async_delegations"},
        )
        if table.scalar_one_or_none() is None:
            return False
        columns = await connection.execute(
            _sa.text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'async_delegations'"
            )
        )
        column_types = {str(row[0]): str(row[1]) for row in columns}
        if not _POSTGRES_DELEGATION_COLUMNS.issubset(column_types):
            return False
        if any(
            column_types[name] != expected
            for name, expected in _POSTGRES_DELEGATION_COLUMN_TYPES.items()
        ):
            return False
        index_result = await connection.execute(
            _sa.text(
                "SELECT c.relname, i.indisvalid, i.indisready, "
                "pg_get_indexdef(i.indexrelid) "
                "FROM pg_class AS c "
                "JOIN pg_index AS i ON i.indexrelid = c.oid "
                "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = current_schema() "
                "AND c.relname = :index_name"
            ),
            {"index_name": _POSTGRES_DELEGATION_INDEX},
        )
        index = index_result.first()
        if index is None or not bool(index[1]) or not bool(index[2]):
            return False
        definition = re.sub(r"\s+", " ", str(index[3]).lower()).strip()
        return "(delivery_state, completed_at)" in definition

    async def _postgres_tables_and_columns_are_current(
        self, connection: Any, metadata: Any
    ) -> bool:
        for table in metadata.tables.values():
            table_name = str(table.name)
            exists = await connection.execute(
                _sa.text("SELECT to_regclass(:table_name)"),
                {"table_name": table_name},
            )
            if exists.scalar_one_or_none() is None:
                return False
            columns = await connection.execute(
                _sa.text(
                    "SELECT attname FROM pg_attribute "
                    "WHERE attrelid = to_regclass(:table_name) "
                    "AND attnum > 0 AND NOT attisdropped"
                ),
                {"table_name": table_name},
            )
            present_columns = {str(row[0]) for row in columns}
            if any(column.name not in present_columns for column in table.columns):
                return False
        return True

    async def _postgres_only_hidden_column_is_missing(
        self, connection: Any, metadata: Any
    ) -> bool:
        """Recognize only the known additive storage gaps."""
        missing: set[tuple[str, str]] = set()
        for table in metadata.tables.values():
            table_name = str(table.name)
            exists = await connection.execute(
                _sa.text("SELECT to_regclass(:table_name)"),
                {"table_name": table_name},
            )
            if exists.scalar_one_or_none() is None:
                missing.add((table_name, "__table__"))
                continue
            columns = await connection.execute(
                _sa.text(
                    "SELECT attname FROM pg_attribute "
                    "WHERE attrelid = to_regclass(:table_name) "
                    "AND attnum > 0 AND NOT attisdropped"
                ),
                {"table_name": table_name},
            )
            present_columns = {str(row[0]) for row in columns}
            missing.update(
                (table_name, str(column.name))
                for column in table.columns
                if column.name not in present_columns
            )
        return bool(missing) and missing.issubset(
            {
                ("sessions", "hidden"),
                ("session_turn_leases", "__table__"),
            }
        )

    async def _postgres_catalog_is_current(
        self, connection: Any, metadata: Any
    ) -> bool:
        """Check the fixed PostgreSQL catalog without changing it."""
        if not await self._postgres_tables_and_columns_are_current(
            connection, metadata
        ):
            return False

        index_rows = await connection.execute(
            _sa.text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = current_schema()"
            )
        )
        if not _POSTGRES_INDEX_NAMES.issubset(
            {str(row[0]) for row in index_rows}
        ):
            return False
        index_states = await connection.execute(
            _sa.text(
                "SELECT c.relname, i.indisvalid, i.indisready "
                "FROM pg_class AS c "
                "JOIN pg_index AS i ON i.indexrelid = c.oid "
                "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = current_schema()"
            )
        )
        states = {
            str(row[0]): (bool(row[1]), bool(row[2]))
            for row in index_states
        }
        if any(
            not states.get(name, (False, False))[0]
            or not states.get(name, (False, False))[1]
            for name in _POSTGRES_INDEX_NAMES
        ):
            return False
        title_index = await connection.execute(
            _sa.text(
                "SELECT pg_get_indexdef(indexrelid) "
                "FROM pg_index "
                "WHERE indexrelid = to_regclass(:index_name)"
            ),
            {"index_name": "idx_sessions_title_unique"},
        )
        title_indexdef = str(title_index.scalar_one_or_none() or "").lower()
        if (
            "create unique index" not in title_indexdef
            or "where (title is not null)" not in title_indexdef
        ):
            return False
        search_index = await connection.execute(
            _sa.text(
                "SELECT pg_get_indexdef(indexrelid) "
                "FROM pg_index "
                "WHERE indexrelid = to_regclass(:index_name)"
            ),
            {"index_name": "messages_hermes_search_idx"},
        )
        if "tool_calls" not in str(search_index.scalar_one_or_none() or "").lower():
            return False

        constraint_rows = await connection.execute(
            _sa.text(
                "SELECT c.conname FROM pg_constraint AS c "
                "JOIN pg_namespace AS n ON n.oid = c.connamespace "
                "WHERE c.contype = 'f' AND n.nspname = current_schema()"
            )
        )
        constraint_names = {str(row[0]) for row in constraint_rows}
        return _POSTGRES_FOREIGN_KEY_NAMES.issubset(constraint_names)

    async def _postgres_schema_versions(self, connection: Any) -> list[int]:
        rows = await connection.execute(
            _sa.text("SELECT version FROM schema_version ORDER BY version")
        )
        return [int(value) for value in rows.scalars().all()]

    async def _postgres_storage_version(self, connection: Any) -> int | None:
        state_meta_exists = await connection.execute(
            _sa.text("SELECT to_regclass(:table_name)"),
            {"table_name": "state_meta"},
        )
        if state_meta_exists.scalar_one_or_none() is None:
            return None
        value = (
            await connection.execute(
                _sa.text(
                    "SELECT value FROM state_meta WHERE key = :key"
                ),
                {"key": _POSTGRES_STORAGE_VERSION_KEY},
            )
        ).scalar_one_or_none()
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "PostgreSQL storage schema version is malformed"
            ) from exc

    async def _postgres_catalog_repair_is_known(
        self, connection: Any
    ) -> bool:
        """Recognize a catalog drift that this backend can repair safely."""
        title_index = (
            await connection.execute(
                _sa.text(
                    "SELECT pg_get_indexdef(indexrelid) FROM pg_index "
                    "WHERE indexrelid = to_regclass(:index_name)"
                ),
                {"index_name": "idx_sessions_title_unique"},
            )
        ).scalar_one_or_none()
        search_index = (
            await connection.execute(
                _sa.text(
                    "SELECT pg_get_indexdef(indexrelid) FROM pg_index "
                    "WHERE indexrelid = to_regclass(:index_name)"
                ),
                {"index_name": "messages_hermes_search_idx"},
            )
        ).scalar_one_or_none()
        title_definition = str(title_index or "").lower()
        search_definition = str(search_index or "").lower()
        known_index_shape = (
            "create unique index" not in title_definition
            or "where (title is not null)" not in title_definition
            or "tool_calls" not in search_definition
        )
        index_rows = await connection.execute(
            _sa.text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = current_schema()"
            )
        )
        if _POSTGRES_INDEX_NAMES - {str(row[0]) for row in index_rows}:
            return True
        constraint_rows = await connection.execute(
            _sa.text(
                "SELECT c.conname FROM pg_constraint AS c "
                "JOIN pg_namespace AS n ON n.oid = c.connamespace "
                "WHERE c.contype = 'f' AND n.nspname = current_schema()"
            )
        )
        missing_constraints = _POSTGRES_FOREIGN_KEY_NAMES - {
            str(row[0]) for row in constraint_rows
        }
        return known_index_shape or bool(missing_constraints)

    async def _postgres_schema_is_current(
        self, connection: Any, metadata: Any
    ) -> bool:
        schema_table = await connection.execute(
            _sa.text("SELECT to_regclass(:table_name)"),
            {"table_name": "schema_version"},
        )
        if schema_table.scalar_one_or_none() is None:
            return False
        versions = await self._postgres_schema_versions(connection)
        storage_version = await self._postgres_storage_version(connection)
        return (
            versions == [SCHEMA_VERSION]
            and storage_version == _POSTGRES_STORAGE_VERSION
            and await self._postgres_catalog_is_current(connection, metadata)
            and await self._postgres_delegation_catalog_is_current(connection)
        )

    async def _lock_postgres_migration_tables(self, connection: Any) -> None:
        quoted_tables = ", ".join(
            f'"{table_name}"' for table_name in _POSTGRES_MIGRATION_TABLES
        )
        await connection.execute(
            _sa.text(
                f"LOCK TABLE {quoted_tables} "
                "IN SHARE ROW EXCLUSIVE MODE"
            )
        )

    async def _write_postgres_schema_versions(self, connection: Any) -> None:
        await connection.execute(_sa.text("DELETE FROM schema_version"))
        await connection.execute(
            _sa.text("INSERT INTO schema_version(version) VALUES (:version)"),
            {"version": SCHEMA_VERSION},
        )
        await connection.execute(
            _sa.text(
                "INSERT INTO state_meta(key, value) VALUES (:key, :value) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value"
            ),
            {
                "key": _POSTGRES_STORAGE_VERSION_KEY,
                "value": str(_POSTGRES_STORAGE_VERSION),
            },
        )

    async def _ensure_ready(self) -> None:
        if self._closed:
            raise RuntimeError("SessionDB is closed")
        current_loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not current_loop:
            raise RuntimeError(
                "PostgreSQL SessionDB is bound to its creating event loop; "
                "create one SessionDB per event loop"
            )
        if self._ready:
            return
        async with self._ready_lock:
            if self._closed:
                raise RuntimeError("SessionDB is closed")
            if self._ready:
                return
            if self._engine is not None:
                # A cancelled/failed schema handshake leaves the engine
                # allocated but not usable. Dispose it before retrying rather
                # than treating a half-initialized pool as ready.
                await self._engine.dispose()
                self._engine = None
                self._tables = {}
            if _sa is None or _create_async_engine is None:
                raise ImportError(
                    "PostgreSQL SessionDB requires the 'postgres' extra "
                    "(SQLAlchemy[asyncio] and psycopg[binary])"
                )
            engine_options = await self._resolve_engine_options()
            if self._read_only:
                engine_options["execution_options"] = {
                    "postgresql_readonly": True,
                }
            self._engine = _create_async_engine(
                self._db_path,
                future=True,
                **engine_options,
            )
            self._loop = current_loop
            metadata = _sa.MetaData()
            self._tables = {
                "schema_version": _sa.Table(
                    "schema_version",
                    metadata,
                    _sa.Column("version", _sa.Integer, primary_key=True),
                ),
                "system_prompts": _sa.Table(
                    "system_prompts",
                    metadata,
                    _sa.Column("hash", _sa.String(128), primary_key=True),
                    _sa.Column("prompt", _sa.Text, nullable=False),
                ),
                "sessions": _sa.Table(
                    "sessions",
                    metadata,
                    _sa.Column("id", _sa.String(255), primary_key=True),
                    _sa.Column("source", _sa.String(128), nullable=False),
                    _sa.Column("user_id", _sa.Text),
                    _sa.Column("session_key", _sa.Text),
                    _sa.Column("chat_id", _sa.Text),
                    _sa.Column("chat_type", _sa.Text),
                    _sa.Column("thread_id", _sa.Text),
                    _sa.Column("display_name", _sa.Text),
                    _sa.Column("origin_json", _sa.Text),
                    _sa.Column("expiry_finalized", _sa.Integer, nullable=False, default=0),
                    _sa.Column("model", _sa.Text),
                    _sa.Column("model_config", _sa.Text),
                    _sa.Column("system_prompt", _sa.Text),
                    _sa.Column(
                        "system_prompt_hash",
                        _sa.String(128),
                        _sa.ForeignKey(
                            "system_prompts.hash",
                            name="fk_sessions_system_prompt_hash",
                        ),
                    ),
                    _sa.Column("payload", _sa.Text, nullable=False),
                    _sa.Column("started_at", _sa.Float, nullable=False),
                    _sa.Column("ended_at", _sa.Float),
                    _sa.Column("end_reason", _sa.Text),
                    _sa.Column(
                        "parent_session_id",
                        _sa.String(255),
                        _sa.ForeignKey(
                            "sessions.id",
                            name="fk_sessions_parent_session_id",
                        ),
                    ),
                    _sa.Column("message_count", _sa.Integer, nullable=False, default=0),
                    _sa.Column("tool_call_count", _sa.Integer, nullable=False, default=0),
                    _sa.Column("input_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("output_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("cache_read_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("cache_write_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("reasoning_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("git_branch", _sa.Text),
                    _sa.Column("git_repo_root", _sa.Text),
                    _sa.Column("billing_provider", _sa.Text),
                    _sa.Column("billing_base_url", _sa.Text),
                    _sa.Column("billing_mode", _sa.Text),
                    _sa.Column("estimated_cost_usd", _sa.Float),
                    _sa.Column("actual_cost_usd", _sa.Float),
                    _sa.Column("cost_status", _sa.Text),
                    _sa.Column("cost_source", _sa.Text),
                    _sa.Column("pricing_version", _sa.Text),
                    _sa.Column("archived", _sa.Integer, nullable=False, default=0),
                    _sa.Column("pinned", _sa.Integer, nullable=False, default=0),
                    _sa.Column("hidden", _sa.Integer, nullable=False, default=0),
                    _sa.Column("last_read_at", _sa.Float),
                    _sa.Column("last_active", _sa.Float),
                    _sa.Column("title", _sa.Text),
                    _sa.Column("title_source", _sa.Text),
                    _sa.Column("cwd", _sa.Text),
                    _sa.Column("last_activity_at", _sa.Float),
                    _sa.Column("last_activity_description", _sa.Text),
                    _sa.Column("last_activity_provenance", _sa.Text),
                    _sa.Column("api_call_count", _sa.Integer, nullable=False, default=0),
                    _sa.Column("handoff_state", _sa.Text),
                    _sa.Column("handoff_platform", _sa.Text),
                    _sa.Column("handoff_error", _sa.Text),
                    _sa.Column("compression_failure_cooldown_until", _sa.Float),
                    _sa.Column("compression_failure_error", _sa.Text),
                    _sa.Column("compression_fallback_streak", _sa.Integer, nullable=False, default=0),
                    _sa.Column("compression_ineffective_count", _sa.Integer, nullable=False, default=0),
                    _sa.Column("profile_name", _sa.Text),
                    _sa.Column("rewind_count", _sa.Integer, nullable=False, default=0),
                ),
                "messages": _sa.Table(
                    "messages",
                    metadata,
                    _sa.Column("id", _sa.BigInteger, _sa.Identity(), primary_key=True),
                    _sa.Column(
                        "session_id",
                        _sa.String(255),
                        _sa.ForeignKey(
                            "sessions.id",
                            name="fk_messages_session_id",
                        ),
                        nullable=False,
                    ),
                    _sa.Column("payload", _sa.Text, nullable=False),
                    _sa.Column("role", _sa.String(64), nullable=False),
                    _sa.Column("content", _sa.Text),
                    _sa.Column("tool_call_id", _sa.Text),
                    _sa.Column("tool_calls", _sa.Text),
                    _sa.Column("tool_name", _sa.Text),
                    _sa.Column("effect_disposition", _sa.Text),
                    _sa.Column("timestamp", _sa.Float, nullable=False),
                    _sa.Column("token_count", _sa.Integer),
                    _sa.Column("finish_reason", _sa.Text),
                    _sa.Column("reasoning", _sa.Text),
                    _sa.Column("reasoning_content", _sa.Text),
                    _sa.Column("reasoning_details", _sa.Text),
                    _sa.Column("codex_reasoning_items", _sa.Text),
                    _sa.Column("codex_message_items", _sa.Text),
                    _sa.Column("platform_message_id", _sa.Text),
                    _sa.Column("observed", _sa.Integer, nullable=False, default=0),
                    _sa.Column("active", _sa.Integer, nullable=False, default=1),
                    _sa.Column("compacted", _sa.Integer, nullable=False, default=0),
                    _sa.Column("api_content", _sa.Text),
                    _sa.Column("display_kind", _sa.Text),
                    _sa.Column("display_metadata", _sa.Text),
                ),
                "session_model_usage": _sa.Table(
                    "session_model_usage",
                    metadata,
                    _sa.Column("id", _sa.BigInteger, _sa.Identity(), primary_key=True),
                    _sa.Column(
                        "session_id",
                        _sa.String(255),
                        _sa.ForeignKey(
                            "sessions.id",
                            name="fk_session_model_usage_session_id",
                            ondelete="CASCADE",
                        ),
                        nullable=False,
                    ),
                    _sa.Column("model", _sa.Text),
                    _sa.Column("billing_provider", _sa.Text),
                    _sa.Column("billing_base_url", _sa.Text),
                    _sa.Column("billing_mode", _sa.Text),
                    _sa.Column("task", _sa.String(255), nullable=False),
                    _sa.Column("api_call_count", _sa.Integer, nullable=False, default=0),
                    _sa.Column("input_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("output_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("cache_read_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("cache_write_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("reasoning_tokens", _sa.Integer, nullable=False, default=0),
                    _sa.Column("estimated_cost_usd", _sa.Float, nullable=False, default=0.0),
                    _sa.Column("actual_cost_usd", _sa.Float, nullable=False, default=0.0),
                    _sa.Column("cost_status", _sa.Text),
                    _sa.Column("cost_source", _sa.Text),
                    _sa.Column("first_seen", _sa.Float),
                    _sa.Column("last_seen", _sa.Float),
                    _sa.Column("payload", _sa.Text, nullable=False),
                ),
                "state_meta": _sa.Table(
                    "state_meta",
                    metadata,
                    _sa.Column("key", _sa.String(255), primary_key=True),
                    _sa.Column("value", _sa.Text),
                ),
                "compression_locks": _sa.Table(
                    "compression_locks",
                    metadata,
                    _sa.Column("session_id", _sa.String(255), primary_key=True),
                    _sa.Column("holder", _sa.String(255), nullable=False),
                    _sa.Column("acquired_at", _sa.Float, nullable=False),
                    _sa.Column("expires_at", _sa.Float, nullable=False),
                ),
                "session_turn_leases": _sa.Table(
                    "session_turn_leases",
                    metadata,
                    _sa.Column("conversation_id", _sa.String(255), primary_key=True),
                    _sa.Column("holder", _sa.String(255), nullable=False),
                    _sa.Column("acquired_at", _sa.Float, nullable=False),
                    _sa.Column("expires_at", _sa.Float, nullable=False),
                ),
            }
            async with self._engine.begin() as connection:
                if self._read_only:
                    transaction_read_only = (
                        await connection.execute(
                            _sa.text("SHOW transaction_read_only")
                        )
                    ).scalar_one()
                    if str(transaction_read_only).lower() != "on":
                        raise RuntimeError(
                            "PostgreSQL read-only SessionDB did not receive "
                            "transaction_read_only=on"
                        )
                    schema_table = await connection.execute(
                        _sa.text("SELECT to_regclass(:table_name)"),
                        {"table_name": "schema_version"},
                    )
                    if schema_table.scalar_one_or_none() is None:
                        raise RuntimeError(
                            "read-only PostgreSQL SessionDB requires an "
                            "initialized schema"
                        )
                    versions = await self._postgres_schema_versions(connection)
                    if any(version > SCHEMA_VERSION for version in versions):
                        raise RuntimeError(
                            "PostgreSQL SessionDB schema is newer than this backend"
                        )
                    if not await self._postgres_schema_is_current(
                        connection, metadata
                    ):
                        raise RuntimeError(
                            "read-only PostgreSQL SessionDB requires the current "
                            "schema; writable migration is required"
                        )
                    self._ready = True
                    return
                await connection.execute(
                    _sa.text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": _POSTGRES_MIGRATION_LOCK},
                )
                if await self._postgres_schema_is_current(connection, metadata):
                    self._ready = True
                    return
                schema_table = await connection.execute(
                    _sa.text("SELECT to_regclass(:table_name)"),
                    {"table_name": "schema_version"},
                )
                if schema_table.scalar_one_or_none() is None:
                    existing_tables = await connection.execute(
                        _sa.text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = current_schema() "
                            "AND table_type = 'BASE TABLE' "
                            "LIMIT 1"
                        )
                    )
                    if existing_tables.first() is not None:
                        raise RuntimeError(
                            "PostgreSQL SessionDB found a non-empty schema without "
                            "schema_version; refusing to guess its migration source"
                        )
                    await connection.run_sync(metadata.create_all)
                    await self._ensure_postgres_constraints(connection)
                    await self._ensure_postgres_delegation_storage(connection)
                    await self._ensure_postgres_indexes(connection)
                    await self._write_postgres_schema_versions(connection)
                    self._ready = True
                    return

                versions = await self._postgres_schema_versions(connection)
                if len(versions) != 1:
                    raise RuntimeError(
                        "PostgreSQL SessionDB requires exactly one schema_version row"
                    )
                current_version = versions[0]
                if current_version > SCHEMA_VERSION:
                    raise RuntimeError(
                        "PostgreSQL SessionDB schema is newer than this backend"
                    )
                if current_version < SCHEMA_VERSION:
                    raise RuntimeError(
                        "PostgreSQL SessionDB logical schema version is unsupported; "
                        "restore a supported backup or migrate it explicitly"
                    )
                storage_version = await self._postgres_storage_version(connection)
                table_shape_current = (
                    await self._postgres_tables_and_columns_are_current(
                        connection, metadata
                    )
                )
                if not table_shape_current and not (
                    storage_version is not None
                    and storage_version < _POSTGRES_STORAGE_VERSION
                    and await self._postgres_only_hidden_column_is_missing(
                        connection, metadata
                    )
                ):
                    raise RuntimeError(
                        "PostgreSQL SessionDB has an unsupported or incomplete "
                        "table shape; refusing implicit column migration"
                    )

                catalog_current = await self._postgres_catalog_is_current(
                    connection, metadata
                )
                if storage_version is None:
                    if catalog_current:
                        # The marker was introduced with the explicit PG
                        # layout migration.  A current pre-marker catalog is
                        # the known source layout for that migration.
                        storage_version = 2
                    elif await self._postgres_catalog_repair_is_known(connection):
                        storage_version = 1
                    else:
                        raise RuntimeError(
                            "PostgreSQL SessionDB physical schema is ambiguous; "
                            "refusing implicit repair"
                        )
                if storage_version > _POSTGRES_STORAGE_VERSION:
                    raise RuntimeError(
                        "PostgreSQL SessionDB physical schema is newer than this "
                        "backend"
                    )
                if storage_version < 1:
                    raise RuntimeError(
                        "PostgreSQL SessionDB physical schema version is unsupported"
                    )

                if storage_version < _POSTGRES_STORAGE_VERSION:
                    if storage_version < 5:
                        # The new table must exist before the migration-table
                        # lock is acquired; PostgreSQL cannot lock a relation
                        # that has not been created yet.
                        await self._ensure_postgres_turn_lease_storage(
                            connection
                        )
                    await self._lock_postgres_migration_tables(connection)
                    if storage_version < 4:
                        await self._ensure_postgres_hidden_session_column(
                            connection
                        )
                    catalog_current = await self._postgres_catalog_is_current(
                        connection, metadata
                    )

                if storage_version == 1 or not catalog_current:
                    if not await self._postgres_catalog_repair_is_known(connection):
                        raise RuntimeError(
                            "PostgreSQL SessionDB physical schema is ambiguous; "
                            "refusing implicit repair"
                        )
                    await self._lock_postgres_migration_tables(connection)
                    await self._ensure_postgres_constraints(connection)
                    await self._ensure_postgres_indexes(connection)
                    if not await self._postgres_catalog_is_current(
                        connection, metadata
                    ):
                        raise RuntimeError(
                            "PostgreSQL SessionDB migration did not produce the "
                            "expected catalog"
                        )

                if storage_version < _POSTGRES_STORAGE_VERSION:
                    await self._ensure_postgres_delegation_storage(connection)
                    await self._ensure_postgres_turn_lease_storage(connection)
                elif not await self._postgres_delegation_catalog_is_current(
                    connection
                ):
                    raise RuntimeError(
                        "PostgreSQL SessionDB delegation schema is incomplete; "
                        "refusing implicit repair"
                    )
                await self._write_postgres_schema_versions(connection)
            self._ready = True

    async def _read(self, operation):
        await self._ensure_ready()
        async with self._engine.connect() as connection:
            return await operation(connection)

    async def _write(self, operation):
        if self._read_only:
            raise PermissionError("read-only PostgreSQL SessionDB")
        await self._ensure_ready()
        async with self._engine.begin() as connection:
            return await operation(connection)

    async def _execute_autocommit(self, statement: str) -> None:
        """Run PostgreSQL maintenance outside an explicit transaction.

        PostgreSQL's ``VACUUM`` cannot run inside a transaction block.  Keep
        this boundary private and narrowly scoped to maintenance statements;
        normal SessionDB writes continue to use ``engine.begin()`` so their
        commit/rollback behaviour remains identical to the other backend.
        """
        if self._read_only:
            raise PermissionError("read-only PostgreSQL SessionDB")
        await self._ensure_ready()
        async with self._engine.connect() as connection:
            await connection.execution_options(isolation_level="AUTOCOMMIT")
            await connection.execute(_sa.text(statement))

    @staticmethod
    def _session_defaults(session_id: str, source: str) -> dict[str, Any]:
        now = time.time()
        return {
            "id": session_id,
            "source": source,
            "user_id": None,
            "session_key": None,
            "chat_id": None,
            "chat_type": None,
            "thread_id": None,
            "display_name": None,
            "origin_json": None,
            "expiry_finalized": 0,
            "model": None,
            "model_config": None,
            "system_prompt": None,
            "system_prompt_hash": None,
            "parent_session_id": None,
            "started_at": now,
            "ended_at": None,
            "end_reason": None,
            "message_count": 0,
            "tool_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "cwd": None,
            "git_branch": None,
            "git_repo_root": None,
            "billing_provider": None,
            "billing_base_url": None,
            "billing_mode": None,
            "estimated_cost_usd": None,
            "actual_cost_usd": None,
            "cost_status": None,
            "cost_source": None,
            "pricing_version": None,
            "title": None,
            "title_source": None,
            "last_activity_at": None,
            "last_activity_description": None,
            "last_activity_provenance": None,
            "api_call_count": 0,
            "handoff_state": None,
            "handoff_platform": None,
            "handoff_error": None,
            "compression_failure_cooldown_until": None,
            "compression_failure_error": None,
            "compression_fallback_streak": 0,
            "compression_ineffective_count": 0,
            "profile_name": None,
            "rewind_count": 0,
            "archived": 0,
            "pinned": 0,
            "hidden": 0,
            "last_read_at": None,
            "last_active": now,
        }

    @staticmethod
    def _session_from_row(row: Any) -> dict[str, Any]:
        data = _json_value(row.payload, {})
        if not isinstance(data, dict):
            data = {}
        data = dict(data)
        for name in (
            "user_id",
            "session_key",
            "chat_id",
            "chat_type",
            "thread_id",
            "display_name",
            "origin_json",
            "expiry_finalized",
            "model_config",
            "system_prompt",
            "system_prompt_hash",
            "end_reason",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "git_branch",
            "git_repo_root",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
            "title_source",
            "last_active",
            "last_activity_at",
            "last_activity_description",
            "last_activity_provenance",
            "api_call_count",
            "handoff_state",
            "handoff_platform",
            "handoff_error",
            "compression_failure_cooldown_until",
            "compression_failure_error",
            "compression_fallback_streak",
            "compression_ineffective_count",
            "profile_name",
            "rewind_count",
        ):
            if hasattr(row, name):
                value = getattr(row, name)
                if value is not None:
                    data[name] = value
        data.update(
            {
                "id": row.id,
                "source": row.source,
                "started_at": row.started_at,
                "ended_at": row.ended_at,
                "parent_session_id": row.parent_session_id,
                "message_count": row.message_count,
                "tool_call_count": row.tool_call_count,
                "archived": row.archived,
                "pinned": row.pinned,
                "hidden": row.hidden,
                "last_read_at": row.last_read_at,
            }
        )
        return data

    @staticmethod
    def _is_listable_child(
        item: dict[str, Any], sessions_by_id: dict[str, dict[str, Any]]
    ) -> bool:
        """Match SQLite's branch/reset child visibility predicate."""
        config = _json_value(item.get("model_config"), {})
        if isinstance(config, dict) and (
            config.get("_branched_from") or config.get("_reset_from")
        ):
            return True
        parent = sessions_by_id.get(item.get("parent_session_id"))
        if parent is None:
            return False
        session_key = item.get("session_key")
        return bool(
            session_key
            and session_key == parent.get("session_key")
            and parent.get("end_reason") in _RESET_END_REASONS
        )

    @staticmethod
    def _message_from_row(row: Any) -> dict[str, Any]:
        data = _json_value(row.payload, {})
        if not isinstance(data, dict):
            data = {}
        data = dict(data)
        content = row.content
        payload_content = data.get("content")
        if isinstance(content, str) and content.startswith(_CONTENT_JSON_PREFIX):
            try:
                content = json.loads(content[len(_CONTENT_JSON_PREFIX) :])
            except (TypeError, ValueError):
                pass
        elif not isinstance(payload_content, str) and "content" in data:
            # SQLite uses a NUL-prefixed marker for non-text content.  NUL is
            # forbidden in PostgreSQL UTF-8 text, so the PostgreSQL backend
            # recovers the original JSON value from its canonical payload
            # instead of inventing a backend-visible content encoding.
            content = payload_content
        for name in (
            "tool_call_id",
            "tool_calls",
            "effect_disposition",
            "token_count",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "platform_message_id",
            "observed",
            "api_content",
            "display_kind",
            "display_metadata",
        ):
            if hasattr(row, name):
                value = getattr(row, name)
                if value is not None:
                    if name in {
                        "tool_calls",
                        "reasoning_details",
                        "codex_reasoning_items",
                        "codex_message_items",
                        "display_metadata",
                    } and isinstance(value, str):
                        value = _json_value(value, value)
                    data[name] = value
        data.update(
            {
                "id": row.id,
                "session_id": row.session_id,
                "role": row.role,
                "content": content,
                "tool_name": row.tool_name,
                "timestamp": row.timestamp,
                "active": row.active,
                "compacted": row.compacted,
            }
        )
        return data

    async def _session(self, connection, session_id: str) -> dict[str, Any] | None:
        table = self._tables["sessions"]
        row = (
            await connection.execute(
                _sa.select(table).where(table.c.id == session_id)
            )
        ).first()
        return self._session_from_row(row) if row is not None else None

    async def _compression_lineage_ids(
        self, connection, session_id: str
    ) -> set[str]:
        """Return a compression root/tip lineage for one transactional write."""
        rows = (
            await connection.execute(
                _sa.select(
                    self._tables["sessions"].c.id,
                    self._tables["sessions"].c.parent_session_id,
                    self._tables["sessions"].c.end_reason,
                    self._tables["sessions"].c.model_config,
                )
            )
        ).all()
        by_id = {
            row.id: {
                "id": row.id,
                "parent_session_id": row.parent_session_id,
                "end_reason": row.end_reason,
                "model_config": row.model_config,
            }
            for row in rows
        }

        def explicit_fork(item: dict[str, Any]) -> bool:
            config = _json_value(item.get("model_config"), {})
            return isinstance(config, dict) and bool(
                config.get("_branched_from") or config.get("_reset_from")
            )

        if session_id not in by_id or explicit_fork(by_id[session_id]):
            return {session_id} if session_id in by_id else set()
        lineage = {session_id}
        current = by_id[session_id]
        while current.get("parent_session_id"):
            parent = by_id.get(current["parent_session_id"])
            if (
                parent is None
                or parent.get("end_reason") != "compression"
                or explicit_fork(current)
                or parent["id"] in lineage
            ):
                break
            lineage.add(parent["id"])
            current = parent
        frontier = list(lineage)
        while frontier:
            parent_id = frontier.pop()
            parent = by_id[parent_id]
            if parent.get("end_reason") != "compression":
                continue
            for child in by_id.values():
                if (
                    child.get("parent_session_id") == parent_id
                    and not explicit_fork(child)
                    and child["id"] not in lineage
                ):
                    lineage.add(child["id"])
                    frontier.append(child["id"])
        return lineage

    async def _is_compression_ancestor(
        self, connection, *, ancestor_id: str, descendant_id: str
    ) -> bool:
        """Return whether *ancestor_id* is a compression predecessor."""
        if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
            return False
        return ancestor_id in await self._compression_lineage_ids(
            connection, descendant_id
        )

    async def _session_turn_lease_key(self, session_id: str) -> str:
        """Return one durable lease key for a compression conversation."""
        if not session_id:
            return session_id
        try:
            current = await self.get_session(session_id)
            seen = {session_id}
            while current:
                config = _json_value(current.get("model_config"), {})
                explicit_fork = current.get("source") == "tool" or (
                    isinstance(config, dict)
                    and bool(
                        config.get("_branched_from")
                        or config.get("_delegate_from")
                    )
                )
                parent_id = current.get("parent_session_id")
                if explicit_fork or not parent_id or parent_id in seen:
                    break
                parent = await self.get_session(parent_id)
                if not parent or parent.get("end_reason") != "compression":
                    break
                seen.add(parent_id)
                current = parent
            return str(current.get("id") or session_id) if current else session_id
        except Exception:
            return session_id

    async def _session_turn_lease_key_on_connection(
        self, connection, session_id: str
    ) -> str:
        """Resolve the lease key on the append transaction's connection."""
        if not session_id:
            return session_id
        rows = (
            await connection.execute(
                _sa.select(
                    self._tables["sessions"].c.id,
                    self._tables["sessions"].c.parent_session_id,
                    self._tables["sessions"].c.end_reason,
                    self._tables["sessions"].c.source,
                    self._tables["sessions"].c.model_config,
                )
            )
        ).all()
        by_id = {str(row.id): row for row in rows}
        current = by_id.get(session_id)
        seen = {session_id}
        while current is not None:
            config = _json_value(current.model_config, {})
            explicit_fork = current.source == "tool" or (
                isinstance(config, dict)
                and bool(config.get("_branched_from") or config.get("_delegate_from"))
            )
            parent_id = current.parent_session_id
            if explicit_fork or not parent_id or str(parent_id) in seen:
                break
            parent = by_id.get(str(parent_id))
            if parent is None or parent.end_reason != "compression":
                break
            seen.add(str(parent_id))
            current = parent
        return str(current.id) if current is not None else session_id

    async def _messages(
        self,
        connection,
        session_id: str,
        *,
        include_inactive: bool = False,
        limit: int | None = None,
        offset: int = 0,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        table = self._tables["messages"]
        conditions = [table.c.session_id == session_id]
        if not include_inactive:
            conditions.append(table.c.active == 1)
        if after_id is not None:
            conditions.append(table.c.id > after_id)
        statement = _sa.select(table).where(*conditions).order_by(table.c.id)
        if offset:
            statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        rows = (await connection.execute(statement)).all()
        return [self._message_from_row(row) for row in rows]

    @staticmethod
    def _duplicate_replayed_user(
        messages: list[dict[str, Any]], message: dict[str, Any]
    ) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content")
        if not isinstance(content, str) or not content:
            return False
        for previous in reversed(messages):
            if previous.get("role") == "user" and previous.get("content") == content:
                return True
            if previous.get("role") == "assistant" and (
                previous.get("content") or previous.get("tool_calls")
            ):
                return False
        return False

    @classmethod
    def _conversation_from_messages(
        cls,
        rows: list[dict[str, Any]],
        *,
        session_id: str,
        include_ancestors: bool,
        repair_alternation: bool,
        include_row_ids: bool,
    ) -> list[dict[str, Any]]:
        """Decode canonical rows into the upstream conversation shape."""
        from agent.memory_manager import sanitize_context

        messages: list[dict[str, Any]] = []
        for row in rows:
            content = row.get("content")
            if row.get("role") in {"user", "assistant"} and isinstance(
                content, str
            ):
                content = sanitize_context(content).strip()
            message: dict[str, Any] = {
                "role": row.get("role"),
                "content": content,
            }
            if include_row_ids and row.get("id") is not None:
                message["_row_id"] = row["id"]
            for key in (
                "api_content",
                "display_kind",
                "effect_disposition",
                "tool_call_id",
                "tool_name",
            ):
                if row.get(key):
                    message[key] = row[key]
            if row.get("display_metadata") is not None:
                message["display_metadata"] = row["display_metadata"]
            if row.get("timestamp"):
                message["timestamp"] = row["timestamp"]
            if row.get("tool_calls"):
                message["tool_calls"] = row["tool_calls"]
            if row.get("platform_message_id"):
                message["message_id"] = row["platform_message_id"]
            if row.get("observed"):
                message["observed"] = True
            if row.get("role") == "assistant":
                for key in (
                    "finish_reason",
                    "reasoning",
                    "reasoning_content",
                    "reasoning_details",
                    "codex_reasoning_items",
                    "codex_message_items",
                ):
                    if row.get(key) is not None and row.get(key) != "":
                        message[key] = row[key]
            if include_ancestors and cls._duplicate_replayed_user(messages, message):
                continue
            messages.append(message)

        if repair_alternation and messages:
            from agent.agent_runtime_helpers import repair_message_sequence

            repair_message_sequence(None, messages)
        del session_id  # retained for parity with the upstream helper contract
        return messages

    async def _conversation_operation(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        session_id = args["session_id"]
        if args["include_ancestors"]:
            session_ids = await self._collection_operation(
                "get_compression_lineage", {"session_id": session_id}
            )
        else:
            session_ids = [session_id]

        async def _collect(connection):
            table = self._tables["messages"]
            conditions = [table.c.session_id.in_(session_ids)]
            if not args["include_inactive"]:
                conditions.append(table.c.active == 1)
            rows = (
                await connection.execute(
                    _sa.select(table).where(*conditions).order_by(table.c.id)
                )
            ).all()
            return self._conversation_from_messages(
                [self._message_from_row(row) for row in rows],
                session_id=session_id,
                include_ancestors=args["include_ancestors"],
                repair_alternation=args["repair_alternation"],
                include_row_ids=args["include_row_ids"],
            )

        return await self._read(_collect)

    async def _update_session(
        self,
        connection,
        session_id: str,
        changes: dict[str, Any],
    ) -> bool:
        table = self._tables["sessions"]
        current_row = (
            await connection.execute(
                _sa.select(table)
                .where(table.c.id == session_id)
                .with_for_update()
            )
        ).first()
        current = (
            self._session_from_row(current_row)
            if current_row is not None
            else None
        )
        if current is None:
            return False
        current.update(changes)
        prompt = current.get("system_prompt")
        prompt_hash = _system_prompt_hash(prompt) if isinstance(prompt, str) else None
        current["system_prompt_hash"] = prompt_hash
        if prompt_hash is not None:
            await connection.execute(
                _sa.text(
                    "INSERT INTO system_prompts(hash, prompt) VALUES "
                    "(:hash, :prompt) ON CONFLICT (hash) DO NOTHING"
                ),
                {"hash": prompt_hash, "prompt": prompt},
            )
        await connection.execute(
            _sa.update(table)
            .where(table.c.id == session_id)
            .values(
                payload=_json_text(current),
                source=str(current.get("source") or "unknown"),
                user_id=current.get("user_id"),
                session_key=current.get("session_key"),
                chat_id=current.get("chat_id"),
                chat_type=current.get("chat_type"),
                thread_id=current.get("thread_id"),
                display_name=current.get("display_name"),
                origin_json=current.get("origin_json"),
                expiry_finalized=int(bool(current.get("expiry_finalized"))),
                model=current.get("model"),
                model_config=_json_text(current["model_config"])
                if isinstance(current.get("model_config"), (dict, list))
                else current.get("model_config"),
                system_prompt=prompt,
                system_prompt_hash=prompt_hash,
                started_at=float(current.get("started_at") or time.time()),
                ended_at=current.get("ended_at"),
                end_reason=current.get("end_reason"),
                parent_session_id=current.get("parent_session_id"),
                message_count=int(current.get("message_count") or 0),
                tool_call_count=int(current.get("tool_call_count") or 0),
                input_tokens=int(current.get("input_tokens") or 0),
                output_tokens=int(current.get("output_tokens") or 0),
                cache_read_tokens=int(current.get("cache_read_tokens") or 0),
                cache_write_tokens=int(current.get("cache_write_tokens") or 0),
                reasoning_tokens=int(current.get("reasoning_tokens") or 0),
                git_branch=current.get("git_branch"),
                git_repo_root=current.get("git_repo_root"),
                billing_provider=current.get("billing_provider"),
                billing_base_url=current.get("billing_base_url"),
                billing_mode=current.get("billing_mode"),
                estimated_cost_usd=current.get("estimated_cost_usd"),
                actual_cost_usd=current.get("actual_cost_usd"),
                cost_status=current.get("cost_status"),
                cost_source=current.get("cost_source"),
                pricing_version=current.get("pricing_version"),
                archived=int(bool(current.get("archived"))),
                pinned=int(bool(current.get("pinned"))),
                last_read_at=current.get("last_read_at"),
                last_active=float(
                    current.get("last_active")
                    or current.get("last_activity_at")
                    or current.get("started_at")
                    or time.time()
                ),
                title=current.get("title"),
                title_source=current.get("title_source"),
                cwd=current.get("cwd"),
                last_activity_at=current.get("last_activity_at"),
                last_activity_description=current.get("last_activity_description"),
                last_activity_provenance=current.get("last_activity_provenance"),
                api_call_count=int(current.get("api_call_count") or 0),
                handoff_state=current.get("handoff_state"),
                handoff_platform=current.get("handoff_platform"),
                handoff_error=current.get("handoff_error"),
                compression_failure_cooldown_until=current.get(
                    "compression_failure_cooldown_until"
                ),
                compression_failure_error=current.get("compression_failure_error"),
                compression_fallback_streak=int(
                    current.get("compression_fallback_streak") or 0
                ),
                compression_ineffective_count=int(
                    current.get("compression_ineffective_count") or 0
                ),
                profile_name=current.get("profile_name"),
                rewind_count=int(current.get("rewind_count") or 0),
            )
        )
        return True

    async def _create_session(
        self,
        connection,
        session_id: str,
        source: str,
        extra: dict[str, Any],
    ) -> str:
        table = self._tables["sessions"]
        await connection.execute(
            _sa.text(
                "SELECT pg_advisory_xact_lock("
                "hashtextextended(:session_id, 0))"
            ),
            {"session_id": session_id},
        )
        if await self._session(connection, session_id) is not None:
            return session_id
        data = self._session_defaults(session_id, source)
        data.update(extra)
        data["id"] = session_id
        data["source"] = source
        if data.get("title") is not None:
            data["title"] = self.sanitize_title(str(data["title"]))
        prompt = data.get("system_prompt")
        prompt_hash = _system_prompt_hash(prompt) if isinstance(prompt, str) else None
        if prompt_hash is not None:
            await connection.execute(
                _sa.text(
                    "INSERT INTO system_prompts(hash, prompt) VALUES "
                    "(:hash, :prompt) ON CONFLICT (hash) DO NOTHING"
                ),
                {"hash": prompt_hash, "prompt": prompt},
            )
        await connection.execute(
            _sa.insert(table).values(
                id=session_id,
                source=source,
                user_id=data.get("user_id"),
                session_key=data.get("session_key"),
                chat_id=data.get("chat_id"),
                chat_type=data.get("chat_type"),
                thread_id=data.get("thread_id"),
                display_name=data.get("display_name"),
                origin_json=data.get("origin_json"),
                expiry_finalized=int(bool(data.get("expiry_finalized"))),
                model=data.get("model"),
                model_config=_json_text(data["model_config"])
                if isinstance(data.get("model_config"), (dict, list))
                else data.get("model_config"),
                system_prompt=prompt,
                system_prompt_hash=prompt_hash,
                payload=_json_text(data),
                started_at=float(data["started_at"]),
                ended_at=data.get("ended_at"),
                parent_session_id=data.get("parent_session_id"),
                end_reason=data.get("end_reason"),
                message_count=int(data.get("message_count") or 0),
                tool_call_count=int(data.get("tool_call_count") or 0),
                input_tokens=int(data.get("input_tokens") or 0),
                output_tokens=int(data.get("output_tokens") or 0),
                cache_read_tokens=int(data.get("cache_read_tokens") or 0),
                cache_write_tokens=int(data.get("cache_write_tokens") or 0),
                reasoning_tokens=int(data.get("reasoning_tokens") or 0),
                git_branch=data.get("git_branch"),
                git_repo_root=data.get("git_repo_root"),
                billing_provider=data.get("billing_provider"),
                billing_base_url=data.get("billing_base_url"),
                billing_mode=data.get("billing_mode"),
                estimated_cost_usd=data.get("estimated_cost_usd"),
                actual_cost_usd=data.get("actual_cost_usd"),
                cost_status=data.get("cost_status"),
                cost_source=data.get("cost_source"),
                pricing_version=data.get("pricing_version"),
                archived=int(bool(data.get("archived"))),
                pinned=int(bool(data.get("pinned"))),
                last_read_at=data.get("last_read_at"),
                last_active=float(data.get("last_active") or data["started_at"]),
                title=data.get("title"),
                title_source=data.get("title_source"),
                cwd=data.get("cwd"),
                last_activity_at=data.get("last_activity_at"),
                last_activity_description=data.get("last_activity_description"),
                last_activity_provenance=data.get("last_activity_provenance"),
                api_call_count=int(data.get("api_call_count") or 0),
                handoff_state=data.get("handoff_state"),
                handoff_platform=data.get("handoff_platform"),
                handoff_error=data.get("handoff_error"),
                compression_failure_cooldown_until=data.get(
                    "compression_failure_cooldown_until"
                ),
                compression_failure_error=data.get("compression_failure_error"),
                compression_fallback_streak=int(
                    data.get("compression_fallback_streak") or 0
                ),
                compression_ineffective_count=int(
                    data.get("compression_ineffective_count") or 0
                ),
                profile_name=data.get("profile_name"),
                rewind_count=int(data.get("rewind_count") or 0),
            )
        )
        # SQLite's retained insert path inherits workspace/profile identity
        # from a parent session.  Keep that behavior at the backend boundary
        # rather than making callers duplicate the values for PostgreSQL.
        parent_id = data.get("parent_session_id")
        if parent_id:
            parent = await self._session(connection, parent_id)
            child = await self._session(connection, session_id)
            if parent is not None and child is not None:
                inherited_fields = [
                    "cwd",
                    "git_repo_root",
                    "git_branch",
                    "profile_name",
                ]
                if parent.get("end_reason") == "compression":
                    inherited_fields.extend(
                        [
                            "user_id",
                            "session_key",
                            "chat_id",
                            "chat_type",
                            "thread_id",
                            "display_name",
                            "origin_json",
                        ]
                    )
                inherited = {
                    field: parent.get(field)
                    for field in inherited_fields
                    if child.get(field) is None and parent.get(field) is not None
                }
                if inherited:
                    await self._update_session(connection, session_id, inherited)
        return session_id

    async def _append(self, connection, values: dict[str, Any]) -> int:
        session_id = str(values["session_id"])
        # Lock the parent row before inserting the child message.  PostgreSQL
        # takes a key-share lock for the FK check; acquiring the update lock
        # first gives concurrent appends one stable order and avoids a
        # deadlock between that FK lock and the counter update below.
        session_table = self._tables["sessions"]
        session_row = (
            await connection.execute(
                _sa.select(session_table)
                .where(session_table.c.id == session_id)
                .with_for_update()
            )
        ).first()
        session = (
            self._session_from_row(session_row)
            if session_row is not None
            else None
        )
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")
        # Compression watermarks make ordinary transcript appends safe while a
        # compressor owns the session lock.  The lock fences competing
        # compression attempts, not a live turn's writes.
        turn_lease_holder = values.get("turn_lease_holder")
        if turn_lease_holder:
            conversation_id = await self._session_turn_lease_key_on_connection(
                connection, session_id
            )
            lease_table = self._tables["session_turn_leases"]
            lease = (
                await connection.execute(
                    _sa.select(lease_table).where(
                        lease_table.c.conversation_id == conversation_id
                    )
                )
            ).first()
            if lease is None or lease.holder != turn_lease_holder:
                from hermes_state import SessionTurnLeaseLostError

                raise SessionTurnLeaseLostError(
                    "Session turn lease lost; refusing transcript write "
                    f"for {session_id!r}"
                )
            now = time.time()
            if float(lease.expires_at) <= now:
                await connection.execute(
                    _sa.update(lease_table)
                    .where(
                        lease_table.c.conversation_id == conversation_id,
                        lease_table.c.holder == turn_lease_holder,
                    )
                    .values(
                        expires_at=now
                        + max(
                            0.1,
                            float(values.get("turn_lease_ttl_seconds", 300.0)),
                        )
                    )
                )
        if (
            session.get("ended_at") is not None
            and session.get("end_reason") == "compression"
        ):
            from hermes_state import CompressionSessionClosedError

            raise CompressionSessionClosedError(session_id)
        timestamp = values.get("timestamp")
        try:
            timestamp = (
                timestamp.timestamp()
                if hasattr(timestamp, "timestamp")
                else float(timestamp)
            )
        except (TypeError, ValueError):
            timestamp = time.time()
        raw_content = values.get("content")
        content = raw_content
        if content is not None and not isinstance(content, str):
            try:
                content = json.dumps(content)
            except (TypeError, ValueError):
                content = _scrub_surrogates(str(content))
        elif isinstance(content, str):
            content = _scrub_surrogates(content)
        tool_calls = values.get("tool_calls")
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError):
                tool_calls = None
        payload = dict(values)
        payload["content"] = (
            _scrub_surrogates(raw_content)
            if isinstance(raw_content, str)
            else raw_content
        )
        payload["tool_calls"] = tool_calls
        payload["timestamp"] = timestamp
        table = self._tables["messages"]
        result = await connection.execute(
            _sa.insert(table)
            .values(
                session_id=session_id,
                payload=_json_text(payload),
                role=str(values.get("role") or "unknown"),
                content=content,
                tool_call_id=values.get("tool_call_id"),
                tool_calls=_json_text(tool_calls) if tool_calls is not None else None,
                tool_name=values.get("tool_name"),
                effect_disposition=values.get("effect_disposition"),
                timestamp=timestamp,
                token_count=values.get("token_count"),
                finish_reason=values.get("finish_reason"),
                reasoning=values.get("reasoning"),
                reasoning_content=values.get("reasoning_content"),
                reasoning_details=_json_text(values.get("reasoning_details"))
                if values.get("reasoning_details") is not None
                else None,
                codex_reasoning_items=_json_text(values.get("codex_reasoning_items"))
                if values.get("codex_reasoning_items") is not None
                else None,
                codex_message_items=_json_text(values.get("codex_message_items"))
                if values.get("codex_message_items") is not None
                else None,
                platform_message_id=values.get("platform_message_id"),
                observed=int(bool(values.get("observed"))),
                active=1,
                compacted=0,
                api_content=values.get("api_content"),
                display_kind=values.get("display_kind"),
                display_metadata=_json_text(values.get("display_metadata"))
                if values.get("display_metadata") is not None
                else None,
            )
            .returning(table.c.id)
        )
        row_id = int(result.scalar_one())
        # SQLite serializes this write path implicitly.  PostgreSQL workers
        # may append to the same session concurrently, so lock the scalar
        # session row before deriving the next counters/payload snapshot.
        current_row = (
            await connection.execute(
                _sa.select(self._tables["sessions"])
                .where(self._tables["sessions"].c.id == session_id)
                .with_for_update()
            )
        ).first()
        current = self._session_from_row(current_row) if current_row is not None else None
        if current is not None:
            tool_count = (
                len(tool_calls) if isinstance(tool_calls, list) else int(bool(tool_calls))
            )
            await self._update_session(
                connection,
                session_id,
                {
                    "message_count": int(current.get("message_count") or 0) + 1,
                    "tool_call_count": int(current.get("tool_call_count") or 0)
                    + tool_count,
                    "last_active": timestamp,
                },
            )
        return row_id

    async def _clear_messages(self, connection, session_id: str) -> None:
        table = self._tables["messages"]
        await connection.execute(_sa.delete(table).where(table.c.session_id == session_id))
        await self._update_session(
            connection,
            session_id,
            {"message_count": 0, "tool_call_count": 0},
        )

    async def _dispatch(self, name: str, values: dict[str, Any]) -> Any:
        args = {key: value for key, value in values.items() if key != "self"}
        if name == "close":
            return await self._close()
        if name in {"create_session", "ensure_session"}:
            extra = dict(args.get("kwargs") or {})
            if name == "ensure_session":
                extra.setdefault("model", args.get("model"))

            async def _create_or_ensure(connection):
                if name == "ensure_session":
                    existing = await self._session(connection, args["session_id"])
                    if existing is not None:
                        changes = {
                            key: value
                            for key, value in extra.items()
                            if value is not None
                        }
                        if changes:
                            await self._update_session(
                                connection, args["session_id"], changes
                            )
                        return args["session_id"]
                return await self._create_session(
                    connection,
                    args["session_id"],
                    args.get("source", "unknown"),
                    extra,
                )

            return await self._write(_create_or_ensure)
        if name == "append_message":
            return await self._write(lambda connection: self._append(connection, args))
        if name == "append_messages_batch":
            messages = args["messages"]
            if not messages:
                return 0

            async def _batch(connection):
                chunk_rows = args.get("chunk_rows")
                if chunk_rows is not None and chunk_rows <= 0:
                    raise ValueError("chunk_rows must be positive")
                inserted = 0
                for message in messages:
                    row = dict(message)
                    row["session_id"] = args["session_id"]
                    row["compression_lock_holder"] = args.get(
                        "compression_lock_holder"
                    )
                    row["turn_lease_holder"] = args.get("turn_lease_holder")
                    row["turn_lease_ttl_seconds"] = args.get(
                        "turn_lease_ttl_seconds", 300.0
                    )
                    inserted += await self._append(connection, row)
                return inserted

            return await self._write(_batch)
        if name == "queue_token_counts":
            # PostgreSQL already provides an async connection pool and row
            # locking, so an in-memory writer queue would only add another
            # lifecycle to own.  Apply the same delta immediately; callers
            # still use the upstream queue/flush names and ``flush`` remains
            # an already-settled barrier.
            token_args = {
                "session_id": args["session_id"],
                "input_tokens": 0,
                "output_tokens": 0,
                "model": None,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost_usd": None,
                "actual_cost_usd": None,
                "cost_status": None,
                "cost_source": None,
                "pricing_version": None,
                "billing_provider": None,
                "billing_base_url": None,
                "billing_mode": None,
                "api_call_count": 0,
                "absolute": False,
            }
            token_args.update(args.get("kwargs") or {})
            return await self._dispatch("update_token_counts", token_args)
        if name == "get_messages_as_conversation":
            return await self._conversation_operation(args)
        if name == "get_session":
            return await self._read(
                lambda connection: self._session(connection, args["session_id"])
            )
        if name == "get_messages":
            async def _get(connection):
                if args["after_id"] is not None and (
                    args["latest"] or args["offset"]
                ):
                    raise ValueError(
                        "after_id is incompatible with latest/offset paging"
                    )
                rows = await self._messages(
                    connection,
                    args["session_id"],
                    include_inactive=args["include_inactive"],
                    limit=None if args["latest"] else args["limit"],
                    offset=0 if args["latest"] else args["offset"],
                    after_id=args["after_id"],
                )
                if args["latest"]:
                    if args["limit"] == 0:
                        return []
                    end = len(rows) - args["offset"] if args["offset"] else len(rows)
                    start = (
                        max(0, end - args["limit"])
                        if args["limit"] is not None
                        else 0
                    )
                    return rows[start:end]
                return rows

            return await self._read(_get)
        if name == "clear_messages":
            return await self._write(
                lambda connection: self._clear_messages(
                    connection, args["session_id"]
                )
            )
        if name == "import_sessions":
            return await self._write(
                lambda connection: self._import_sessions(
                    connection, args["sessions"]
                )
            )
        if name == "replace_messages":
            async def _replace(connection):
                table = self._tables["messages"]
                predicate = [table.c.session_id == args["session_id"]]
                if args["active_only"]:
                    predicate.append(table.c.active == 1)
                if args["archive_dropped"]:
                    await connection.execute(
                        _sa.update(table)
                        .where(*predicate)
                        .values(active=0, compacted=1)
                    )
                else:
                    await connection.execute(_sa.delete(table).where(*predicate))
                await self._update_session(
                    connection,
                    args["session_id"],
                    {"message_count": 0, "tool_call_count": 0},
                )
                for message in args["messages"]:
                    row = dict(message)
                    row["session_id"] = args["session_id"]
                    await self._append(connection, row)

            return await self._write(_replace)
        if name in {"message_count", "count_empty_sessions", "session_count"}:
            return await self._count_operation(name, args)
        if name in {
            "get_session_title",
            "get_session_title_source",
            "get_session_activity",
            "get_session_model_config_value",
            "get_session_by_title",
            "get_session_rich_row",
            "has_archived_messages",
            "get_message_role",
            "latest_message_row_id",
            "latest_user_message_row_id",
            "list_recent_user_messages",
            "get_first_assistant_text",
            "has_platform_message_id",
        }:
            return await self._read_operation(name, args)
        if name in {
            "try_acquire_compression_lock",
            "refresh_compression_lock",
            "release_compression_lock",
            "get_compression_lock_holder",
            "try_acquire_session_turn_lease",
            "acquire_session_turn_lease",
            "refresh_session_turn_lease",
            "release_session_turn_lease",
        }:
            return await self._lock_operation(name, args)
        if name in {"get_meta", "set_meta"}:
            return await self._meta_operation(name, args)
        if name in {
            "set_session_title",
            "set_auto_title",
            "set_auto_title_if_empty",
            "set_session_title_source",
            "set_session_archived",
            "set_session_pinned",
            "set_session_hidden",
            "set_session_read",
            "update_session_model",
            "update_session_cwd",
            "update_session_meta",
            "update_session_billing_route",
            "update_session_runtime_lock",
            "update_system_prompt",
            "touch_session_activity",
            "clear_session_activity_labels",
            "patch_session_model_config",
            "set_compression_fallback_streak",
            "set_compression_ineffective_count",
            "record_compression_failure_cooldown",
            "clear_compression_failure_cooldown",
            "restore_compression_failure_cooldown_row",
            "update_token_counts",
            "queue_token_counts",
            "set_latest_user_api_content",
            "end_session",
            "reopen_session",
            "backfill_repo_roots",
        }:
            return await self._write_operation(name, args)
        if name in {
            "search_messages",
            "search_sessions",
            "search_sessions_by_id",
            "list_sessions_rich",
            "list_prune_candidates",
            "count_open_prune_matches",
            "list_skill_scaffolded_sessions",
            "distinct_session_cwds",
            "session_count_by_source",
            "get_ancestor_display_prefix",
            "get_compression_lineage",
            "get_compression_tip",
            "get_conversation_root",
            "get_next_title_in_lineage",
            "get_resume_conversations",
            "get_resume_message_count",
            "resolve_session_id",
            "resolve_session_by_title",
            "resolve_resume_session_id",
            "get_session_delete_targets",
            "export_session",
            "export_session_lineage",
            "export_all",
            "get_messages_around",
            "get_anchored_view",
        }:
            return await self._collection_operation(name, args)
        if name in {"rewind_to_message", "restore_rewound"}:
            return await self._rewind_operation(name, args)
        if name in {
            "archive_sessions",
            "archive_stale_sessions",
            "delete_empty_sessions",
            "delete_session",
            "delete_session_if_empty",
            "delete_sessions",
            "prune_empty_ghost_sessions",
            "prune_sessions",
            "maybe_auto_archive",
            "maybe_auto_prune_and_vacuum",
            "finalize_orphaned_compression_sessions",
            "reopen_orphaned_compression_session",
            "archive_and_compact",
            "publish_compression_child",
        }:
            return await self._maintenance_operation(name, args)
        if name == "record_auxiliary_usage":
            return await self._usage_operation(args)
        if name in {
            "fts_cjk_rebuild_status",
            "fts_rebuild_status",
            "fts_optimize_available",
            "fts_cjk_rebuild_step",
            "fts_rebuild_step",
            "optimize_fts",
            "rebuild_fts",
            "optimize_fts_storage",
            "vacuum",
            "logical_size_bytes",
            "flush_token_counts",
            "session_count_ge",
            "assert_export_safe",
            "assert_resume_safe",
            "get_compression_failure_cooldown",
            "get_compression_failure_cooldown_row",
            "get_compression_fallback_streak",
            "get_compression_ineffective_count",
            "find_live_compression_child",
            "get_active_message_watermark",
        }:
            return await self._maintenance_read(name, args)
        return await self._fallback_operation(name, args)

    async def _count_operation(self, name: str, args: dict[str, Any]) -> Any:
        async def _count(connection):
            sessions = self._tables["sessions"]
            messages = self._tables["messages"]
            if name == "message_count":
                statement = _sa.select(_sa.func.count()).select_from(messages)
                if args.get("session_id") is not None:
                    statement = statement.where(messages.c.session_id == args["session_id"])
                return int((await connection.execute(statement)).scalar_one())
            rows = (await connection.execute(_sa.select(sessions))).all()
            values = [self._session_from_row(row) for row in rows]
            sessions_by_id = {item["id"]: item for item in values}
            if name == "count_empty_sessions":
                return sum(
                    int(
                        not item.get("message_count")
                        and item.get("ended_at") is not None
                        and not item.get("archived")
                    )
                    for item in values
                )
            result = []
            for item in values:
                if args.get("archived_only"):
                    if not item.get("archived"):
                        continue
                elif not args.get("include_archived", False) and item.get(
                    "archived"
                ):
                    continue
                if args.get("source") is not None and item["source"] != args["source"]:
                    continue
                if args.get("sources") is not None and item["source"] not in args["sources"]:
                    continue
                if args.get("exclude_sources") and item["source"] in args["exclude_sources"]:
                    continue
                if args.get("exclude_children") and item.get("parent_session_id"):
                    if not self._is_listable_child(item, sessions_by_id):
                        continue
                if args.get("cwd_prefix") and not str(item.get("cwd") or "").startswith(
                    args["cwd_prefix"]
                ):
                    continue
                if item.get("message_count", 0) < args.get("min_message_count", 0):
                    continue
                result.append(item)
            return len(result)

        return await self._read(_count)

    async def _read_operation(self, name: str, args: dict[str, Any]) -> Any:
        async def _read_one(connection):
            sid = args.get("session_id")
            session = await self._session(connection, sid) if sid else None
            if name == "get_session_by_title":
                rows = (await connection.execute(_sa.select(self._tables["sessions"]))).all()
                return next(
                    (
                        self._session_from_row(row)
                        for row in rows
                        if self._session_from_row(row).get("title") == args["title"]
                    ),
                    None,
                )
            if name == "get_message_role":
                table = self._tables["messages"]
                row = (
                    await connection.execute(
                        _sa.select(table.c.role).where(
                            table.c.session_id == sid,
                            table.c.id == args["row_id"],
                        )
                    )
                ).first()
                return row[0] if row else None
            if name in {"latest_message_row_id", "latest_user_message_row_id"}:
                table = self._tables["messages"]
                role = "user" if name == "latest_user_message_row_id" else args["role"]
                statement = _sa.select(table.c.id).where(
                    table.c.session_id == sid,
                    table.c.role == role,
                    table.c.active == 1,
                )
                if args.get("require_text", True):
                    # Match SQLite's text predicate: an empty string (the
                    # common tool-call-only assistant row) is not a textual
                    # message, while ``require_text=False`` still returns it.
                    statement = statement.where(
                        table.c.content.is_not(None),
                        _sa.func.trim(table.c.content) != "",
                    )
                rows = (await connection.execute(statement.order_by(table.c.id.desc()))).all()
                offset = args.get("offset", 0) if name != "latest_user_message_row_id" else 0
                return int(rows[offset][0]) if len(rows) > offset else None
            if name == "list_recent_user_messages":
                if args["limit"] <= 0:
                    return []
                table = self._tables["messages"]
                conditions = [
                    table.c.session_id == sid,
                    table.c.role == "user",
                    _sa.or_(
                        table.c.display_kind.is_(None),
                        table.c.display_kind == "",
                    ),
                ]
                if not args["include_inactive"]:
                    conditions.append(table.c.active == 1)
                rows = (
                    await connection.execute(
                        _sa.select(table)
                        .where(*conditions)
                        .order_by(table.c.id.desc())
                        .limit(args["limit"])
                    )
                ).all()
                result = []
                for row in rows:
                    message = self._message_from_row(row)
                    content = message.get("content")
                    if isinstance(content, list):
                        preview = " ".join(
                            str(part.get("text", ""))
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ).strip() or "[multimodal content]"
                    elif isinstance(content, str):
                        preview = describe_skill_invocation(content) or content
                    else:
                        preview = ""
                    preview = " ".join(preview.split())
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    result.append(
                        {
                            "id": message["id"],
                            "timestamp": message["timestamp"],
                            "preview": preview,
                        }
                    )
                return result
            if name == "get_first_assistant_text":
                rows = await self._messages(connection, sid)
                return next(
                    (
                        str(row["content"])
                        for row in rows
                        if row.get("role") == "assistant" and row.get("content")
                    ),
                    "",
                )
            if name == "has_archived_messages":
                table = self._tables["messages"]
                row = (
                    await connection.execute(
                        _sa.select(table.c.id)
                        .where(
                            table.c.session_id == sid,
                            table.c.active == 0,
                        )
                        .limit(1)
                    )
                ).first()
                return row is not None
            if session is None:
                return None
            if name == "get_session_title":
                return session.get("title")
            if name == "get_session_title_source":
                return session.get("title_source")
            if name == "get_session_activity":
                return build_activity_snapshot(
                    last_activity_at=session.get("last_activity_at"),
                    last_activity_description=session.get(
                        "last_activity_description"
                    ),
                    last_activity_provenance=session.get(
                        "last_activity_provenance"
                    ),
                )
            if name == "get_session_model_config_value":
                config = _json_value(session.get("model_config"), {})
                return config.get(args["key"], args["default"]) if isinstance(config, dict) else args["default"]
            if name == "get_session_rich_row":
                messages = await self._messages(connection, sid)
                first_user = next(
                    (
                        row.get("content")
                        for row in messages
                        if row.get("role") == "user" and row.get("content") is not None
                    ),
                    "",
                )
                rich = dict(session)
                rich["preview"] = _shape_preview(first_user)
                rich["unread"] = self.session_unread(rich)
                return rich
            if name == "has_platform_message_id":
                rows = await self._messages(connection, sid, include_inactive=True)
                return any(
                    row.get("platform_message_id") == args["platform_message_id"]
                    for row in rows
                )
            return None

        return await self._read(_read_one)

    async def _write_operation(self, name: str, args: dict[str, Any]) -> Any:
        async def _write_one(connection):
            sid = args.get("session_id")
            session = None
            if sid:
                row = (
                    await connection.execute(
                        _sa.select(self._tables["sessions"])
                        .where(self._tables["sessions"].c.id == sid)
                        .with_for_update()
                    )
                ).first()
                session = self._session_from_row(row) if row is not None else None
            if session is None and sid and name == "update_token_counts":
                await self._create_session(
                    connection,
                    sid,
                    "unknown",
                    {"model": args.get("model")},
                )
                session = await self._session(connection, sid)
            if name == "restore_compression_failure_cooldown_row":
                snapshot = args["snapshot"]
                expected_exists = bool(snapshot.get("session_exists"))
                if session is None:
                    if expected_exists:
                        raise RuntimeError(
                            f"compression cooldown rollback session missing: {sid}"
                        )
                    return None
                if not expected_exists:
                    raise RuntimeError(
                        "cannot restore absent compression cooldown row: "
                        "session now exists"
                    )
                changes = {
                    "compression_failure_cooldown_until": snapshot.get(
                        "cooldown_until"
                    ),
                    "compression_failure_error": snapshot.get("error"),
                }
                await self._update_session(connection, sid, changes)
                return None
            if session is None and sid:
                return False if name.startswith("set_") else None
            if name in {
                "set_session_archived",
                "set_session_pinned",
                "set_session_hidden",
                "set_session_read",
            }:
                if not sid:
                    return False
                lineage_ids = await self._compression_lineage_ids(connection, sid)
                if not lineage_ids:
                    return False
                await connection.execute(
                    _sa.select(self._tables["sessions"].c.id)
                    .where(self._tables["sessions"].c.id.in_(lineage_ids))
                    .order_by(self._tables["sessions"].c.id)
                    .with_for_update()
                )
                field = {
                    "set_session_archived": "archived",
                    "set_session_pinned": "pinned",
                    "set_session_hidden": "hidden",
                    "set_session_read": "last_read_at",
                }[name]
                value = (
                    int(bool(args["archived"]))
                    if name == "set_session_archived"
                    else int(bool(args["pinned"]))
                    if name == "set_session_pinned"
                    else int(bool(args["hidden"]))
                    if name == "set_session_hidden"
                    else time.time() if args["read"] else 0.0
                )
                result = await connection.execute(
                    _sa.update(self._tables["sessions"])
                    .where(self._tables["sessions"].c.id.in_(lineage_ids))
                    .values(**{field: value})
                )
                return bool(result.rowcount or 0)
            changes: dict[str, Any] = {}
            if name in {"set_session_title", "set_auto_title", "set_auto_title_if_empty"}:
                if name == "set_auto_title_if_empty" and session.get("title"):
                    return False
                source = (
                    self.TITLE_SOURCE_USER
                    if name == "set_session_title"
                    else args.get("source", self.TITLE_SOURCE_LLM)
                )
                if name != "set_session_title" and source not in (
                    self.TITLE_SOURCE_DERIVED,
                    self.TITLE_SOURCE_LLM,
                ):
                    raise ValueError(f"invalid automatic title source: {source!r}")
                title = self.sanitize_title(args["title"])
                if title is not None and source != self.TITLE_SOURCE_USER:
                    if (
                        session.get("title") is not None
                        and self._title_rank(session.get("title_source"))
                        >= self._title_rank(source)
                    ):
                        return False
                if title is not None:
                    conflicts = (
                        await connection.execute(
                            _sa.select(self._tables["sessions"].c.id).where(
                                self._tables["sessions"].c.title == title,
                                self._tables["sessions"].c.id != sid,
                            )
                        )
                    ).all()
                    for conflict in conflicts:
                        if await self._is_compression_ancestor(
                            connection,
                            ancestor_id=conflict.id,
                            descendant_id=sid,
                        ):
                            await self._update_session(
                                connection,
                                conflict.id,
                                {"title": None, "title_source": None},
                            )
                        else:
                            raise ValueError(
                                f"Title '{title}' is already in use by session "
                                f"{conflict.id}"
                            )
                changes = {
                    "title": title,
                    "title_source": source if title is not None else None,
                }
            elif name == "set_session_title_source":
                source = args["source"]
                if source not in self._TITLE_SOURCE_RANK:
                    raise ValueError(f"invalid title source: {source!r}")
                if session.get("title") is None:
                    return False
                changes["title_source"] = source
            elif name == "update_session_model":
                changes["model"] = args["model"]
                config = _json_value(session.get("model_config"), None)
                if isinstance(config, dict) and "browser_model_lock" in config:
                    config.pop("browser_model_lock", None)
                    changes["model_config"] = json.dumps(
                        config,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        default=str,
                    )
            elif name == "update_session_cwd":
                if not args.get("session_id") or not args.get("cwd"):
                    return None
                changes["cwd"] = args["cwd"]
                if args["git_branch"] is not None or args["replace_git_meta"]:
                    changes["git_branch"] = args["git_branch"]
                if args["git_repo_root"] is not None or args["replace_git_meta"]:
                    changes["git_repo_root"] = args["git_repo_root"]
            elif name == "update_session_meta":
                changes.update({"model_config": args["model_config_json"], "model": args["model"]})
            elif name == "update_session_billing_route":
                changes.update(
                    {
                        "billing_provider": args["provider"],
                        "billing_base_url": args["base_url"],
                        "billing_mode": args["billing_mode"],
                    }
                )
            elif name == "update_session_runtime_lock":
                config = _json_value(session.get("model_config"), {})
                if not isinstance(config, dict):
                    config = {}
                config["browser_model_lock"] = {
                    "model": args["model"] or "",
                    "provider": args["provider"] or "",
                    "model_options": args["model_options"] or {},
                    "route_source": args["route_source"] or "",
                    "confirmed": bool(args["confirmed"]),
                    "updated_at": time.time(),
                }
                changes["model_config"] = _json_text(config)
                if args["model"] is not None:
                    changes["model"] = args["model"]
                changes["system_prompt"] = None
            elif name == "update_system_prompt":
                changes["system_prompt"] = args["system_prompt"]
            elif name == "touch_session_activity":
                timestamp = float(args["ts"] or time.time())
                previous = session.get("last_activity_at")
                if previous is not None and float(previous) > timestamp:
                    return None
                changes.update(
                    {
                        "last_activity_at": timestamp,
                        "last_activity_description": args["description"],
                        "last_activity_provenance": getattr(
                            args["provenance"], "value", args["provenance"]
                        ),
                        "last_active": timestamp,
                    }
                )
            elif name == "clear_session_activity_labels":
                if not session.get("last_activity_description") and (
                    not session.get("last_activity_provenance")
                    or session.get("last_activity_provenance")
                    == ActivityProvenance.UNKNOWN.value
                ):
                    return None
                changes.update(
                    {
                        "last_activity_description": "",
                        "last_activity_provenance": ActivityProvenance.UNKNOWN.value,
                    }
                )
            elif name == "patch_session_model_config":
                current = _json_value(session.get("model_config"), {})
                if not isinstance(current, dict):
                    current = {}
                current.update(args["patch"])
                changes["model_config"] = _json_text(current)
            elif name == "set_compression_fallback_streak":
                changes["compression_fallback_streak"] = args["streak"]
            elif name == "set_compression_ineffective_count":
                changes["compression_ineffective_count"] = args["count"]
            elif name == "record_compression_failure_cooldown":
                changes.update(
                    {
                        "compression_failure_cooldown_until": args["cooldown_until"],
                        "compression_failure_error": args["error"],
                    }
                )
            elif name == "clear_compression_failure_cooldown":
                changes.update(
                    {
                        "compression_failure_cooldown_until": None,
                        "compression_failure_error": None,
                    }
                )
            elif name == "restore_compression_failure_cooldown_row":
                changes.update(args["snapshot"])
            elif name == "update_token_counts":
                numeric = (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "reasoning_tokens",
                )
                for key in numeric:
                    value = args[key]
                    if args["absolute"]:
                        changes[key] = value
                    else:
                        changes[key] = int(session.get(key) or 0) + int(value or 0)
                if args["absolute"]:
                    changes["estimated_cost_usd"] = (
                        args["estimated_cost_usd"]
                        if args["estimated_cost_usd"] is not None
                        else 0.0
                    )
                    if args["actual_cost_usd"] is not None:
                        changes["actual_cost_usd"] = args["actual_cost_usd"]
                elif args["estimated_cost_usd"] is not None:
                    changes["estimated_cost_usd"] = float(
                        session.get("estimated_cost_usd") or 0.0
                    ) + float(args["estimated_cost_usd"])
                elif session.get("estimated_cost_usd") is None:
                    changes["estimated_cost_usd"] = 0.0
                if args["actual_cost_usd"] is not None:
                    changes["actual_cost_usd"] = (
                        args["actual_cost_usd"]
                        if args["absolute"]
                        else float(session.get("actual_cost_usd") or 0.0)
                        + float(args["actual_cost_usd"])
                    )
                for key in ("cost_status", "cost_source", "pricing_version"):
                    if args.get(key) is not None:
                        changes[key] = args[key]
                has_accounted_usage = bool(
                    args["input_tokens"]
                    or args["output_tokens"]
                    or args["cache_read_tokens"]
                    or args["cache_write_tokens"]
                    or args["reasoning_tokens"]
                    or args["api_call_count"]
                    or args["estimated_cost_usd"]
                    or args["actual_cost_usd"]
                )
                if has_accounted_usage and int(session.get("api_call_count") or 0) == 0:
                    if args.get("model"):
                        changes["model"] = args["model"]
                    if args.get("billing_provider"):
                        changes["billing_provider"] = args["billing_provider"]
                        changes["billing_base_url"] = args.get("billing_base_url")
                        changes["billing_mode"] = args.get("billing_mode")
                elif has_accounted_usage:
                    for key in (
                        "model",
                        "billing_provider",
                        "billing_base_url",
                        "billing_mode",
                    ):
                        value = args.get(key)
                        if value is not None and session.get(key) is None:
                            changes[key] = value
                changes["api_call_count"] = (
                    args["api_call_count"]
                    if args["absolute"]
                    else int(session.get("api_call_count") or 0)
                    + args["api_call_count"]
                )
            elif name == "queue_token_counts":
                changes["queued_token_counts"] = args["kwargs"]
            elif name == "set_latest_user_api_content":
                table = self._tables["messages"]
                rows = await self._messages(connection, sid, include_inactive=True)
                for row in reversed(rows):
                    if row.get("role") != "user":
                        continue
                    payload = dict(row)
                    payload["api_content"] = args["api_content"]
                    await connection.execute(
                        _sa.update(table)
                        .where(table.c.id == row["id"])
                        .values(
                            payload=_json_text(payload),
                            api_content=args["api_content"],
                        )
                    )
                    return int(row["id"])
                return 0
            elif name == "end_session":
                changes.update({"ended_at": time.time(), "end_reason": args["end_reason"]})
            elif name == "reopen_session":
                if (
                    session.get("end_reason") in _RESET_END_REASONS
                    and session.get("session_key")
                ):
                    child_rows = (
                        await connection.execute(
                            _sa.select(self._tables["sessions"]).where(
                                self._tables["sessions"].c.parent_session_id == sid,
                                self._tables["sessions"].c.session_key
                                == session.get("session_key"),
                            )
                        )
                    ).all()
                    for child_row in child_rows:
                        child = self._session_from_row(child_row)
                        config = _json_value(child.get("model_config"), {})
                        if not isinstance(config, dict):
                            config = {}
                        if "_reset_from" not in config:
                            config["_reset_from"] = sid
                            await self._update_session(
                                connection,
                                child["id"],
                                {"model_config": _json_text(config)},
                            )
                changes.update({"ended_at": None, "end_reason": None})
            elif name == "backfill_repo_roots":
                table = self._tables["sessions"]
                for cwd, root in args["cwd_to_root"].items():
                    if not cwd or not root:
                        continue
                    rows = (
                        await connection.execute(
                            _sa.select(table.c.id).where(
                                table.c.cwd == cwd,
                                _sa.or_(
                                    table.c.git_repo_root.is_(None),
                                    table.c.git_repo_root == "",
                                ),
                            )
                        )
                    ).all()
                    for row in rows:
                        await self._update_session(
                            connection, row.id, {"git_repo_root": root}
                        )
                return None
            changed = await self._update_session(connection, sid, changes)
            if name == "update_session_runtime_lock":
                prompts = self._tables["system_prompts"]
                sessions_table = self._tables["sessions"]
                await connection.execute(
                    _sa.delete(prompts).where(
                        ~_sa.exists(
                            _sa.select(1).where(
                                sessions_table.c.system_prompt_hash
                                == prompts.c.hash
                            )
                        )
                    )
                )
            if name == "update_token_counts" and not args["absolute"] and has_accounted_usage:
                await self._record_model_usage(
                    connection,
                    session_id=sid,
                    model=args.get("model") or session.get("model"),
                    billing_provider=args.get("billing_provider")
                    or session.get("billing_provider"),
                    billing_base_url=args.get("billing_base_url")
                    or session.get("billing_base_url"),
                    billing_mode=args.get("billing_mode") or session.get("billing_mode"),
                    task="",
                    api_call_count=args["api_call_count"],
                    input_tokens=args["input_tokens"],
                    output_tokens=args["output_tokens"],
                    cache_read_tokens=args["cache_read_tokens"],
                    cache_write_tokens=args["cache_write_tokens"],
                    reasoning_tokens=args["reasoning_tokens"],
                    estimated_cost_usd=args["estimated_cost_usd"],
                    actual_cost_usd=args["actual_cost_usd"],
                    cost_status=args["cost_status"],
                    cost_source=args["cost_source"],
                )
            return bool(changed) if name.startswith("set_") else None

        return await self._write(_write_one)

    async def _lock_operation(self, name: str, args: dict[str, Any]) -> Any:
        if "session_turn_lease" in name:
            if name == "acquire_session_turn_lease":
                deadline = time.monotonic() + max(0.0, float(args["wait_seconds"]))
                last_notice = -float("inf")
                while True:
                    should_abort = args.get("should_abort")
                    if should_abort is not None:
                        try:
                            if should_abort():
                                return False
                        except Exception:
                            logger.debug(
                                "session turn lease should_abort callback failed",
                                exc_info=True,
                            )
                    if await self._dispatch(
                        "try_acquire_session_turn_lease",
                        {
                            "session_id": args["session_id"],
                            "holder": args["holder"],
                            "ttl_seconds": args["ttl_seconds"],
                            "patience_s": args.get("acquire_patience_s", 0.5),
                        },
                    ):
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    elapsed = max(
                        0.0,
                        float(args["wait_seconds"]) - remaining,
                    )
                    on_wait = args.get("on_wait")
                    interval = float(args.get("wait_notice_interval_seconds", 15.0))
                    if on_wait is not None and (
                        elapsed == 0.0 or interval <= 0.0 or elapsed - last_notice >= interval
                    ):
                        try:
                            on_wait(elapsed)
                            last_notice = elapsed
                        except Exception:
                            logger.debug(
                                "session turn lease on_wait callback failed",
                                exc_info=True,
                            )
                    await asyncio.sleep(
                        min(
                            max(0.01, float(args["poll_interval_seconds"])),
                            remaining,
                        )
                    )

            session_id = args["session_id"]
            holder = args["holder"]
            if not session_id or not holder:
                return False if name != "release_session_turn_lease" else None
            conversation_id = await self._session_turn_lease_key(session_id)

            async def _turn_lease(connection):
                table = self._tables["session_turn_leases"]
                if name == "try_acquire_session_turn_lease":
                    await connection.execute(
                        _sa.text(
                            "SELECT pg_advisory_xact_lock("
                            "hashtextextended(:conversation_id, 0))"
                        ),
                        {"conversation_id": conversation_id},
                    )
                    now = float(
                        (
                            await connection.execute(
                                _sa.text(
                                    "SELECT extract(epoch FROM clock_timestamp())"
                                )
                            )
                        ).scalar_one()
                    )
                    row = (
                        await connection.execute(
                            _sa.select(table)
                            .where(table.c.conversation_id == conversation_id)
                            .with_for_update()
                        )
                    ).first()
                    if row is not None and float(row.expires_at) <= now:
                        await connection.execute(
                            _sa.delete(table).where(
                                table.c.conversation_id == conversation_id
                            )
                        )
                    await connection.execute(
                        _sa.text(
                            "INSERT INTO session_turn_leases "
                            "(conversation_id, holder, acquired_at, expires_at) "
                            "VALUES (:conversation_id, :holder, :acquired_at, :expires_at) "
                            "ON CONFLICT (conversation_id) DO NOTHING"
                        ),
                        {
                            "conversation_id": conversation_id,
                            "holder": holder,
                            "acquired_at": now,
                            "expires_at": now
                            + max(0.1, float(args["ttl_seconds"])),
                        },
                    )
                    owner = (
                        await connection.execute(
                            _sa.select(table.c.holder).where(
                                table.c.conversation_id == conversation_id
                            )
                        )
                    ).scalar_one_or_none()
                    return owner == holder
                if name == "refresh_session_turn_lease":
                    result = await connection.execute(
                        _sa.text(
                            "UPDATE session_turn_leases SET expires_at = "
                            "extract(epoch FROM clock_timestamp()) + :ttl "
                            "WHERE conversation_id = :conversation_id "
                            "AND holder = :holder"
                        ),
                        {
                            "ttl": max(0.1, float(args["ttl_seconds"])),
                            "conversation_id": conversation_id,
                            "holder": holder,
                        },
                    )
                    return result.rowcount > 0
                await connection.execute(
                    _sa.delete(table).where(
                        table.c.conversation_id == conversation_id,
                        table.c.holder == holder,
                    )
                )
                return None

            return await self._write(_turn_lease)

        async def _lock(connection):
            table = self._tables["compression_locks"]
            sid = args["session_id"]
            now = time.time()
            if name != "get_compression_lock_holder":
                # A row-level lock cannot serialize the first insert when two
                # workers observe no row.  PostgreSQL's transaction advisory
                # lock gives this one logical session a stable, bounded
                # serialization point without a process-global mutex.
                await connection.execute(
                    _sa.text(
                        "SELECT pg_advisory_xact_lock("
                        "hashtextextended(:session_id, 0))"
                    ),
                    {"session_id": sid},
                )
            if name == "get_compression_lock_holder":
                row = (
                    await connection.execute(
                        _sa.select(table.c.holder).where(
                            table.c.session_id == sid,
                            table.c.expires_at >= now,
                        )
                    )
                ).first()
                return row[0] if row else None
            if name == "release_compression_lock":
                await connection.execute(
                    _sa.delete(table).where(
                        table.c.session_id == sid,
                        table.c.holder == args["holder"],
                    )
                )
                return None
            row = (
                await connection.execute(
                    _sa.select(table)
                    .where(table.c.session_id == sid)
                    .with_for_update()
                )
            ).first()
            if name == "refresh_compression_lock":
                if row is None or row.holder != args["holder"]:
                    return False
                await connection.execute(
                    _sa.update(table)
                    .where(
                        table.c.session_id == sid,
                        table.c.holder == args["holder"],
                    )
                    .values(expires_at=now + args["ttl_seconds"])
                )
                return True
            if not args["session_id"] or not args["holder"]:
                return False
            if row is not None and row.expires_at >= now and row.holder != args["holder"]:
                return False
            if row is not None:
                await connection.execute(
                    _sa.delete(table).where(table.c.session_id == sid)
                )
            await connection.execute(
                _sa.insert(table).values(
                    session_id=sid,
                    holder=args["holder"],
                    acquired_at=now,
                    expires_at=now + args["ttl_seconds"],
                )
            )
            return True

        if name == "get_compression_lock_holder":
            return await self._read(_lock)
        return await self._write(_lock)

    async def _meta_operation(self, name: str, args: dict[str, Any]) -> Any:
        table = self._tables["state_meta"]
        if name == "get_meta":
            async def _get(connection):
                row = (
                    await connection.execute(
                        _sa.select(table.c.value).where(table.c.key == args["key"])
                    )
                ).first()
                return row[0] if row else None

            return await self._read(_get)

        async def _set(connection):
            await connection.execute(_sa.delete(table).where(table.c.key == args["key"]))
            await connection.execute(
                _sa.insert(table).values(key=args["key"], value=args["value"])
            )

        return await self._write(_set)

    async def _rewind_operation(self, name: str, args: dict[str, Any]) -> Any:
        async def _rewind(connection):
            table = self._tables["messages"]
            if name == "restore_rewound":
                rows = (
                    await connection.execute(
                        _sa.select(table.c.id).where(
                            table.c.session_id == args["session_id"],
                            table.c.id >= args["since_message_id"],
                            table.c.active == 0,
                        )
                    )
                ).all()
                ids = [row[0] for row in rows]
                if ids:
                    await connection.execute(
                        _sa.update(table)
                        .where(table.c.id.in_(ids))
                        .values(active=1)
                    )
                return len(ids)

            target_row = (
                await connection.execute(
                    _sa.select(table).where(
                        table.c.id == args["target_message_id"],
                        table.c.session_id == args["session_id"],
                    )
                )
            ).first()
            if target_row is None:
                raise ValueError(
                    f"message {args['target_message_id']} not found in session "
                    f"{args['session_id']}"
                )
            # ``rewind_to_message`` intentionally returns the raw message row
            # shape used by SQLite: only the encoded content field is decoded;
            # JSON columns such as ``display_metadata`` remain their stored
            # text representation.  ``payload`` is PostgreSQL's private
            # snapshot column and is not part of the retained public row.
            target = {
                column.name: getattr(target_row, column.name)
                for column in table.columns
                if column.name != "payload"
            }
            target["content"] = self._message_from_row(target_row)["content"]
            if target.get("role") != "user":
                raise ValueError(
                    "rewind target must be a 'user' message "
                    f"(got role={target.get('role')!r}, id={args['target_message_id']})"
                )
            rows = (
                await connection.execute(
                    _sa.select(table.c.id).where(
                        table.c.session_id == args["session_id"],
                        table.c.id >= args["target_message_id"],
                        table.c.active == 1,
                    )
                )
            ).all()
            ids = [row[0] for row in rows]
            if ids:
                await connection.execute(
                    _sa.update(table).where(table.c.id.in_(ids)).values(active=0)
                )
            session = await self._session(connection, args["session_id"])
            if session is not None:
                await self._update_session(
                    connection,
                    args["session_id"],
                    {"rewind_count": int(session.get("rewind_count") or 0) + 1},
                )
            head = (
                await connection.execute(
                    _sa.select(_sa.func.max(table.c.id)).where(
                        table.c.session_id == args["session_id"],
                        table.c.active == 1,
                    )
                )
            ).scalar_one()
            return {
                "rewound_count": len(ids),
                "target_message": target,
                "new_head_id": int(head) if head is not None else None,
            }

        return await self._write(_rewind)

    async def _collection_operation(self, name: str, args: dict[str, Any]) -> Any:
        if name == "list_prune_candidates" and args.get("filters"):
            args = {**args, **args["filters"]}
        async def _collect(connection):
            sessions_table = self._tables["sessions"]
            messages_table = self._tables["messages"]
            rows = (
                await connection.execute(_sa.select(sessions_table))
            ).all()
            sessions = [self._session_from_row(row) for row in rows]
            sessions_by_id = {item["id"]: item for item in sessions}

            async def _rich_session(item: dict[str, Any]) -> dict[str, Any]:
                """Add the same presentation-only fields as SQLite listings."""
                rich = dict(item)
                messages = await self._messages(connection, item["id"])
                first_user = next(
                    (
                        row.get("content")
                        for row in messages
                        if row.get("role") == "user"
                        and row.get("content") is not None
                    ),
                    "",
                )
                rich["preview"] = _shape_preview(first_user)
                rich["unread"] = self.session_unread(rich)
                return rich

            if name == "search_messages":
                query = str(args["query"] or "").strip()
                if not query:
                    return []
                fields = args.get("fields")
                if isinstance(fields, str):
                    raise TypeError(
                        "search fields must be a collection of field names, not a string"
                    )
                allowed_fields = (
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
                result_fields = None
                if fields is not None:
                    field_set = set(fields)
                    unknown = field_set.difference(allowed_fields)
                    if unknown:
                        raise ValueError(
                            "unknown search result field(s): "
                            + ", ".join(sorted(unknown))
                        )
                    result_fields = tuple(
                        field for field in allowed_fields if field in field_set
                    )
                visible = _sa.or_(
                    messages_table.c.active == 1,
                    messages_table.c.compacted == 1,
                )
                empty = _sa.literal_column("''")
                search_text = (
                    _sa.func.coalesce(messages_table.c.content, empty)
                    + _sa.literal_column("' '")
                    + _sa.func.coalesce(messages_table.c.tool_name, empty)
                    + _sa.literal_column("' '")
                    + _sa.func.coalesce(messages_table.c.tool_calls, empty)
                )
                search_vector = _sa.func.to_tsvector("simple", search_text)
                search_query = _sa.func.websearch_to_tsquery("simple", query)
                rank = _sa.func.ts_rank_cd(search_vector, search_query)
                headline = _sa.func.ts_headline(
                    "simple",
                    search_text,
                    search_query,
                    "StartSel=>>>, StopSel=<<<, MaxFragments=1, MaxWords=40",
                )
                # The native vector/GIN path handles normal prose and boolean
                # queries.  The ILIKE branch deliberately remains alongside it
                # for CJK and substring searches when pg_trgm is unavailable.
                native_match = search_vector.op("@@")(search_query)
                # PostgreSQL's simple dictionary does not tokenize CJK text
                # the way SQLite's retained CJK/LIKE route does.  Preserve
                # that route (including its OR-token behavior) before trying
                # the native ranking expression; ordinary prose keeps the
                # exact single-literal fallback below.
                if _contains_cjk(query):
                    terms = [
                        token.strip('"')
                        for token in re.findall(r'"[^"]+"|\S+', query)
                        if token.upper() not in {"AND", "OR", "NOT"}
                    ] or [query]
                    term_clauses = []
                    for term in terms:
                        literal_term = _escape_like(term)
                        term_clauses.append(
                            _sa.or_(
                                messages_table.c.content.ilike(
                                    f"%{literal_term}%", escape="\\"
                                ),
                                messages_table.c.tool_name.ilike(
                                    f"%{literal_term}%", escape="\\"
                                ),
                                messages_table.c.payload.ilike(
                                    f"%{literal_term}%", escape="\\"
                                ),
                            )
                        )
                    substring_match = _sa.or_(*term_clauses)
                else:
                    literal_query = _escape_like(query)
                    substring_match = _sa.or_(
                        messages_table.c.content.ilike(
                            f"%{literal_query}%", escape="\\"
                        ),
                        messages_table.c.tool_name.ilike(
                            f"%{literal_query}%", escape="\\"
                        ),
                        messages_table.c.payload.ilike(
                            f"%{literal_query}%", escape="\\"
                        ),
                    )
                statement = (
                    _sa.select(
                        messages_table,
                        sessions_table.c.source,
                        sessions_table.c.model,
                        sessions_table.c.started_at,
                        rank.label("_search_rank"),
                        headline.label("_search_headline"),
                    )
                    .join(
                        sessions_table,
                        sessions_table.c.id == messages_table.c.session_id,
                    )
                    .where(_sa.or_(native_match, substring_match))
                )
                if not args["include_inactive"]:
                    statement = statement.where(visible)
                if args["source_filter"] is not None:
                    statement = statement.where(
                        sessions_table.c.source.in_(args["source_filter"])
                    )
                if args["exclude_sources"]:
                    statement = statement.where(
                        ~sessions_table.c.source.in_(args["exclude_sources"])
                    )
                if args["role_filter"]:
                    statement = statement.where(
                        messages_table.c.role.in_(args["role_filter"])
                    )
                sort = args.get("sort")
                sort = sort.strip().lower() if isinstance(sort, str) else None
                direction = (
                    _sa.asc
                    if sort == "oldest"
                    else _sa.desc
                )
                if sort is None:
                    statement = statement.order_by(
                        _sa.desc(rank),
                        _sa.desc(messages_table.c.timestamp),
                        _sa.desc(messages_table.c.id),
                    )
                else:
                    statement = statement.order_by(
                        direction(messages_table.c.timestamp),
                        direction(messages_table.c.id),
                    )
                statement = statement.offset(args["offset"]).limit(args["limit"])
                result = []
                for row in (await connection.execute(statement)).all():
                    message = self._message_from_row(row)
                    tool_calls = message.get("tool_calls")
                    tool_text = (
                        json.dumps(tool_calls, ensure_ascii=False, default=str)
                        if tool_calls is not None
                        else ""
                    )
                    text = str(
                        message.get("content")
                        or message.get("tool_name")
                        or tool_text
                        or ""
                    )
                    snippet = getattr(row, "_search_headline", None) or text
                    if ">>>" not in snippet:
                        index = text.lower().find(query.lower())
                        if index >= 0:
                            snippet = (
                                text[:index]
                                + ">>>"
                                + text[index : index + len(query)]
                                + "<<<"
                                + text[index + len(query) :]
                            )
                    context_rows = await self._messages(
                        connection,
                        message["session_id"],
                        include_inactive=True,
                    )
                    try:
                        index = next(
                            index
                            for index, candidate in enumerate(context_rows)
                            if candidate["id"] == message["id"]
                        )
                    except StopIteration:
                        index = 0
                    context = []
                    for candidate in context_rows[max(0, index - 1) : index + 2]:
                        content_value = candidate.get("content")
                        if not isinstance(content_value, str):
                            content_value = ""
                        context.append(
                            {
                                "role": candidate.get("role"),
                                "content": content_value[:200],
                            }
                        )
                    item = {
                        "id": message["id"],
                        "session_id": message["session_id"],
                        "role": message["role"],
                        "snippet": snippet,
                        "timestamp": message["timestamp"],
                        "tool_name": message.get("tool_name"),
                        "source": row.source,
                        "model": row.model,
                        "session_started": row.started_at,
                        "context": context,
                    }
                    result.append(
                        {field: item[field] for field in result_fields}
                        if result_fields is not None
                        else item
                    )
                return result
            if name in {"search_sessions", "search_sessions_by_id"}:
                result = []
                for item in sessions:
                    if args.get("source") is not None and item["source"] != args["source"]:
                        continue
                    if args.get("sources") is not None and item["source"] not in args["sources"]:
                        continue
                    if args.get("exclude_sources") and item["source"] in args["exclude_sources"]:
                        continue
                    if args.get("workspace_key") and item.get("cwd") != args["workspace_key"]:
                        continue
                    if name == "search_sessions_by_id" and args["query"].lower() not in item["id"].lower():
                        continue
                    if not args.get("include_archived", True) and item.get("archived"):
                        continue
                    result.append(item)
                result.sort(
                    key=lambda item: (
                        item.get("last_active") or item.get("started_at") or 0,
                        item.get("started_at") or 0,
                        item.get("id") or "",
                    ),
                    reverse=True,
                )
                start = args.get("offset", 0)
                return result[start : start + args.get("limit", 20)]
            if name == "distinct_session_cwds":
                grouped: dict[str, dict[str, Any]] = {}
                for item in sessions:
                    if not args["include_archived"] and item.get("archived"):
                        continue
                    cwd = item.get("cwd")
                    if not cwd:
                        continue
                    group = grouped.setdefault(
                        cwd,
                        {"cwd": cwd, "sessions": 0, "last_active": 0.0},
                    )
                    group["sessions"] += 1
                    group["last_active"] = max(
                        float(group["last_active"]),
                        float(item.get("ended_at") or item.get("started_at") or 0),
                    )
                return list(grouped.values())
            if name == "list_skill_scaffolded_sessions":
                result = []
                for item in sorted(
                    sessions,
                    key=lambda value: value.get("started_at") or 0,
                    reverse=True,
                ):
                    if item.get("title") is None:
                        continue
                    messages = await self._messages(connection, item["id"])
                    first_user = next(
                        (
                            row.get("content")
                            for row in messages
                            if row.get("role") == "user"
                            and isinstance(row.get("content"), str)
                        ),
                        None,
                    )
                    if first_user is not None and first_user.startswith(
                        SKILL_SCAFFOLD_SQL_LIKE[:-1]
                    ):
                        result.append(
                            {
                                "id": item["id"],
                                "title": item["title"],
                                "content": first_user,
                            }
                        )
                    if len(result) >= args["limit"]:
                        break
                return result
            if name == "list_prune_candidates":
                prune_args = dict(args)
                cutoff = prune_args.get("last_active_before")
                if cutoff is None and prune_args.get("older_than_days") is not None:
                    cutoff = time.time() - float(prune_args["older_than_days"]) * 86_400
                result = []
                for item in sessions:
                    if not self._matches_prune(
                        item,
                        source=prune_args.get("source"),
                        last_active_before=cutoff,
                        **{
                            key: value
                            for key, value in prune_args.items()
                            if key
                            in {
                                "last_active_after",
                                "started_before",
                                "started_after",
                                "title_like",
                                "end_reason",
                                "cwd_prefix",
                                "min_messages",
                                "max_messages",
                                "archived",
                                "model_like",
                                "provider",
                                "user_id",
                                "chat_id",
                                "chat_type",
                                "branch_like",
                                "min_tokens",
                                "max_tokens",
                                "min_cost",
                                "max_cost",
                                "min_tool_calls",
                                "max_tool_calls",
                            }
                        },
                    ):
                        continue
                    result.append(
                        {
                            key: item.get(key)
                            for key in (
                                "id",
                                "source",
                                "title",
                                "model",
                                "started_at",
                                "last_active",
                                "ended_at",
                                "message_count",
                                "archived",
                            )
                        }
                    )
                return sorted(
                    result,
                    key=lambda item: (
                        item.get("last_active") or 0,
                        item.get("started_at") or 0,
                    ),
                )
            if name == "count_open_prune_matches":
                prune_args = dict(args)
                cutoff = prune_args.get("last_active_before")
                if cutoff is None and prune_args.get("older_than_days") is not None:
                    cutoff = time.time() - float(prune_args["older_than_days"]) * 86_400
                filter_names = {
                    "last_active_after",
                    "started_before",
                    "started_after",
                    "title_like",
                    "end_reason",
                    "cwd_prefix",
                    "min_messages",
                    "max_messages",
                    "archived",
                    "model_like",
                    "provider",
                    "user_id",
                    "chat_id",
                    "chat_type",
                    "branch_like",
                    "min_tokens",
                    "max_tokens",
                    "min_cost",
                    "max_cost",
                    "min_tool_calls",
                    "max_tool_calls",
                }
                count = 0
                for item in sessions:
                    if item.get("ended_at") is not None:
                        continue
                    # _matches_prune deliberately rejects open sessions. Use
                    # a transient ended marker to reuse the exact predicate,
                    # while retaining the real row for the count only.
                    candidate = dict(item)
                    candidate["ended_at"] = 0.0
                    if self._matches_prune(
                        candidate,
                        source=prune_args.get("source"),
                        last_active_before=cutoff,
                        **{
                            key: value
                            for key, value in prune_args.items()
                            if key in filter_names and value is not None
                        },
                    ):
                        count += 1
                return count
            if name == "list_sessions_rich":
                result = []
                for item in sessions:
                    if args.get("archived_only"):
                        if not item.get("archived"):
                            continue
                    elif not args.get("include_archived", False) and item.get(
                        "archived"
                    ):
                        continue
                    if not args.get("include_hidden", False) and item.get(
                        "hidden"
                    ):
                        continue
                    if args.get("source") is not None and item["source"] != args["source"]:
                        continue
                    if args.get("sources") is not None and item["source"] not in args["sources"]:
                        continue
                    if args.get("exclude_sources") and item["source"] in args["exclude_sources"]:
                        continue
                    if args.get("session_key") is not None and item.get(
                        "session_key"
                    ) != args["session_key"]:
                        continue
                    if args.get("cwd_prefix") and not str(item.get("cwd") or "").startswith(args["cwd_prefix"]):
                        continue
                    if args.get("id_query") and args["id_query"].lower() not in item["id"].lower():
                        continue
                    if args.get("search_query"):
                        needle = args["search_query"].lower()
                        if needle not in item["id"].lower() and needle not in str(
                            item.get("title") or ""
                        ).lower():
                            continue
                    if item.get("message_count", 0) < args.get("min_message_count", 0):
                        continue
                    if not args.get("include_children", False) and item.get(
                        "parent_session_id"
                    ):
                        # Compression/delegate children are hidden from the
                        # picker; explicit branch/reset children remain visible.
                        if not self._is_listable_child(item, sessions_by_id):
                            continue
                    if name == "list_prune_candidates" and item.get("pinned"):
                        continue
                    result.append(
                        await _rich_session(item)
                        if name == "list_sessions_rich"
                        else item
                    )
                if args.get("order_by_last_active"):
                    result.sort(
                        key=lambda item: (
                            item.get("last_active") or item.get("started_at") or 0,
                            item.get("started_at") or 0,
                            item.get("id") or "",
                        ),
                        reverse=True,
                    )
                else:
                    result.sort(
                        key=lambda item: (
                            item.get("started_at") or 0,
                            item.get("id") or "",
                        ),
                        reverse=True,
                    )
                offset = args.get("offset", 0)
                limit = args.get("limit", 20)
                page = result[offset : offset + limit]
                if args.get("include_pinned"):
                    seen_ids = {item["id"] for item in page}
                    page.extend(
                        item
                        for item in result
                        if item.get("pinned") and item["id"] not in seen_ids
                    )
                if args.get("project_compression_tips") and not args.get(
                    "include_children", False
                ):
                    projected = []
                    for item in page:
                        if item.get("end_reason") != "compression":
                            projected.append(item)
                            continue
                        tip_id = await self._collection_operation(
                            "get_compression_tip", {"session_id": item["id"]}
                        )
                        tip = next(
                            (candidate for candidate in sessions if candidate["id"] == tip_id),
                            None,
                        )
                        if tip is None or tip_id == item["id"]:
                            projected.append(item)
                            continue
                        tip_row = await _rich_session(tip)
                        merged = dict(item)
                        for key in (
                            "id",
                            "ended_at",
                            "end_reason",
                            "message_count",
                            "tool_call_count",
                            "title",
                            "last_active",
                            "preview",
                            "model",
                            "system_prompt",
                            "cwd",
                            "git_branch",
                            "git_repo_root",
                        ):
                            if key in tip_row:
                                merged[key] = tip_row[key]
                        merged["_lineage_root_id"] = item["id"]
                        projected.append(merged)
                    page = projected
                return page
            if name == "session_count_by_source":
                result: dict[str, int] = {}
                for item in sessions:
                    if args["archived_only"]:
                        if not item.get("archived"):
                            continue
                    elif not args["include_archived"] and item.get("archived"):
                        continue
                    if args["exclude_children"] and item.get("parent_session_id"):
                        if not self._is_listable_child(item, sessions_by_id):
                            continue
                    source = item["source"] or "cli"
                    result[source] = result.get(source, 0) + 1
                return result
            if name == "get_ancestor_display_prefix":
                session_id = args["session_id"]
                lineage = await self._collection_operation(
                    "get_compression_lineage", {"session_id": session_id}
                )
                if len(lineage) <= 1:
                    return []
                rows = (
                    await connection.execute(
                        _sa.select(messages_table)
                        .where(
                            messages_table.c.session_id.in_(lineage),
                            messages_table.c.active == 1,
                        )
                        .order_by(messages_table.c.id)
                    )
                ).all()
                ancestor_rows = [
                    self._message_from_row(row)
                    for row in rows
                    if row.session_id != session_id
                ]
                return self._conversation_from_messages(
                    ancestor_rows,
                    session_id=session_id,
                    include_ancestors=True,
                    repair_alternation=False,
                    include_row_ids=False,
                )
            if name in {
                "get_compression_lineage",
                "get_compression_tip",
                "get_conversation_root",
            }:
                session_id = args["session_id"]
                session = next(
                    (item for item in sessions if item["id"] == session_id),
                    None,
                )
                if session is None:
                    return [] if name == "get_compression_lineage" else session_id

                def _explicit_fork(item: dict[str, Any]) -> bool:
                    config = _json_value(item.get("model_config"), {})
                    return isinstance(config, dict) and bool(
                        config.get("_branched_from") or config.get("_reset_from")
                    )

                if _explicit_fork(session):
                    if name == "get_compression_lineage":
                        return [session_id]
                    return session_id

                root = session
                seen = {root["id"]}
                while root.get("parent_session_id"):
                    parent = next(
                        (
                            item
                            for item in sessions
                            if item["id"] == root["parent_session_id"]
                        ),
                        None,
                    )
                    if (
                        parent is None
                        or parent.get("end_reason") != "compression"
                        or _explicit_fork(root)
                        or parent["id"] in seen
                    ):
                        break
                    root = parent
                    seen.add(root["id"])

                lineage = [root["id"]]
                lineage_ids = {root["id"]}
                current = root
                while current.get("end_reason") == "compression":
                    children = sorted(
                        (
                            item
                            for item in sessions
                            if item.get("parent_session_id") == current["id"]
                            and not _explicit_fork(item)
                        ),
                        key=lambda item: item.get("started_at") or 0,
                    )
                    next_child = next(
                        (item for item in children if item["id"] not in lineage_ids),
                        None,
                    )
                    if next_child is None:
                        break
                    lineage.append(next_child["id"])
                    lineage_ids.add(next_child["id"])
                    current = next_child
                if name == "get_compression_lineage":
                    return lineage if session_id in lineage else [session_id]
                if not lineage:
                    return session_id
                return lineage[-1] if name == "get_compression_tip" else lineage[0]
            if name == "get_next_title_in_lineage":
                base_title = args["base_title"]
                match = re.match(r"^(.*?) #(\d+)$", base_title)
                base = match.group(1) if match else base_title
                escaped = _escape_like(base)
                title_rows = (
                    await connection.execute(
                        _sa.select(sessions_table.c.title).where(
                            _sa.or_(
                                sessions_table.c.title == base,
                                sessions_table.c.title.ilike(
                                    f"{escaped} #%", escape="\\"
                                ),
                            )
                        )
                    )
                ).all()
                titles = [row[0] for row in title_rows]
                if not titles:
                    return base
                suffixes = [
                    int(found.group(1))
                    for title in titles
                    if (
                        found := re.match(
                            rf"^{re.escape(base)} #(\d+)$", title or ""
                        )
                    )
                ]
                return f"{base} #{max(suffixes, default=1) + 1}"
            if name == "get_resume_message_count":
                lineage = await self._collection_operation(
                    "get_compression_lineage",
                    {"session_id": args["session_id"]},
                )
                count = (
                    await connection.execute(
                        _sa.select(_sa.func.count())
                        .select_from(messages_table)
                        .where(
                            messages_table.c.session_id.in_(lineage),
                            messages_table.c.active == 1,
                        )
                    )
                ).scalar_one()
                return int(count)
            if name == "get_resume_conversations":
                model_history = await self._conversation_operation(
                    {
                        "session_id": args["session_id"],
                        "include_ancestors": False,
                        "include_inactive": False,
                        "repair_alternation": True,
                        "include_row_ids": True,
                    }
                )
                display_history = await self._conversation_operation(
                    {
                        "session_id": args["session_id"],
                        "include_ancestors": True,
                        "include_inactive": False,
                        "repair_alternation": False,
                        "include_row_ids": True,
                    }
                )
                return model_history, display_history
            if name == "resolve_session_id":
                query = args["session_id_or_prefix"]
                matches = [
                    item["id"]
                    for item in sessions
                    if item["id"] == query or item["id"].startswith(query)
                ]
                return matches[0] if len(matches) == 1 else None
            if name == "resolve_session_by_title":
                return next(
                    (
                        item["id"]
                        for item in sessions
                        if item.get("title") == args["title"]
                    ),
                    None,
                )
            if name == "resolve_resume_session_id":
                return args["session_id"]
            if name == "get_session_delete_targets":
                targets = [args["session_id"]]
                changed = True
                while changed:
                    changed = False
                    for item in sessions:
                        config = _json_value(item.get("model_config"), {})
                        is_delegate = isinstance(config, dict) and config.get(
                            "_delegate_from"
                        )
                        if (
                            item.get("parent_session_id") in targets
                            and is_delegate
                            and item["id"] not in targets
                        ):
                            targets.append(item["id"])
                            changed = True
                return targets
            if name in {"export_session", "export_session_lineage", "export_all"}:
                if name == "export_all":
                    selected = sessions
                elif name == "export_session":
                    selected = [
                        item
                        for item in sessions
                        if item["id"] == args["session_id"]
                    ]
                else:
                    lineage = await self._collection_operation(
                        "get_compression_lineage",
                        {"session_id": args["session_id"]},
                    )
                    selected = [item for item in sessions if item["id"] in lineage]
                exported = []
                for item in selected:
                    exported.append(
                        {
                            **item,
                            "messages": await self._messages(connection, item["id"]),
                        }
                    )
                if name == "export_all":
                    return exported
                if not exported:
                    return None
                if name == "export_session":
                    return exported[0]
                base = dict(exported[-1])
                base["segments"] = exported
                base["lineage_session_ids"] = [item["id"] for item in exported]
                base["messages"] = [
                    message
                    for item in exported
                    for message in item["messages"]
                ]
                return base
            if name == "get_messages_around":
                rows = await self._messages(
                    connection,
                    args["session_id"],
                    include_inactive=True,
                )
                index = next(
                    (
                        index
                        for index, row in enumerate(rows)
                        if row["id"] == args["around_message_id"]
                    ),
                    None,
                )
                if index is None:
                    return {
                        "window": [],
                        "messages_before": 0,
                        "messages_after": 0,
                    }
                window = max(0, int(args["window"]))
                before = rows[max(0, index - window) : index + 1]
                after = rows[index + 1 : index + 1 + window]
                return {
                    "window": before + after,
                    "messages_before": max(0, len(before) - 1),
                    "messages_after": len(after),
                }
            if name == "get_anchored_view":
                rows = await self._messages(
                    connection,
                    args["session_id"],
                    include_inactive=True,
                )
                anchor_index = next(
                    (
                        index
                        for index, row in enumerate(rows)
                        if row["id"] == args["around_message_id"]
                    ),
                    None,
                )
                if anchor_index is None:
                    return {
                        "window": [],
                        "messages_before": 0,
                        "messages_after": 0,
                        "bookend_start": [],
                        "bookend_end": [],
                    }
                window = max(0, int(args["window"]))
                primitive_rows = rows[
                    max(0, anchor_index - window) : anchor_index + window + 1
                ]
                keep_roles = args["keep_roles"]
                if keep_roles is not None:
                    allowed = set(keep_roles)
                    visible_rows = [
                        row
                        for row in primitive_rows
                        if row["id"] == args["around_message_id"]
                        or row.get("role") in allowed
                    ]
                else:
                    visible_rows = primitive_rows
                bookend = max(0, int(args["bookend"]))
                allowed = set(keep_roles) if keep_roles is not None else None
                starts = [
                    row
                    for row in rows[: max(0, anchor_index - window)]
                    if (allowed is None or row.get("role") in allowed)
                    and str(row.get("content") or "")
                ][:bookend]
                trailing = [
                    row
                    for row in rows[anchor_index + window + 1 :]
                    if (allowed is None or row.get("role") in allowed)
                    and str(row.get("content") or "")
                ]
                ends = trailing[-bookend:] if bookend else []
                return {
                    "window": visible_rows,
                    "messages_before": min(window, anchor_index),
                    "messages_after": min(window, len(rows) - anchor_index - 1),
                    "bookend_start": starts,
                    "bookend_end": ends,
                }
            return []

        return await self._read(_collect)

    async def _import_sessions(
        self,
        connection,
        sessions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        imported = 0
        errors = []
        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
                errors.append({"index": index, "error": "session id is required"})
                continue
            session_id = str(raw["id"]).strip()
            data = {key: value for key, value in raw.items() if key != "messages"}
            source = str(data.pop("source", "import"))
            await self._create_session(connection, session_id, source, data)
            for message in raw.get("messages") or []:
                row = dict(message)
                row.pop("id", None)
                row["session_id"] = session_id
                await self._append(connection, row)
            imported += 1
        return {"imported": imported, "errors": errors}

    @staticmethod
    def _matches_prune(
        item: dict[str, Any],
        *,
        last_active_before: float | None = None,
        last_active_after: float | None = None,
        started_before: float | None = None,
        started_after: float | None = None,
        source: str | None = None,
        title_like: str | None = None,
        end_reason: str | None = None,
        cwd_prefix: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        archived: bool | None = None,
        model_like: str | None = None,
        provider: str | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        branch_like: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        min_cost: float | None = None,
        max_cost: float | None = None,
        min_tool_calls: int | None = None,
        max_tool_calls: int | None = None,
    ) -> bool:
        """Apply the upstream prune predicate to a decoded session row."""
        if item.get("ended_at") is None:
            return False
        active = float(item.get("last_active") or item.get("started_at") or 0)
        started = float(item.get("started_at") or 0)
        if last_active_before is not None and active >= last_active_before:
            return False
        if last_active_after is not None and active < last_active_after:
            return False
        if started_before is not None and started >= started_before:
            return False
        if started_after is not None and started < started_after:
            return False
        if source and item.get("source") != source:
            return False
        if title_like and title_like.casefold() not in str(item.get("title") or "").casefold():
            return False
        if end_reason and item.get("end_reason") != end_reason:
            return False
        if cwd_prefix and not str(item.get("cwd") or "").startswith(cwd_prefix):
            return False
        messages = int(item.get("message_count") or 0)
        if min_messages is not None and messages < min_messages:
            return False
        if max_messages is not None and messages > max_messages:
            return False
        if archived is not None and bool(item.get("archived")) is not archived:
            return False
        if model_like and model_like.casefold() not in str(item.get("model") or "").casefold():
            return False
        if provider and str(item.get("billing_provider") or "").casefold() != provider.casefold():
            return False
        for key, expected in (
            ("user_id", user_id),
            ("chat_id", chat_id),
            ("chat_type", chat_type),
        ):
            if expected and item.get(key) != expected:
                return False
        if branch_like and branch_like.casefold() not in str(item.get("git_branch") or "").casefold():
            return False
        tokens = int(item.get("input_tokens") or 0) + int(item.get("output_tokens") or 0)
        if min_tokens is not None and tokens < min_tokens:
            return False
        if max_tokens is not None and tokens > max_tokens:
            return False
        cost = item.get("actual_cost_usd")
        if cost is None:
            cost = item.get("estimated_cost_usd") or 0
        if min_cost is not None and float(cost) < min_cost:
            return False
        if max_cost is not None and float(cost) > max_cost:
            return False
        calls = int(item.get("tool_call_count") or 0)
        if min_tool_calls is not None and calls < min_tool_calls:
            return False
        if max_tool_calls is not None and calls > max_tool_calls:
            return False
        return True

    async def _maintenance_operation(self, name: str, args: dict[str, Any]) -> Any:
        if name in {"archive_sessions", "prune_sessions"} and args.get("filters"):
            args = {**args, **args["filters"]}
        if name == "publish_compression_child":
            if not args["messages"]:
                raise RuntimeError("Compression child handoff must not be empty")
            async def _publish(connection):
                lock = (
                    await connection.execute(
                        _sa.select(self._tables["compression_locks"]).where(
                            self._tables["compression_locks"].c.session_id
                            == args["parent_session_id"]
                        )
                    )
                ).first()
                if args["require_compression_lease"] and (
                    lock is None
                    or lock.holder != args["compression_lock_holder"]
                    or float(lock.expires_at) <= time.time()
                ):
                    raise RuntimeError(
                        f"Compression lease lost before publication: "
                        f"{args['parent_session_id']}"
                    )
                parent = await self._session(connection, args["parent_session_id"])
                if parent is None:
                    raise RuntimeError(
                        f"Compression parent not found: {args['parent_session_id']}"
                    )
                if parent.get("ended_at") is not None:
                    raise RuntimeError(
                        f"Compression parent already ended: {args['parent_session_id']}"
                    )
                await self._create_session(
                    connection,
                    args["child_session_id"],
                    args["source"],
                    {
                        "parent_session_id": args["parent_session_id"],
                        "model": args["model"],
                        "model_config": args["model_config"],
                        "system_prompt": args["system_prompt"],
                        "cwd": args["cwd"] or parent.get("cwd"),
                        "git_branch": parent.get("git_branch"),
                        "git_repo_root": parent.get("git_repo_root"),
                        "profile_name": args["profile_name"] or parent.get("profile_name"),
                        "user_id": parent.get("user_id"),
                        "session_key": parent.get("session_key"),
                        "chat_id": parent.get("chat_id"),
                        "chat_type": parent.get("chat_type"),
                        "thread_id": parent.get("thread_id"),
                        "display_name": parent.get("display_name"),
                        "origin_json": parent.get("origin_json"),
                    },
                )
                for message in args["messages"]:
                    row = dict(message)
                    row["session_id"] = args["child_session_id"]
                    await self._append(connection, row)
                tail_ids: list[int] = []
                tail_tool_calls = 0
                if args.get("watermark") is not None:
                    tail_query = _sa.select(
                        self._tables["messages"].c.id,
                        self._tables["messages"].c.tool_calls,
                    ).where(
                        self._tables["messages"].c.session_id
                        == args["parent_session_id"],
                        self._tables["messages"].c.active == 1,
                        self._tables["messages"].c.id > int(args["watermark"]),
                    )
                    if args.get("watermark_ceiling") is not None:
                        tail_query = tail_query.where(
                            self._tables["messages"].c.id
                            <= int(args["watermark_ceiling"])
                        )
                    tail_rows = (
                        await connection.execute(tail_query.order_by(self._tables["messages"].c.id))
                    ).all()
                    tail_ids = [int(row.id) for row in tail_rows]
                    for row in tail_rows:
                        raw_tool_calls = row.tool_calls
                        if not raw_tool_calls:
                            continue
                        try:
                            parsed = (
                                json.loads(raw_tool_calls)
                                if isinstance(raw_tool_calls, str)
                                else raw_tool_calls
                            )
                        except (TypeError, ValueError):
                            continue
                        if isinstance(parsed, list):
                            tail_tool_calls += len(parsed)
                if tail_ids:
                    messages_table = self._tables["messages"]
                    clone_columns = [
                        column
                        for column in messages_table.c
                        if column.name not in {"id", "session_id", "active", "compacted"}
                    ]
                    clone_select = _sa.select(
                        *clone_columns,
                        _sa.literal(args["child_session_id"]).label("session_id"),
                        _sa.literal(1).label("active"),
                        _sa.literal(0).label("compacted"),
                    ).where(messages_table.c.id.in_(tail_ids)).order_by(messages_table.c.id)
                    await connection.execute(
                        _sa.insert(messages_table).from_select(
                            [
                                *(column.name for column in clone_columns),
                                "session_id",
                                "active",
                                "compacted",
                            ],
                            clone_select,
                        )
                    )
                    child_state = await self._session(
                        connection, args["child_session_id"]
                    )
                    await self._update_session(
                        connection,
                        args["child_session_id"],
                        {
                            "message_count": int(child_state.get("message_count") or 0)
                            + len(tail_ids),
                            "tool_call_count": int(child_state.get("tool_call_count") or 0)
                            + tail_tool_calls,
                        },
                    )
                await self._update_session(
                    connection,
                    args["parent_session_id"],
                    {"ended_at": time.time(), "end_reason": "compression"},
                )

            return await self._write(_publish)
        if name == "archive_and_compact":
            async def _archive_compact(connection):
                session = await self._session(connection, args["session_id"])
                if session is None:
                    raise RuntimeError(f"Session not found: {args['session_id']}")
                if args.get("lock_holder") is not None:
                    lock = (
                        await connection.execute(
                            _sa.select(self._tables["compression_locks"]).where(
                                self._tables["compression_locks"].c.session_id
                                == args["session_id"]
                            )
                        )
                    ).first()
                    if (
                        lock is None
                        or lock.holder != args["lock_holder"]
                        or float(lock.expires_at) <= time.time()
                    ):
                        from hermes_state import SessionCompressionInProgressError

                        raise SessionCompressionInProgressError(
                            f"Compression lease for {args['session_id']!r} lost "
                            "before commit; refusing to publish a stale compaction"
                        )
                if args.get("model_config_patch") is not None:
                    config = _json_value(session.get("model_config"), {})
                    if not isinstance(config, dict):
                        config = {}
                    for key, value in args["model_config_patch"].items():
                        if value is None:
                            config.pop(key, None)
                        else:
                            config[key] = value
                    await self._update_session(
                        connection,
                        args["session_id"],
                        {"model_config": _json_text(config)},
                    )
                messages_table = self._tables["messages"]
                tail_ids: list[int] = []
                tail_tool_calls = 0
                if args.get("watermark") is not None:
                    tail_rows = (
                        await connection.execute(
                            _sa.select(
                                messages_table.c.id,
                                messages_table.c.tool_calls,
                            )
                            .where(
                                messages_table.c.session_id == args["session_id"],
                                messages_table.c.active == 1,
                                messages_table.c.id > int(args["watermark"]),
                            )
                            .order_by(messages_table.c.id)
                        )
                    ).all()
                    for tail in tail_rows:
                        tail_ids.append(int(tail.id))
                        raw_tool_calls = tail.tool_calls
                        if raw_tool_calls:
                            try:
                                parsed = (
                                    json.loads(raw_tool_calls)
                                    if isinstance(raw_tool_calls, str)
                                    else raw_tool_calls
                                )
                                tail_tool_calls += (
                                    len(parsed) if isinstance(parsed, list) else 0
                                )
                            except (TypeError, ValueError):
                                pass
                await connection.execute(
                    _sa.update(messages_table)
                    .where(
                        messages_table.c.session_id == args["session_id"],
                        messages_table.c.active == 1,
                    )
                    .values(active=0, compacted=1)
                )
                await self._update_session(
                    connection,
                    args["session_id"],
                    {"message_count": 0, "tool_call_count": 0},
                )
                for message in args["compacted_messages"]:
                    row = dict(message)
                    row["session_id"] = args["session_id"]
                    await self._append(connection, row)
                if tail_ids:
                    clone_columns = [
                        column
                        for column in messages_table.c
                        if column.name not in {"id", "active", "compacted"}
                    ]
                    clone_select = (
                        _sa.select(
                            *clone_columns,
                            _sa.literal(1).label("active"),
                            _sa.literal(0).label("compacted"),
                        )
                        .where(messages_table.c.id.in_(tail_ids))
                        .order_by(messages_table.c.id)
                    )
                    await connection.execute(
                        _sa.insert(messages_table).from_select(
                            [
                                *(column.name for column in clone_columns),
                                "active",
                                "compacted",
                            ],
                            clone_select,
                        )
                    )
                    await self._update_session(
                        connection,
                        args["session_id"],
                        {
                            "message_count": len(args["compacted_messages"])
                            + len(tail_ids),
                            "tool_call_count": tail_tool_calls,
                        },
                    )
                return len(args["compacted_messages"]) + len(tail_ids)

            return await self._write(_archive_compact)

        if name == "maybe_auto_archive":
            result: dict[str, Any] = {"skipped": False, "archived": 0}
            try:
                last_raw = await self._dispatch("get_meta", {"key": "last_auto_archive"})
                now = time.time()
                if last_raw:
                    try:
                        if now - float(last_raw) < args["min_interval_hours"] * 3_600:
                            result["skipped"] = True
                            return result
                    except (TypeError, ValueError):
                        pass
                result["archived"] = await self._dispatch(
                    "archive_stale_sessions",
                    {
                        "idle_days": args["idle_days"],
                        "exclude_pinned": args["exclude_pinned"],
                    },
                )
                await self._dispatch(
                    "set_meta",
                    {"key": "last_auto_archive", "value": str(now)},
                )
            except Exception as exc:
                result["error"] = str(exc)
            return result

        if name == "maybe_auto_prune_and_vacuum":
            result = {"skipped": False, "pruned": 0, "vacuumed": False}
            try:
                last_raw = await self._dispatch("get_meta", {"key": "last_auto_prune"})
                now = time.time()
                if last_raw:
                    try:
                        if now - float(last_raw) < args["min_interval_hours"] * 3_600:
                            result["skipped"] = True
                            return result
                    except (TypeError, ValueError):
                        pass
                result["pruned"] = await self._dispatch(
                    "prune_sessions",
                    {
                        "older_than_days": args["retention_days"],
                        "source": None,
                        "sessions_dir": args.get("sessions_dir"),
                        "filters": {},
                    },
                )
                if result["pruned"] and args["vacuum"]:
                    result["vacuumed"] = bool(await self._dispatch("vacuum", {}))
                await self._dispatch(
                    "set_meta",
                    {"key": "last_auto_prune", "value": str(now)},
                )
            except Exception as exc:
                result["error"] = str(exc)
            return result

        async def _maintenance(connection):
            sessions = self._tables["sessions"]
            messages = self._tables["messages"]
            usage = self._tables["session_model_usage"]
            if name in {"delete_session", "delete_session_if_empty"}:
                session_row = (
                    await connection.execute(
                        _sa.select(sessions)
                        .where(sessions.c.id == args["session_id"])
                        .with_for_update()
                    )
                ).first()
                session = (
                    self._session_from_row(session_row)
                    if session_row is not None
                    else None
                )
                if session is None or (
                    name == "delete_session_if_empty" and session["message_count"]
                ):
                    return False
                all_rows = (await connection.execute(_sa.select(sessions))).all()
                decoded = [self._session_from_row(row) for row in all_rows]
                targets = [args["session_id"]]
                changed = True
                while changed:
                    changed = False
                    for item in decoded:
                        parent = item.get("parent_session_id")
                        config = _json_value(item.get("model_config"), {})
                        is_delegate = isinstance(config, dict) and config.get(
                            "_delegate_from"
                        )
                        if (
                            parent in targets
                            and is_delegate
                            and item["id"] not in targets
                        ):
                            targets.append(item["id"])
                            changed = True
                expected = args.get("expected_delete_ids")
                if expected is not None and list(expected) != targets:
                    return False
                await connection.execute(
                    _sa.delete(messages).where(
                        messages.c.session_id.in_(targets)
                    )
                )
                await connection.execute(
                    _sa.delete(usage).where(usage.c.session_id.in_(targets))
                )
                await connection.execute(
                    _sa.update(sessions)
                    .where(
                        sessions.c.parent_session_id.in_(targets),
                        ~sessions.c.id.in_(targets),
                    )
                    .values(parent_session_id=None)
                )
                await connection.execute(
                    _sa.delete(sessions).where(sessions.c.id.in_(targets))
                )
                return True
            if name == "delete_sessions":
                await connection.execute(
                    _sa.delete(messages).where(
                        messages.c.session_id.in_(args["session_ids"])
                    )
                )
                await connection.execute(
                    _sa.delete(usage).where(
                        usage.c.session_id.in_(args["session_ids"])
                    )
                )
                result = await connection.execute(
                    _sa.delete(sessions).where(sessions.c.id.in_(args["session_ids"]))
                )
                return int(result.rowcount or 0)
            if name == "archive_sessions":
                rows = (await connection.execute(_sa.select(sessions))).all()
                selected = []
                cutoff = args.get("last_active_before")
                if cutoff is None and args.get("older_than_days") is not None:
                    cutoff = time.time() - float(args["older_than_days"]) * 86_400
                filter_names = {
                    "last_active_after",
                    "started_before",
                    "started_after",
                    "title_like",
                    "end_reason",
                    "cwd_prefix",
                    "min_messages",
                    "max_messages",
                    "archived",
                    "model_like",
                    "provider",
                    "user_id",
                    "chat_id",
                    "chat_type",
                    "branch_like",
                    "min_tokens",
                    "max_tokens",
                    "min_cost",
                    "max_cost",
                    "min_tool_calls",
                    "max_tool_calls",
                }
                for row in rows:
                    item = self._session_from_row(row)
                    if self._matches_prune(
                        item,
                        source=args.get("source"),
                        last_active_before=cutoff,
                        **{
                            key: args.get(key)
                            for key in filter_names
                            if args.get(key) is not None
                        },
                    ):
                        selected.append(item["id"])
                if selected:
                    result = await connection.execute(
                        _sa.update(sessions)
                        .where(sessions.c.id.in_(selected))
                        .values(archived=1)
                    )
                    return int(result.rowcount or 0)
                return 0
            if name == "archive_stale_sessions":
                cutoff = time.time() - float(args["idle_days"]) * 86_400
                rows = (await connection.execute(_sa.select(sessions))).all()
                selected = []
                for row in rows:
                    item = self._session_from_row(row)
                    if item.get("archived") or item.get("end_reason") == "compression":
                        continue
                    if args["exclude_pinned"] and item.get("pinned"):
                        continue
                    if float(item.get("last_active") or item.get("started_at") or 0) < cutoff:
                        selected.append(item["id"])
                if selected:
                    result = await connection.execute(
                        _sa.update(sessions)
                        .where(sessions.c.id.in_(selected))
                        .values(archived=1)
                    )
                    return int(result.rowcount or 0)
                return 0
            if name == "prune_empty_ghost_sessions":
                cutoff = time.time() - 86_400
                rows = (await connection.execute(_sa.select(sessions))).all()
                ids = []
                for row in rows:
                    item = self._session_from_row(row)
                    if (
                        item.get("source") == "tui"
                        and item.get("title") is None
                        and item.get("ended_at") is not None
                        and float(item.get("started_at") or 0) < cutoff
                        and item.get("message_count", 0) == 0
                    ):
                        ids.append(item["id"])
                if ids:
                    await connection.execute(
                        _sa.delete(sessions).where(sessions.c.id.in_(ids))
                    )
                    await connection.execute(
                        _sa.delete(usage).where(usage.c.session_id.in_(ids))
                    )
                return len(ids)
            if name == "delete_empty_sessions":
                rows = (await connection.execute(_sa.select(sessions))).all()
                ids = [
                    row.id
                    for row in rows
                    if row.message_count == 0
                    and row.ended_at is not None
                    and row.archived == 0
                ]
                if ids:
                    await connection.execute(
                        _sa.delete(sessions).where(sessions.c.id.in_(ids))
                    )
                    await connection.execute(
                        _sa.delete(usage).where(usage.c.session_id.in_(ids))
                    )
                return len(ids)
            if name == "prune_sessions":
                rows = (await connection.execute(_sa.select(sessions))).all()
                cutoff = args.get("last_active_before")
                if cutoff is None and args.get("older_than_days") is not None:
                    cutoff = time.time() - float(args["older_than_days"]) * 86_400
                filter_names = {
                    "last_active_after",
                    "started_before",
                    "started_after",
                    "title_like",
                    "end_reason",
                    "cwd_prefix",
                    "min_messages",
                    "max_messages",
                    "archived",
                    "model_like",
                    "provider",
                    "user_id",
                    "chat_id",
                    "chat_type",
                    "branch_like",
                    "min_tokens",
                    "max_tokens",
                    "min_cost",
                    "max_cost",
                    "min_tool_calls",
                    "max_tool_calls",
                }
                selected = []
                for row in rows:
                    item = self._session_from_row(row)
                    if self._matches_prune(
                        item,
                        source=args.get("source"),
                        last_active_before=cutoff,
                        **{
                            key: args.get(key)
                            for key in filter_names
                            if args.get(key) is not None
                        },
                    ):
                        selected.append(item["id"])
                if not selected:
                    return 0
                await connection.execute(
                    _sa.update(sessions)
                    .where(sessions.c.parent_session_id.in_(selected))
                    .values(parent_session_id=None)
                )
                await connection.execute(
                    _sa.delete(messages).where(messages.c.session_id.in_(selected))
                )
                await connection.execute(
                    _sa.delete(usage).where(usage.c.session_id.in_(selected))
                )
                result = await connection.execute(
                    _sa.delete(sessions).where(sessions.c.id.in_(selected))
                )
                return int(result.rowcount or 0)
            if name == "finalize_orphaned_compression_sessions":
                cutoff = time.time() - 604800
                # Parent state is encoded in payload, so filter the candidate
                # rows in Python rather than interpolating JSON path syntax.
                candidates = (await connection.execute(_sa.select(sessions))).all()
                selected = []
                for row in candidates:
                    item = self._session_from_row(row)
                    if (
                        item.get("api_call_count", 0) == 0
                        and item.get("ended_at") is None
                        and (item.get("started_at") or 0) < cutoff
                        and item.get("parent_session_id")
                        and item.get("message_count", 0) > 0
                    ):
                        parent = await self._session(
                            connection, item["parent_session_id"]
                        )
                        if parent and parent.get("end_reason") == "compression":
                            selected.append(item["id"])
                if selected:
                    result = await connection.execute(
                        _sa.update(sessions)
                        .where(sessions.c.id.in_(selected))
                        .values(ended_at=time.time())
                    )
                    # end_reason lives in the retained payload.  Keep the
                    # scalar columns and JSON snapshot synchronized.
                    for session_id in selected:
                        await self._update_session(
                            connection,
                            session_id,
                            {"ended_at": time.time(), "end_reason": "orphaned_compression"},
                        )
                    return int(result.rowcount or 0)
                return 0
            if name == "reopen_orphaned_compression_session":
                session = await self._session(connection, args["session_id"])
                if not session or session.get("end_reason") != "compression":
                    return False
                children = (
                    await connection.execute(
                        _sa.select(sessions).where(
                            sessions.c.parent_session_id == args["session_id"]
                        )
                    )
                ).all()
                if children:
                    return False
                result = await connection.execute(
                        _sa.update(sessions)
                        .where(sessions.c.id == args["session_id"])
                        .values(ended_at=None)
                    )
                await self._update_session(
                    connection,
                    args["session_id"],
                    {"ended_at": None, "end_reason": None},
                )
                return bool(result.rowcount or 0)
            return {"archived": 0, "pruned": 0, "vacuumed": False}

        return await self._write(_maintenance)

    async def _record_model_usage(
        self,
        connection,
        *,
        session_id: str,
        model: str | None,
        billing_provider: str | None,
        billing_base_url: str | None,
        billing_mode: str | None,
        task: str,
        api_call_count: int,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        reasoning_tokens: int,
        estimated_cost_usd: float | None,
        actual_cost_usd: float | None,
        cost_status: str | None,
        cost_source: str | None,
    ) -> None:
        """Accumulate usage with the same composite key as SQLite."""
        effective_model = model or "unknown"
        effective_provider = billing_provider or ""
        effective_base_url = billing_base_url or ""
        effective_billing_mode = billing_mode or ""
        effective_task = task or ""
        await connection.execute(
            _sa.text(
                "SELECT pg_advisory_xact_lock("
                "hashtextextended(:session_id, 0))"
            ),
            {"session_id": session_id},
        )
        table = self._tables["session_model_usage"]
        row = (
            await connection.execute(
                _sa.select(table)
                .where(
                    table.c.session_id == session_id,
                    _sa.func.coalesce(table.c.model, "") == effective_model,
                    _sa.func.coalesce(table.c.billing_provider, "")
                    == effective_provider,
                    _sa.func.coalesce(table.c.billing_base_url, "")
                    == effective_base_url,
                    _sa.func.coalesce(table.c.billing_mode, "")
                    == effective_billing_mode,
                    table.c.task == effective_task,
                )
                .with_for_update()
            )
        ).first()
        now = time.time()
        payload = {
            "model": effective_model,
            "billing_provider": effective_provider,
            "billing_base_url": effective_base_url,
            "billing_mode": effective_billing_mode,
            "task": effective_task,
            "api_call_count": int(api_call_count or 0),
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cache_read_tokens": int(cache_read_tokens or 0),
            "cache_write_tokens": int(cache_write_tokens or 0),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "estimated_cost_usd": float(estimated_cost_usd or 0.0),
            "actual_cost_usd": float(actual_cost_usd or 0.0),
            "cost_status": cost_status,
            "cost_source": cost_source,
        }
        if row is None:
            await connection.execute(
                _sa.insert(table).values(
                    session_id=session_id,
                    model=effective_model,
                    billing_provider=effective_provider,
                    billing_base_url=effective_base_url,
                    billing_mode=effective_billing_mode,
                    task=effective_task,
                    api_call_count=payload["api_call_count"],
                    input_tokens=payload["input_tokens"],
                    output_tokens=payload["output_tokens"],
                    cache_read_tokens=payload["cache_read_tokens"],
                    cache_write_tokens=payload["cache_write_tokens"],
                    reasoning_tokens=payload["reasoning_tokens"],
                    estimated_cost_usd=payload["estimated_cost_usd"],
                    actual_cost_usd=payload["actual_cost_usd"],
                    cost_status=cost_status,
                    cost_source=cost_source,
                    first_seen=now,
                    last_seen=now,
                    payload=_json_text(payload),
                )
            )
            return
        merged = {
            key: int(getattr(row, key) or 0) + payload[key]
            for key in (
                "api_call_count",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
            )
        }
        merged["estimated_cost_usd"] = float(
            getattr(row, "estimated_cost_usd") or 0.0
        ) + payload["estimated_cost_usd"]
        merged["actual_cost_usd"] = float(
            getattr(row, "actual_cost_usd") or 0.0
        ) + payload["actual_cost_usd"]
        merged_payload = _json_value(getattr(row, "payload", None), {})
        if not isinstance(merged_payload, dict):
            merged_payload = {}
        merged_payload.update(payload)
        await connection.execute(
            _sa.update(table)
            .where(table.c.id == row.id)
            .values(
                **merged,
                cost_status=cost_status or row.cost_status,
                cost_source=cost_source or row.cost_source,
                last_seen=now,
                payload=_json_text(merged_payload),
            )
        )

    async def _usage_operation(self, args: dict[str, Any]) -> None:
        if not args.get("session_id") or not args.get("task"):
            return None

        async def _record(connection):
            if await self._session(connection, args["session_id"]) is None:
                await self._create_session(
                    connection,
                    args["session_id"],
                    "unknown",
                    {"model": args.get("model")},
                )
            await self._record_model_usage(
                connection,
                session_id=args["session_id"],
                model=args.get("model"),
                billing_provider=args.get("billing_provider"),
                billing_base_url=args.get("billing_base_url"),
                billing_mode=None,
                task=args["task"],
                api_call_count=1,
                input_tokens=args.get("input_tokens", 0),
                output_tokens=args.get("output_tokens", 0),
                cache_read_tokens=args.get("cache_read_tokens", 0),
                cache_write_tokens=args.get("cache_write_tokens", 0),
                reasoning_tokens=args.get("reasoning_tokens", 0),
                estimated_cost_usd=args.get("estimated_cost_usd"),
                actual_cost_usd=None,
                cost_status=None,
                cost_source=None,
            )

        await self._write(_record)

    async def _maintenance_read(self, name: str, args: dict[str, Any]) -> Any:
        if name in {"fts_cjk_rebuild_step", "fts_rebuild_step"}:
            return False
        if name == "fts_optimize_available":
            return False
        if name in {"fts_cjk_rebuild_status", "fts_rebuild_status"}:
            return None
        if name == "optimize_fts":
            if self._read_only:
                return 0
            await self._execute_autocommit("ANALYZE messages")
            return 1
        if name == "rebuild_fts":
            if self._read_only:
                return 0
            await self._execute_autocommit(
                "REINDEX INDEX messages_hermes_search_idx"
            )
            return 1
        if name == "vacuum":
            if self._read_only:
                return 0
            optimized = await self._maintenance_read("optimize_fts", args)
            await self._execute_autocommit("VACUUM (ANALYZE)")
            return optimized
        if name == "optimize_fts_storage":
            if self._read_only:
                return {"ok": False, "reason": "read_only"}
            await self._maintenance_read("rebuild_fts", args)
            vacuumed = False
            if args.get("vacuum", True):
                await self._execute_autocommit("VACUUM (ANALYZE)")
                vacuumed = True
            progress_cb = args.get("progress_cb")
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "done",
                        "percent": 100,
                        "indexed": 0,
                        "total": 0,
                    }
                )
            return {"ok": True, "vacuumed": vacuumed}
        if name == "logical_size_bytes":
            async def _size(connection):
                row = (
                    await connection.execute(
                        _sa.text(
                            "SELECT COALESCE(sum(pg_total_relation_size(quote_ident(tablename))), 0) "
                            "FROM pg_tables WHERE schemaname = current_schema() "
                            "AND tablename IN ('sessions','messages',"
                            "'session_model_usage','state_meta','compression_locks')"
                        )
                    )
                ).first()
                return int(row[0] or 0) if row else 0

            return await self._read(_size)
        if name == "get_active_message_watermark":
            async def _watermark(connection):
                table = self._tables["messages"]
                row = (
                    await connection.execute(
                        _sa.select(_sa.func.coalesce(_sa.func.max(table.c.id), 0))
                        .where(
                            table.c.session_id == args["session_id"],
                            table.c.active == 1,
                        )
                    )
                ).first()
                return int(row[0] or 0) if row else 0

            return await self._read(_watermark)
        if name == "flush_token_counts":
            return True
        if name == "session_count_ge":
            count = await self._count_operation("session_count", {})
            return count >= args["n"]
        if name in {"assert_export_safe", "assert_resume_safe"}:
            count = await self._dispatch(
                "message_count",
                {"session_id": args["session_id"]},
            )
            if args["max_messages"] is not None and count > args["max_messages"]:
                raise ValueError(f"session has too many messages: {count}")
            return count
        if name == "get_compression_failure_cooldown":
            session = await self._dispatch(
                "get_session", {"session_id": args["session_id"]}
            )
            if not session or session.get("compression_failure_cooldown_until") is None:
                return None
            if float(session["compression_failure_cooldown_until"]) <= time.time():
                return None
            cooldown_until = float(session["compression_failure_cooldown_until"])
            return {
                "cooldown_until": cooldown_until,
                "remaining_seconds": cooldown_until - time.time(),
                "error": session.get("compression_failure_error"),
            }
        if name == "get_compression_failure_cooldown_row":
            session = await self._dispatch(
                "get_session", {"session_id": args["session_id"]}
            )
            if not session:
                return {
                    "session_exists": False,
                    "cooldown_until": None,
                    "error": None,
                }
            deadline = session.get("compression_failure_cooldown_until")
            return {
                "session_exists": True,
                "cooldown_until": float(deadline) if deadline is not None else None,
                "error": session.get("compression_failure_error"),
            }
        if name in {"get_compression_fallback_streak", "get_compression_ineffective_count"}:
            session = await self._dispatch(
                "get_session", {"session_id": args["session_id"]}
            )
            key = (
                "compression_fallback_streak"
                if name == "get_compression_fallback_streak"
                else "compression_ineffective_count"
            )
            return int((session or {}).get(key) or 0)
        if name == "find_live_compression_child":
            rows = await self._collection_operation(
                "list_sessions_rich",
                {
                    "include_archived": True,
                    "archived_only": False,
                    "source": None,
                    "sources": None,
                    "exclude_sources": None,
                    "cwd_prefix": None,
                    "limit": 10000,
                    "offset": 0,
                    "include_children": True,
                    "min_message_count": 0,
                    "project_compression_tips": True,
                    "order_by_last_active": False,
                    "id_query": None,
                    "search_query": None,
                    "compact_rows": False,
                    "include_pinned": True,
                    "session_key": None,
                },
            )
            return next(
                (
                    row
                    for row in rows
                    if row.get("parent_session_id") == args["parent_session_id"]
                    and row.get("ended_at") is None
                ),
                None,
            )
        return None

    async def _fallback_operation(self, name: str, args: dict[str, Any]) -> Any:
        del args
        raise RuntimeError(
            f"PostgreSQL SessionDB operation is not implemented: {name}"
        )

    async def append_message(self, session_id: str, role: str, content: str | None = None, tool_name: str | None = None, tool_calls: Any = None, tool_call_id: str | None = None, token_count: int | None = None, finish_reason: str | None = None, reasoning: str | None = None, reasoning_content: str | None = None, reasoning_details: Any = None, codex_reasoning_items: Any = None, codex_message_items: Any = None, platform_message_id: str | None = None, observed: bool = False, effect_disposition: str | None = None, timestamp: Any = None, api_content: str | None = None, display_kind: str | None = None, display_metadata: dict[str, typing.Any] | None = None, compression_lock_holder: str | None = None, turn_lease_holder: str | None = None, turn_lease_ttl_seconds: float = 300.0) -> int:
        return await self._dispatch('append_message', locals())

    async def append_messages_batch(self, session_id: str, messages: list[dict[str, typing.Any]], compression_lock_holder: str | None = None, turn_lease_holder: str | None = None, chunk_rows: int | None = None, turn_lease_ttl_seconds: float = 300.0) -> int:
        return await self._dispatch('append_messages_batch', locals())

    async def get_active_message_watermark(self, session_id: str) -> int:
        return await self._dispatch('get_active_message_watermark', locals())

    async def archive_and_compact(self, session_id: str, compacted_messages: list[dict[str, typing.Any]], model_config_patch: dict[str, typing.Any] | None = None, watermark: int | None = None, lock_holder: str | None = None) -> int:
        return await self._dispatch('archive_and_compact', locals())

    async def archive_sessions(self, older_than_days: float | None = None, source: str | None = None, **filters) -> int:
        return await self._dispatch('archive_sessions', locals())

    async def archive_stale_sessions(self, idle_days: float, *, exclude_pinned: bool = True) -> int:
        return await self._dispatch('archive_stale_sessions', locals())

    async def assert_export_safe(self, session_id: str, max_messages: int | None = None) -> int:
        return await self._dispatch('assert_export_safe', locals())

    async def assert_resume_safe(self, session_id: str, max_messages: int | None = None) -> int:
        return await self._dispatch('assert_resume_safe', locals())

    async def backfill_repo_roots(self, cwd_to_root: dict[str, str]) -> None:
        return await self._dispatch('backfill_repo_roots', locals())

    async def clear_compression_failure_cooldown(self, session_id: str) -> None:
        return await self._dispatch('clear_compression_failure_cooldown', locals())

    async def clear_messages(self, session_id: str) -> None:
        return await self._dispatch('clear_messages', locals())

    async def clear_session_activity_labels(self, session_id: str) -> None:
        return await self._dispatch('clear_session_activity_labels', locals())

    async def close(self) -> None:
        return await self._dispatch('close', locals())

    async def count_empty_sessions(self) -> int:
        return await self._dispatch('count_empty_sessions', locals())

    async def create_session(self, session_id: str, source: str, **kwargs) -> str:
        return await self._dispatch('create_session', locals())

    async def delete_empty_sessions(self, sessions_dir: pathlib.Path | None = None) -> int:
        return await self._dispatch('delete_empty_sessions', locals())

    async def delete_session(self, session_id: str, sessions_dir: pathlib.Path | None = None, expected_delete_ids: list[str] | None = None) -> bool:
        return await self._dispatch('delete_session', locals())

    async def delete_session_if_empty(self, session_id: str, sessions_dir: pathlib.Path | None = None) -> bool:
        return await self._dispatch('delete_session_if_empty', locals())

    async def delete_sessions(self, session_ids: list[str], sessions_dir: pathlib.Path | None = None) -> int:
        return await self._dispatch('delete_sessions', locals())

    async def distinct_session_cwds(self, include_archived: bool = False) -> list[dict[str, Any]]:
        return await self._dispatch('distinct_session_cwds', locals())

    async def end_session(self, session_id: str, end_reason: str) -> None:
        return await self._dispatch('end_session', locals())

    async def ensure_session(self, session_id: str, source: str = 'unknown', model: str | None = None, **kwargs) -> str:
        return await self._dispatch('ensure_session', locals())

    async def export_all(self, source: str | None = None) -> list[dict[str, Any]]:
        return await self._dispatch('export_all', locals())

    async def export_session(self, session_id: str) -> dict[str, Any] | None:
        return await self._dispatch('export_session', locals())

    async def export_session_lineage(self, session_id: str) -> dict[str, Any] | None:
        return await self._dispatch('export_session_lineage', locals())

    async def finalize_orphaned_compression_sessions(self) -> int:
        return await self._dispatch('finalize_orphaned_compression_sessions', locals())

    async def find_live_compression_child(self, parent_session_id: str) -> dict[str, typing.Any] | None:
        return await self._dispatch('find_live_compression_child', locals())

    async def flush_token_counts(self, timeout: float = 5.0) -> bool:  # noqa: ASYNC109 - retained upstream signature
        return await self._dispatch('flush_token_counts', locals())

    async def fts_cjk_rebuild_status(self) -> dict[str, Any] | None:
        return await self._dispatch('fts_cjk_rebuild_status', locals())

    async def fts_cjk_rebuild_step(self) -> bool:
        return await self._dispatch('fts_cjk_rebuild_step', locals())

    async def fts_optimize_available(self) -> bool:
        return await self._dispatch('fts_optimize_available', locals())

    async def fts_rebuild_status(self) -> dict[str, Any] | None:
        return await self._dispatch('fts_rebuild_status', locals())

    async def fts_rebuild_step(self) -> bool:
        return await self._dispatch('fts_rebuild_step', locals())

    async def get_ancestor_display_prefix(self, session_id: str) -> list[dict[str, typing.Any]]:
        return await self._dispatch('get_ancestor_display_prefix', locals())

    async def get_anchored_view(self, session_id: str, around_message_id: int, window: int = 5, bookend: int = 3, keep_roles: tuple[str, ...] | None = ('user', 'assistant')) -> dict[str, Any]:
        return await self._dispatch('get_anchored_view', locals())

    async def get_compression_failure_cooldown(self, session_id: str) -> dict[str, typing.Any] | None:
        return await self._dispatch('get_compression_failure_cooldown', locals())

    async def get_compression_failure_cooldown_row(self, session_id: str) -> dict[str, typing.Any]:
        return await self._dispatch('get_compression_failure_cooldown_row', locals())

    async def get_compression_fallback_streak(self, session_id: str) -> int:
        return await self._dispatch('get_compression_fallback_streak', locals())

    async def get_compression_ineffective_count(self, session_id: str) -> int:
        return await self._dispatch('get_compression_ineffective_count', locals())

    async def get_compression_lineage(self, session_id: str) -> list[str]:
        return await self._dispatch('get_compression_lineage', locals())

    async def get_compression_lock_holder(self, session_id: str) -> str | None:
        return await self._dispatch('get_compression_lock_holder', locals())

    async def get_compression_tip(self, session_id: str) -> str | None:
        return await self._dispatch('get_compression_tip', locals())

    async def get_conversation_root(self, session_id: str) -> str:
        return await self._dispatch('get_conversation_root', locals())

    async def get_first_assistant_text(self, session_id: str) -> str:
        return await self._dispatch('get_first_assistant_text', locals())

    async def get_message_role(self, session_id: str, row_id: int) -> str | None:
        return await self._dispatch('get_message_role', locals())

    async def get_messages(self, session_id: str, include_inactive: bool = False, limit: int | None = None, offset: int = 0, latest: bool = False, after_id: int | None = None) -> list[dict[str, typing.Any]]:
        return await self._dispatch('get_messages', locals())

    async def get_messages_around(self, session_id: str, around_message_id: int, window: int = 5) -> dict[str, typing.Any]:
        return await self._dispatch('get_messages_around', locals())

    async def get_messages_as_conversation(self, session_id: str, include_ancestors: bool = False, include_inactive: bool = False, repair_alternation: bool = False, include_row_ids: bool = False) -> list[dict[str, typing.Any]]:
        return await self._dispatch('get_messages_as_conversation', locals())

    async def get_meta(self, key: str) -> str | None:
        return await self._dispatch('get_meta', locals())

    async def get_next_title_in_lineage(self, base_title: str) -> str:
        return await self._dispatch('get_next_title_in_lineage', locals())

    async def get_resume_conversations(self, session_id: str) -> tuple[list[dict[str, typing.Any]], list[dict[str, typing.Any]]]:
        return await self._dispatch('get_resume_conversations', locals())

    async def get_resume_message_count(self, session_id: str) -> int:
        return await self._dispatch('get_resume_message_count', locals())

    async def get_session(self, session_id: str) -> dict[str, typing.Any] | None:
        return await self._dispatch('get_session', locals())

    async def get_session_activity(self, session_id: str) -> dict[str, typing.Any] | None:
        return await self._dispatch('get_session_activity', locals())

    async def get_session_by_title(self, title: str) -> dict[str, typing.Any] | None:
        return await self._dispatch('get_session_by_title', locals())

    async def get_session_delete_targets(self, session_id: str) -> list[str]:
        return await self._dispatch('get_session_delete_targets', locals())

    async def get_session_model_config_value(self, session_id: str, key: str, default: Any = None) -> Any:
        return await self._dispatch('get_session_model_config_value', locals())

    async def get_session_rich_row(self, session_id: str, compact_rows: bool = False) -> dict[str, Any] | None:
        return await self._dispatch('get_session_rich_row', locals())

    async def get_session_title(self, session_id: str) -> str | None:
        return await self._dispatch('get_session_title', locals())

    async def get_session_title_source(self, session_id: str) -> str | None:
        return await self._dispatch('get_session_title_source', locals())

    async def has_archived_messages(self, session_id: str) -> bool:
        return await self._dispatch('has_archived_messages', locals())

    async def has_platform_message_id(self, session_id: str, platform_message_id: str) -> bool:
        return await self._dispatch('has_platform_message_id', locals())

    async def import_sessions(self, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._dispatch('import_sessions', locals())

    async def latest_message_row_id(self, session_id: str, *, role: str = 'user', offset: int = 0, require_text: bool = True) -> int | None:
        return await self._dispatch('latest_message_row_id', locals())

    async def latest_user_message_row_id(self, session_id: str) -> int | None:
        return await self._dispatch('latest_user_message_row_id', locals())

    async def list_prune_candidates(self, older_than_days: float | None = None, source: str | None = None, **filters) -> list[dict[str, typing.Any]]:
        return await self._dispatch('list_prune_candidates', locals())

    async def count_open_prune_matches(self, older_than_days: float | None = None, source: str | None = None, **filters) -> int:
        return await self._dispatch('count_open_prune_matches', locals())

    async def list_recent_user_messages(self, session_id: str, limit: int = 20, include_inactive: bool = False) -> list[dict[str, Any]]:
        return await self._dispatch('list_recent_user_messages', locals())

    async def list_sessions_rich(self, source: str | None = None, sources: list[str] | None = None, exclude_sources: list[str] | None = None, cwd_prefix: str | None = None, limit: int = 20, offset: int = 0, include_children: bool = False, min_message_count: int = 0, project_compression_tips: bool = True, order_by_last_active: bool = False, include_archived: bool = False, archived_only: bool = False, id_query: str | None = None, search_query: str | None = None, compact_rows: bool = False, include_pinned: bool = False, session_key: str | None = None, include_hidden: bool = False) -> list[dict[str, typing.Any]]:
        return await self._dispatch('list_sessions_rich', locals())

    async def list_skill_scaffolded_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        return await self._dispatch('list_skill_scaffolded_sessions', locals())

    async def logical_size_bytes(self) -> int | None:
        return await self._dispatch('logical_size_bytes', locals())

    async def maybe_auto_archive(self, idle_days: float = 3, min_interval_hours: int = 24, exclude_pinned: bool = True) -> dict[str, typing.Any]:
        return await self._dispatch('maybe_auto_archive', locals())

    async def maybe_auto_prune_and_vacuum(self, retention_days: int = 90, min_interval_hours: int = 24, vacuum: bool = True, sessions_dir: pathlib.Path | None = None, min_vacuum_interval_days: int = 30) -> dict[str, typing.Any]:
        return await self._dispatch('maybe_auto_prune_and_vacuum', locals())

    async def message_count(self, session_id: str | None = None) -> int:
        return await self._dispatch('message_count', locals())

    async def optimize_fts(self) -> int:
        return await self._dispatch('optimize_fts', locals())

    async def optimize_fts_storage(self, *, progress_cb: Callable[[dict[str, Any]], None] | None = None, vacuum: bool = True) -> dict[str, Any]:
        return await self._dispatch('optimize_fts_storage', locals())

    async def patch_session_model_config(self, session_id: str, patch: dict[str, typing.Any]) -> None:
        return await self._dispatch('patch_session_model_config', locals())

    async def prune_empty_ghost_sessions(self, sessions_dir: Path | None = None) -> int:
        return await self._dispatch('prune_empty_ghost_sessions', locals())

    async def prune_sessions(self, older_than_days: float | None = 90, source: str | None = None, sessions_dir: pathlib.Path | None = None, **filters) -> int:
        return await self._dispatch('prune_sessions', locals())

    async def publish_compression_child(self, *, parent_session_id: str, child_session_id: str, source: str, messages: list[dict[str, typing.Any]], model: str | None = None, model_config: dict[str, typing.Any] | None = None, system_prompt: str | None = None, cwd: str | None = None, profile_name: str | None = None, compression_lock_holder: str | None = None, require_compression_lease: bool = True, watermark: int | None = None, watermark_ceiling: int | None = None) -> None:
        return await self._dispatch('publish_compression_child', locals())

    async def queue_token_counts(self, session_id: str, **kwargs) -> None:
        return await self._dispatch('queue_token_counts', locals())

    async def rebuild_fts(self) -> int:
        return await self._dispatch('rebuild_fts', locals())

    async def record_auxiliary_usage(self, session_id: str, task: str, *, model: str | None = None, billing_provider: str | None = None, billing_base_url: str | None = None, input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0, estimated_cost_usd: float | None = None, api_call_count: int = 1) -> None:
        return await self._dispatch('record_auxiliary_usage', locals())

    async def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: str | None = None) -> None:
        return await self._dispatch('record_compression_failure_cooldown', locals())

    async def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return await self._dispatch('refresh_compression_lock', locals())

    async def try_acquire_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
        patience_s: float | None = None,
    ) -> bool:
        return await self._dispatch('try_acquire_session_turn_lease', locals())

    async def acquire_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
        wait_seconds: float = 1800.0,
        poll_interval_seconds: float = 1.0,
        on_wait: Callable[[float], None] | None = None,
        wait_notice_interval_seconds: float = 15.0,
        should_abort: Callable[[], bool] | None = None,
        acquire_patience_s: float = 0.5,
    ) -> bool:
        return await self._dispatch('acquire_session_turn_lease', locals())

    async def refresh_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
    ) -> bool:
        return await self._dispatch('refresh_session_turn_lease', locals())

    async def release_session_turn_lease(self, session_id: str, holder: str) -> None:
        return await self._dispatch('release_session_turn_lease', locals())

    async def release_compression_lock(self, session_id: str, holder: str) -> None:
        return await self._dispatch('release_compression_lock', locals())

    async def reopen_orphaned_compression_session(self, session_id: str) -> bool:
        return await self._dispatch('reopen_orphaned_compression_session', locals())

    async def reopen_session(self, session_id: str) -> None:
        return await self._dispatch('reopen_session', locals())

    async def replace_messages(self, session_id: str, messages: list[dict[str, typing.Any]], active_only: bool = False, archive_dropped: bool = False) -> None:
        return await self._dispatch('replace_messages', locals())

    async def resolve_resume_session_id(self, session_id: str) -> str:
        return await self._dispatch('resolve_resume_session_id', locals())

    async def resolve_session_by_title(self, title: str) -> str | None:
        return await self._dispatch('resolve_session_by_title', locals())

    async def resolve_session_id(self, session_id_or_prefix: str) -> str | None:
        return await self._dispatch('resolve_session_id', locals())

    async def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: dict[str, typing.Any]) -> None:
        return await self._dispatch('restore_compression_failure_cooldown_row', locals())

    async def restore_rewound(self, session_id: str, since_message_id: int) -> int:
        return await self._dispatch('restore_rewound', locals())

    async def rewind_to_message(self, session_id: str, target_message_id: int) -> dict[str, typing.Any]:
        return await self._dispatch('rewind_to_message', locals())

    async def search_messages(self, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None, role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None, include_inactive: bool = False, fields: Collection[str] | None = None) -> list[dict[str, Any]]:
        return await self._dispatch('search_messages', locals())

    async def search_sessions(self, source: str | None = None, limit: int = 20, offset: int = 0, workspace_key: str | None = None) -> list[dict[str, typing.Any]]:
        return await self._dispatch('search_sessions', locals())

    async def search_sessions_by_id(self, query: str, limit: int = 20, include_archived: bool = True, source: str | None = None, sources: list[str] | None = None, exclude_sources: list[str] | None = None) -> list[dict[str, Any]]:
        return await self._dispatch('search_sessions_by_id', locals())

    async def session_count(self, source: str | None = None, sources: list[str] | None = None, cwd_prefix: str | None = None, min_message_count: int = 0, include_archived: bool = False, archived_only: bool = False, exclude_children: bool = False, exclude_sources: list[str] | None = None) -> int:
        return await self._dispatch('session_count', locals())

    async def session_count_by_source(self, *, include_archived: bool = False, archived_only: bool = False, exclude_children: bool = False) -> dict[str, int]:
        return await self._dispatch('session_count_by_source', locals())

    async def session_count_ge(self, n: int = 1) -> bool:
        return await self._dispatch('session_count_ge', locals())

    async def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        return await self._dispatch('set_auto_title', locals())

    async def set_auto_title_if_empty(self, session_id: str, title: str) -> bool:
        return await self._dispatch('set_auto_title_if_empty', locals())

    async def set_compression_fallback_streak(self, session_id: str, streak: int) -> None:
        return await self._dispatch('set_compression_fallback_streak', locals())

    async def set_compression_ineffective_count(self, session_id: str, count: int) -> None:
        return await self._dispatch('set_compression_ineffective_count', locals())

    async def set_latest_user_api_content(self, session_id: str, content: Any, api_content: str) -> int:
        return await self._dispatch('set_latest_user_api_content', locals())

    async def set_meta(self, key: str, value: str, *, cursor: sqlite3.Cursor | None = None) -> None:
        return await self._dispatch('set_meta', locals())

    async def set_session_archived(self, session_id: str, archived: bool) -> bool:
        return await self._dispatch('set_session_archived', locals())

    async def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        return await self._dispatch('set_session_pinned', locals())

    async def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        return await self._dispatch('set_session_hidden', locals())

    async def set_session_read(self, session_id: str, read: bool = True) -> bool:
        return await self._dispatch('set_session_read', locals())

    async def set_session_title(self, session_id: str, title: str) -> bool:
        return await self._dispatch('set_session_title', locals())

    async def set_session_title_source(self, session_id: str, source: str) -> bool:
        return await self._dispatch('set_session_title_source', locals())

    async def touch_session_activity(self, session_id: str, ts: float | None = None, *, description: str | None = None, provenance: ActivityProvenance | None = None) -> None:
        return await self._dispatch('touch_session_activity', locals())

    async def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return await self._dispatch('try_acquire_compression_lock', locals())

    async def update_session_billing_route(self, session_id: str, *, provider: str, base_url: str, billing_mode: str | None = None) -> None:
        return await self._dispatch('update_session_billing_route', locals())

    async def update_session_cwd(self, session_id: str, cwd: str, git_branch: str | None = None, git_repo_root: str | None = None, replace_git_meta: bool = False) -> None:
        return await self._dispatch('update_session_cwd', locals())

    async def update_session_meta(self, session_id: str, model_config_json: str, model: str | None = None) -> None:
        return await self._dispatch('update_session_meta', locals())

    async def update_session_model(self, session_id: str, model: str) -> None:
        return await self._dispatch('update_session_model', locals())

    async def update_session_runtime_lock(self, session_id: str, *, model: str | None = None, provider: str | None = None, model_options: dict[str, typing.Any] | None = None, route_source: str | None = None, confirmed: bool = False) -> None:
        return await self._dispatch('update_session_runtime_lock', locals())

    async def update_system_prompt(self, session_id: str, system_prompt: str | None) -> None:
        return await self._dispatch('update_system_prompt', locals())

    async def update_token_counts(self, session_id: str, input_tokens: int = 0, output_tokens: int = 0, model: str | None = None, cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0, estimated_cost_usd: float | None = None, actual_cost_usd: float | None = None, cost_status: str | None = None, cost_source: str | None = None, pricing_version: str | None = None, billing_provider: str | None = None, billing_base_url: str | None = None, billing_mode: str | None = None, api_call_count: int = 0, absolute: bool = False) -> None:
        return await self._dispatch('update_token_counts', locals())

    async def vacuum(self) -> int:
        return await self._dispatch('vacuum', locals())


    async def _close(self) -> None:
        if self._close_task is None:
            async def _dispose() -> None:
                async with self._ready_lock:
                    if self._engine is not None:
                        await self._engine.dispose()
                    self._engine = None
                    self._tables = {}
                    self._ready = False
                    self._closed = True

            self._close_task = asyncio.create_task(_dispose())
        first_cancel: asyncio.CancelledError | None = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
                first_cancel = first_cancel or exc
        await self._close_task
        if first_cancel is not None:
            raise first_cancel

    @staticmethod
    def sanitize_title(title: str | None) -> str | None:
        if not title:
            return None
        title = _scrub_surrogates(title)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", title)
        cleaned = re.sub(
            r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )
        return cleaned

    @staticmethod
    def session_unread(session_row: dict[str, Any]) -> bool:
        last_read = session_row.get("last_read_at")
        if last_read is None:
            return False
        last_active = session_row.get("last_active") or session_row.get("started_at")
        return float(last_active or 0) > float(last_read)
