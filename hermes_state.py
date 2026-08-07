#!/usr/bin/env python3
"""Native-async SQLite session state for Hermes Agent."""

import asyncio
import errno
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles.os

from agent.memory_manager import sanitize_context
from agent.message_sanitization import _sanitize_surrogates
from hermes_constants import get_hermes_home
from hermes_state_common import (
    DEFERRED_INDEX_SQL,
    MAX_FTS5_QUERY_CHARS,
    SCHEMA_VERSION,
    SCHEMA_SQL,
    _LISTABLE_CHILD_SQL,
    _shape_preview,
)

try:
    import psutil
except ImportError:  # pragma: no cover - minimal installs
    psutil = None

logger = logging.getLogger(__name__)
_COMPRESSION_LOCK_HOLDER_PID_RE = re.compile(r"(?:^|:)pid=(\d+)(?::|$)")

_DISK_FULL_MARKERS = (
    "no space left on device",
    "not enough space",
    "database or disk is full",
    "disk full",
    "full disk",
    "enospc",
)


def _system_prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def is_disk_full_error(exc: BaseException | str | None) -> bool:
    """Return whether an error reports ENOSPC or SQLite's disk-full state."""
    if exc is None:
        return False
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    text = exc if isinstance(exc, str) else str(exc)
    return any(marker in text.lower() for marker in _DISK_FULL_MARKERS)


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """Return true only when a structured local compression owner is gone."""
    match = _COMPRESSION_LOCK_HOLDER_PID_RE.search(holder or "")
    if match is None:
        return False
    try:
        pid = int(match.group(1))
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    if psutil is not None:
        try:
            return not psutil.pid_exists(pid)
        except Exception:
            return False
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError, OverflowError):
        return False
    return False


def _scrub_surrogates(value: Any) -> Any:
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def _delegate_from_json(column: str = "model_config") -> str:
    return f"json_extract(COALESCE({column}, '{{}}'), '$._delegate_from')"


async def _collect_delegate_child_ids(connection, parent_ids: List[str]) -> List[str]:
    """Return recursively discovered delegate children, excluding parents."""
    delegate_from = _delegate_from_json()
    seeds = {session_id for session_id in parent_ids if session_id}
    found = set(seeds)
    frontier = list(seeds)
    while frontier:
        placeholders = ",".join("?" for _ in frontier)
        rows = await (
            await connection.execute(
                f"SELECT id FROM sessions WHERE {delegate_from} IN ({placeholders}) "
                f"OR (parent_session_id IN ({placeholders}) "
                f"AND {delegate_from} IS NOT NULL)",
                [*frontier, *frontier],
            )
        ).fetchall()
        frontier = [row["id"] for row in rows if row["id"] not in found]
        found.update(frontier)
    return [session_id for session_id in found if session_id not in seeds]


async def _delete_delegate_children(connection, parent_ids: List[str]) -> List[str]:
    session_ids = await _collect_delegate_child_ids(connection, parent_ids)
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        await connection.execute(
            f"DELETE FROM messages WHERE session_id IN ({placeholders})",
            session_ids,
        )
        await connection.execute(
            f"UPDATE sessions SET parent_session_id = NULL "
            f"WHERE parent_session_id IN ({placeholders})",
            session_ids,
        )
        await connection.execute(
            f"DELETE FROM sessions WHERE id IN ({placeholders})",
            session_ids,
        )
    return session_ids


def _cwd_prefix_clause(cwd_prefix: str) -> Tuple[str, List[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    return "(s.cwd = ? OR s.cwd LIKE ? OR s.cwd LIKE ?)", [
        prefix,
        f"{prefix}/%",
        f"{prefix}\\%",
    ]


DEFAULT_DB_PATH = get_hermes_home() / "state.db"
_IMPORT_DEFAULT_DB_PATH = DEFAULT_DB_PATH


def _default_db_path() -> Path:
    if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH
    return get_hermes_home() / "state.db"


_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(message: Dict[str, Any]) -> bool:
    if not isinstance(message, dict) or message.get("role") not in {"user", "system"}:
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    return any(content.lstrip().startswith(prefix) for prefix in _REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not messages:
        return messages
    result: List[Dict[str, Any]] = []
    skip_next_assistant = False
    for message in messages:
        if _is_background_review_harness_message(message):
            skip_next_assistant = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if isinstance(message, dict) and message.get("role") == "assistant":
                continue
        result.append(message)
    return result


class CompressionSessionClosedError(RuntimeError):
    """A durable write targeted a parent already closed by compression."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} is closed by compression; "
            "adopt its live continuation before appending messages"
        )


class CompressionSessionBusyError(RuntimeError):
    """A non-owner tried to write while compression owns the session."""


class SessionCompressionInProgressError(CompressionSessionBusyError):
    """A normal writer collided with another writer's live compression lease."""


class SessionDB:
    """Native-async session store used by the agent turn path.

    The constructor accepts a database path. The SQLite connection and schema
    are created lazily on the first awaited operation, keeping
    ``AIAgent.__init__`` state-only.

    Deliberately do not provide a generic ``__getattr__`` bridge: an unknown
    synchronous database method must fail loudly instead of silently moving
    work into a thread pool.
    """

    _WRITE_PATIENCE_S = 60.0
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S = 0.02
    _WRITE_RETRY_MAX_S = 0.15

    _CONTENT_JSON_PREFIX = "\x00json:"
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, timestamp, "
        "api_content, display_kind, display_metadata"
    )

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        err = str(exc).lower()
        if "no such module" in err and "fts5" in err:
            return True
        # SQLite builds that have FTS5 but lack the optional trigram tokenizer
        # raise "no such tokenizer: trigram" instead of "no such module".
        # Scope to trigram specifically to avoid masking unrelated tokenizer errors.
        if "no such tokenizer: trigram" in err:
            return True
        # The cjk_unicode61 tokenizer is a loadable extension — a process
        # that couldn't load it sees the same capability-error shape.
        if "no such tokenizer: cjk_unicode61" in err:
            return True
        return False


    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize structured (list/dict) message content for sqlite.

        sqlite3 can only bind ``str``, ``bytes``, ``int``, ``float``, and ``None``
        to query parameters. Multimodal messages have ``content`` as a list of
        parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``), which
        raises ``ProgrammingError: Error binding parameter N: type 'list' is
        not supported`` when bound directly.

        Returns the value unchanged when it's already a safe scalar, or a
        sentinel-prefixed JSON string for lists/dicts. Paired with
        :meth:`_decode_content` on read.
        """
        if isinstance(content, str):
            # Lone UTF-16 surrogates reach here inside tool results scraped
            # from the web/social platforms (the same input that crashed the
            # guardrail hasher). The proactive sanitizer upstream only cleans
            # the *api_messages* copy, and the recovery sanitizer only runs
            # after the API call itself raises — which it no longer does — so
            # the canonical history keeps them and this write is where they
            # land. Left raw, sqlite3 raises UnicodeEncodeError, the flush is
            # abandoned, and the session silently stops persisting for the
            # rest of its life. Scrub so persistence never fails.
            return _sanitize_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            # json.dumps defaults to ensure_ascii=True, which escapes any
            # surrogate as \udXXX — already safe to bind.
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            # Last-resort fallback: stringify so persistence never fails.
            return _sanitize_surrogates(str(content))


    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to decode JSON-encoded message content; "
                    "returning raw string"
                )
                return content
        return content


    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """Serialize ``display_metadata`` for its TEXT column without double-encoding.

        Import/replace paths can hand us an already-serialized JSON string (the
        same hazard ``tool_calls`` guards against above). ``json.dumps`` on that
        string would store a quoted JSON string, and the single ``json.loads``
        on read then yields a ``str`` instead of a dict.
        """
        if not display_metadata:
            return None
        if isinstance(display_metadata, str):
            try:
                parsed = json.loads(display_metadata)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Ignoring non-JSON display metadata on write")
                return None
            if not isinstance(parsed, dict):
                logger.warning("Ignoring non-object display metadata on write")
                return None
            return json.dumps(parsed)
        if isinstance(display_metadata, dict):
            return json.dumps(display_metadata)
        logger.warning(
            "Ignoring unexpected display metadata type on write: %s",
            type(display_metadata).__name__,
        )
        return None


    @staticmethod
    def _decode_display_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Decode a ``display_metadata`` column into the dict every reader expects.

        Every message read path must go through this. Returning the raw TEXT
        instead reaches the desktop as a string, where ``'task_count' in meta``
        throws and fails the whole resume. Rows written before the encode guard
        landed are double-encoded, so unwrap a second layer when we find one.
        """
        if raw is None:
            return None
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(meta, str):
                meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid display metadata on message row")
            return None
        if not isinstance(meta, dict):
            logger.warning("Ignoring non-object display metadata on message row")
            return None
        return meta


    @staticmethod
    def _is_duplicate_replayed_user_message(messages: List[Dict[str, Any]], msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        for prev in reversed(messages):
            if prev.get("role") == "user" and prev.get("content") == content:
                return True
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return False
        return False


    def _rows_to_conversation(
        self,
        rows,
        *,
        session_id: str,
        include_ancestors: bool,
        repair_alternation: bool,
        include_row_ids: bool = False,
    ) -> List[Dict[str, Any]]:
        """Decode fetched message rows into the OpenAI conversation format.

        Extracted from get_messages_as_conversation so get_resume_conversations
        can build the model-fed and display views from one SELECT. ``rows`` must
        already be ordered by ``id`` (insertion order) and filtered to the
        desired session set / active state by the caller.
        """
        messages = []
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = row["id"]
            # api_content is the byte-fidelity sidecar: the exact string sent
            # to the API when it differed from the clean content. Returned
            # VERBATIM — no sanitize_context, no strip — because the replay
            # path substitutes it for content to keep the provider prompt
            # cache prefix byte-stable across turns. Cleaning it here would
            # re-introduce the divergence it exists to remove.
            if row["api_content"]:
                msg["api_content"] = row["api_content"]
            if row["display_kind"]:
                msg["display_kind"] = row["display_kind"]
            if row["display_metadata"]:
                decoded = self._decode_display_metadata(row["display_metadata"])
                if decoded is not None:
                    msg["display_metadata"] = decoded
            if row["timestamp"]:
                msg["timestamp"] = row["timestamp"]
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["effect_disposition"]:
                msg["effect_disposition"] = row["effect_disposition"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []
            # Surface the platform-side message id (e.g. yuanbao msg_id,
            # telegram update_id) so platform-specific flows like recall
            # can match by external identifier instead of having to fall
            # back to content-match heuristics.  Exposed as ``message_id``
            # for backward compatibility with the JSONL transcript shape.
            if row["platform_message_id"]:
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["finish_reason"]:
                    msg["finish_reason"] = row["finish_reason"]
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize reasoning_details, falling back to None")
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_reasoning_items, falling back to None")
                        msg["codex_reasoning_items"] = None
                if row["codex_message_items"]:
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_message_items, falling back to None")
                        msg["codex_message_items"] = None
            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)
        # DEFENSE-IN-DEPTH against background-review session pollution: a forked
        # skill/memory review that (in older builds, before the _persist_disabled
        # fix) shared the parent's session_id wrote its harness turn into this
        # real session. The harness is a user/system message instructing the
        # agent to "Review the conversation above and update the skill library /
        # save to memory" under a hard tool restriction; re-loading it as live
        # history makes the agent adopt the curator role and refuse the user's
        # actual task. Strip any such harness message AND the curator-mode
        # assistant reply immediately following it, so a polluted session
        # resumes clean even if stray rows exist.
        messages = _strip_background_review_harness(messages)
        if repair_alternation and messages:
            # Lazy import: hermes_state already depends on agent.* (see
            # sanitize_context above), but keep this optional path from
            # widening the import surface at module load.
            from agent.agent_runtime_helpers import repair_message_sequence

            repaired = repair_message_sequence(None, messages)
            if repaired:
                logger.info(
                    "Repaired %d message-alternation violation(s) while "
                    "restoring session %s — durable transcript kept them, "
                    "see repair_message_sequence",
                    repaired,
                    session_id,
                )
        return messages


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
    async def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        import aiosqlite

        ref = await aiosqlite.connect(":memory:")
        try:
            await ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            cursor = await ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            for (tbl,) in await cursor.fetchall():
                cols: Dict[str, str] = {}
                column_cursor = await ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                )
                for row in await column_cursor.fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            await ref.close()

    def __init__(
        self,
        db_path: "os.PathLike[str] | str" = None,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._db_path = self.db_path
        self.read_only = bool(read_only)
        self._connection = None
        self._connect_lock = None
        self._write_lock = None
        self._schema_ready = False
        self._closed = False

    def _get_connect_lock(self) -> asyncio.Lock:
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        return self._connect_lock

    def _get_write_lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def _get_connection(self):
        if self._closed:
            raise RuntimeError("SessionDB is closed")
        if self._connection is not None:
            return self._connection

        async with self._get_connect_lock():
            if self._connection is not None:
                return self._connection
            import aiosqlite

            import aiofiles.os

            if not self.read_only and not await aiofiles.os.path.exists(self._db_path.parent):
                await aiofiles.os.makedirs(self._db_path.parent, exist_ok=True)
            database = (
                f"file:{os.path.abspath(self._db_path)}?mode=ro"
                if self.read_only
                else self._db_path
            )
            connection = await aiosqlite.connect(
                database,
                timeout=1.0,
                isolation_level=None,
                uri=self.read_only,
            )
            connection.row_factory = sqlite3.Row
            try:
                await connection.execute("PRAGMA foreign_keys=ON")
                await connection.execute("PRAGMA busy_timeout=1000")
                if not self.read_only:
                    try:
                        cursor = await connection.execute("PRAGMA journal_mode=WAL")
                        await cursor.fetchone()
                    except sqlite3.OperationalError:
                        logger.warning(
                            "WAL mode unavailable for %s; using SQLite default journal mode",
                            self._db_path,
                        )
                    await self._ensure_schema(connection)
            except BaseException:
                await connection.close()
                raise
            self._connection = connection
            return connection

    async def _ensure_schema(self, connection) -> None:
        """Create/reconcile the transcript tables without a sync DB hop.

        FTS maintenance remains owned by the optional session-search surface;
        the active turn only needs the durable session/message tables and their
        indexes.  Column reconciliation is deliberately performed with
        ``aiosqlite`` PRAGMA/ALTER statements so a first turn against a fresh
        or older database never executes synchronous SQLite I/O.
        """
        if self._schema_ready:
            return
        await connection.executescript(SCHEMA_SQL)
        expected = await self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_columns in expected.items():
            cursor = await connection.execute(
                f'PRAGMA table_info("{table_name}")'
            )
            live_columns = {row[1] for row in await cursor.fetchall()}
            for column_name, column_type in declared_columns.items():
                if column_name in live_columns:
                    continue
                safe_name = column_name.replace('"', '""')
                await connection.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {column_type}'
                )
        await self._heal_session_model_usage_pk(connection)
        await self._migrate_inline_system_prompts(connection)
        await connection.executescript(DEFERRED_INDEX_SQL)
        await connection.execute(
            "UPDATE messages SET active = 1 WHERE active IS NULL"
        )
        version_row = await (
            await connection.execute("SELECT version FROM schema_version LIMIT 1")
        ).fetchone()
        if version_row is None:
            await connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        else:
            await connection.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )
        await connection.commit()
        self._schema_ready = True

    @staticmethod
    async def _store_system_prompt(connection, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = _system_prompt_hash(system_prompt)
        await connection.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    async def _delete_unreferenced_system_prompts(connection) -> None:
        await connection.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions "
            "WHERE sessions.system_prompt_hash = system_prompts.hash)"
        )

    async def _migrate_inline_system_prompts(self, connection) -> None:
        rows = await (
            await connection.execute(
                "SELECT id, system_prompt FROM sessions "
                "WHERE system_prompt IS NOT NULL AND system_prompt_hash IS NULL"
            )
        ).fetchall()
        for row in rows:
            prompt_hash = await self._store_system_prompt(
                connection, row["system_prompt"]
            )
            await connection.execute(
                "UPDATE sessions SET system_prompt = NULL, system_prompt_hash = ? "
                "WHERE id = ?",
                (prompt_hash, row["id"]),
            )
        if rows:
            await self._delete_unreferenced_system_prompts(connection)

    @staticmethod
    def _session_row_dict(row) -> Dict[str, Any]:
        data = dict(row)
        resolved = data.pop("_system_prompt_resolved", None)
        if "system_prompt" in data:
            data["system_prompt"] = resolved
        return data

    async def _heal_session_model_usage_pk(self, connection) -> None:
        """Rebuild legacy usage tables whose composite key omits ``task``."""
        rows = await (
            await connection.execute('PRAGMA table_info("session_model_usage")')
        ).fetchall()
        if not rows:
            return
        pk_columns = {row["name"] for row in rows if row["pk"]}
        if "task" in pk_columns:
            return

        logger.info(
            "session_model_usage has legacy primary key %r; rebuilding",
            sorted(pk_columns),
        )
        await connection.execute("PRAGMA foreign_keys=OFF")
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "ALTER TABLE session_model_usage "
                "RENAME TO session_model_usage_legacy_pk"
            )
            await connection.execute(
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
            await connection.execute(
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
            await connection.execute("DROP TABLE session_model_usage_legacy_pk")
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
                "ON session_model_usage(session_id)"
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
                "ON session_model_usage(model)"
            )
            await connection.commit()
        except sqlite3.OperationalError as exc:
            await connection.rollback()
            logger.debug("session_model_usage PK heal skipped: %s", exc)
        finally:
            await connection.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _is_retryable_lock_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message

    async def _write(self, operation):
        """Run one short transaction, yielding between SQLite lock retries."""
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        compression_deadline: Optional[float] = None
        delay = self._WRITE_RETRY_MIN_S
        while True:
            try:
                async with self._get_write_lock():
                    connection = await self._get_connection()
                    await connection.execute("BEGIN IMMEDIATE")
                    try:
                        result = await operation(connection)
                        await connection.commit()
                        return result
                    except BaseException:
                        await connection.rollback()
                        raise
            except asyncio.CancelledError:
                raise
            except SessionCompressionInProgressError:
                if compression_deadline is None:
                    compression_deadline = min(
                        time.monotonic() + self._COMPRESSION_BUSY_WAIT_S,
                        deadline,
                    )
                remaining = compression_deadline - time.monotonic()
                if remaining <= 0:
                    raise
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, self._WRITE_RETRY_MAX_S)
            except Exception as exc:
                no_more_rows = (
                    isinstance(exc, sqlite3.Error)
                    and "no more rows available" in str(exc).lower()
                )
                if not no_more_rows and not self._is_retryable_lock_error(exc):
                    raise
                if time.monotonic() >= deadline:
                    if no_more_rows:
                        raise
                    raise sqlite3.OperationalError(
                        "database is locked by another Hermes process; "
                        "the database appears healthy but remained busy past "
                        f"the {self._WRITE_PATIENCE_S:g}s write deadline"
                    ) from exc
                await asyncio.sleep(random.uniform(delay / 2, delay))
                delay = min(delay * 2, self._WRITE_RETRY_MAX_S)

    async def _check_transcript_write_guards(
        self,
        connection,
        session_id: str,
        compression_lock_holder: Optional[str],
    ) -> None:
        """Apply transcript admission guards inside the active transaction."""
        active_lock = await (
            await connection.execute(
                "SELECT holder FROM compression_locks "
                "WHERE session_id = ? AND expires_at > ?",
                (session_id, time.time()),
            )
        ).fetchone()
        if (
            active_lock is not None
            and active_lock["holder"] != compression_lock_holder
        ):
            raise SessionCompressionInProgressError(
                f"Session {session_id!r} is being compressed by another writer"
            )
        session = await (
            await connection.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        if (
            session is not None
            and session["ended_at"] is not None
            and session["end_reason"] == "compression"
        ):
            raise CompressionSessionClosedError(session_id)

    async def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create or enrich the turn's session row without blocking the loop."""
        model_config = kwargs.get("model_config")
        parent_session_id = kwargs.get("parent_session_id")

        async def _create(connection):
            system_prompt_hash = await self._store_system_prompt(
                connection, kwargs.get("system_prompt")
            )
            await connection.execute(
                """INSERT INTO sessions (
                   id, source, user_id, session_key, chat_id, chat_type, thread_id,
                   model, model_config, system_prompt, system_prompt_hash,
                   parent_session_id, cwd,
                   profile_name, git_repo_root, started_at
                )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = COALESCE(sessions.model_config, excluded.model_config),
                       system_prompt_hash = COALESCE(
                           sessions.system_prompt_hash,
                           excluded.system_prompt_hash
                       ),
                       system_prompt = CASE
                           WHEN sessions.system_prompt_hash IS NULL
                                AND excluded.system_prompt_hash IS NOT NULL
                           THEN NULL
                           ELSE sessions.system_prompt
                       END,
                       session_key = COALESCE(sessions.session_key, excluded.session_key),
                       chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                       chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                       thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                       parent_session_id = COALESCE(sessions.parent_session_id, excluded.parent_session_id),
                       cwd = COALESCE(sessions.cwd, excluded.cwd),
                       profile_name = COALESCE(sessions.profile_name, excluded.profile_name),
                       git_repo_root = COALESCE(sessions.git_repo_root, excluded.git_repo_root)""",
                (
                    session_id,
                    source,
                    kwargs.get("user_id"),
                    kwargs.get("session_key"),
                    kwargs.get("chat_id"),
                    kwargs.get("chat_type"),
                    kwargs.get("thread_id"),
                    kwargs.get("model"),
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    kwargs.get("cwd"),
                    kwargs.get("profile_name"),
                    kwargs.get("git_repo_root"),
                    time.time(),
                ),
            )
            if system_prompt_hash is not None:
                await self._delete_unreferenced_system_prompts(connection)
            if parent_session_id:
                await connection.execute(
                    """UPDATE sessions
                       SET cwd = COALESCE(sessions.cwd,
                                 (SELECT p.cwd FROM sessions p
                                   WHERE p.id = sessions.parent_session_id)),
                           git_repo_root = COALESCE(sessions.git_repo_root,
                                           (SELECT p.git_repo_root FROM sessions p
                                             WHERE p.id = sessions.parent_session_id)),
                           git_branch = COALESCE(sessions.git_branch,
                                        (SELECT p.git_branch FROM sessions p
                                          WHERE p.id = sessions.parent_session_id)),
                           profile_name = COALESCE(sessions.profile_name,
                                          (SELECT p.profile_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL""",
                    (session_id,),
                )
                await connection.execute(
                    """UPDATE sessions
                       SET user_id = COALESCE(sessions.user_id,
                                     (SELECT p.user_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           session_key = COALESCE(sessions.session_key,
                                         (SELECT p.session_key FROM sessions p
                                           WHERE p.id = sessions.parent_session_id)),
                           chat_id = COALESCE(sessions.chat_id,
                                     (SELECT p.chat_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           chat_type = COALESCE(sessions.chat_type,
                                       (SELECT p.chat_type FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           thread_id = COALESCE(sessions.thread_id,
                                       (SELECT p.thread_id FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           display_name = COALESCE(sessions.display_name,
                                          (SELECT p.display_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id)),
                           origin_json = COALESCE(sessions.origin_json,
                                         (SELECT p.origin_json FROM sessions p
                                           WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL
                       AND EXISTS (
                           SELECT 1 FROM sessions p
                           WHERE p.id = sessions.parent_session_id
                             AND p.end_reason = 'compression'
                       )""",
                    (session_id,),
                )

        await self._write(_create)
        return session_id

    async def update_session_cwd(
        self,
        session_id: str,
        cwd: str,
        git_branch: str = None,
        git_repo_root: str = None,
    ) -> None:
        """Persist workspace identity without blocking the event loop."""
        if not session_id or not cwd:
            return

        assignments = ["cwd = ?"]
        parameters: list[Any] = [cwd]
        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()
        if branch:
            assignments.append("git_branch = ?")
            parameters.append(branch)
        if repo_root:
            assignments.append("git_repo_root = ?")
            parameters.append(repo_root)
        parameters.append(session_id)

        async def _update(connection):
            await connection.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )

        await self._write(_update)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
        effect_disposition: Optional[str] = None,
        timestamp: Any = None,
        api_content: Optional[str] = None,
        display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None,
        compression_lock_holder: Optional[str] = None,
    ) -> int:
        """Append one transcript row using the same wire serialization as SessionDB."""
        display_metadata_json = self._encode_display_metadata(display_metadata)
        reasoning_details_json = json.dumps(reasoning_details) if reasoning_details else None
        codex_items_json = json.dumps(codex_reasoning_items) if codex_reasoning_items else None
        codex_message_items_json = (
            json.dumps(codex_message_items) if codex_message_items else None
        )
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                tool_calls = []
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        stored_content = self._encode_content(content)
        message_timestamp = time.time()
        if timestamp is not None:
            try:
                message_timestamp = float(
                    timestamp.timestamp() if hasattr(timestamp, "timestamp") else timestamp
                )
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid explicit message timestamp: %r", timestamp)
        num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else int(tool_calls is not None)

        async def _append(connection):
            await self._check_transcript_write_guards(
                connection, session_id, compression_lock_holder
            )
            cursor = await connection.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, role, stored_content, tool_call_id, tool_calls_json,
                    _scrub_surrogates(tool_name), effect_disposition, message_timestamp,
                    token_count, finish_reason, _scrub_surrogates(reasoning),
                    _scrub_surrogates(reasoning_content), reasoning_details_json,
                    codex_items_json, codex_message_items_json, platform_message_id,
                    1 if observed else 0, 1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(display_kind) if isinstance(display_kind, str) else None,
                    display_metadata_json,
                ),
            )
            if num_tool_calls:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + 1, "
                    "tool_call_count = tool_call_count + ? WHERE id = ?",
                    (num_tool_calls, session_id),
                )
            else:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return cursor.lastrowid

        return await self._write(_append)

    async def _insert_message_rows(
        self,
        connection,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, int]:
        """Insert transcript rows in the caller's active transaction."""
        now_ts = time.time()
        inserted = 0
        tool_calls_total = 0
        for message in messages:
            role = message.get("role", "unknown")
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []
            timestamp = message.get("timestamp", now_ts)
            try:
                timestamp = float(
                    timestamp.timestamp()
                    if hasattr(timestamp, "timestamp")
                    else timestamp
                )
            except (TypeError, ValueError):
                logger.debug(
                    "Ignoring invalid explicit message timestamp: %r",
                    message.get("timestamp"),
                )
                timestamp = now_ts
            reasoning_details = (
                message.get("reasoning_details") if role == "assistant" else None
            )
            codex_reasoning_items = (
                message.get("codex_reasoning_items") if role == "assistant" else None
            )
            codex_message_items = (
                message.get("codex_message_items") if role == "assistant" else None
            )
            api_content = message.get("api_content")
            display_kind = message.get("display_kind")
            await connection.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    self._encode_content(message.get("content")),
                    message.get("tool_call_id"),
                    json.dumps(tool_calls) if tool_calls else None,
                    _scrub_surrogates(message.get("tool_name")),
                    message.get("effect_disposition"),
                    timestamp,
                    message.get("token_count"),
                    message.get("finish_reason"),
                    _scrub_surrogates(message.get("reasoning"))
                    if role == "assistant"
                    else None,
                    _scrub_surrogates(message.get("reasoning_content"))
                    if role == "assistant"
                    else None,
                    json.dumps(reasoning_details) if reasoning_details else None,
                    json.dumps(codex_reasoning_items) if codex_reasoning_items else None,
                    json.dumps(codex_message_items) if codex_message_items else None,
                    message.get("platform_message_id") or message.get("message_id"),
                    1 if message.get("observed") else 0,
                    1,
                    _scrub_surrogates(api_content)
                    if isinstance(api_content, str)
                    else None,
                    _scrub_surrogates(display_kind)
                    if isinstance(display_kind, str)
                    else None,
                    self._encode_display_metadata(message.get("display_metadata")),
                ),
            )
            inserted += 1
            if tool_calls is not None:
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else 1
                )
            now_ts = max(now_ts + 1e-6, timestamp + 1e-6)
        return inserted, tool_calls_total

    async def append_messages_batch(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        compression_lock_holder: Optional[str] = None,
        chunk_rows: Optional[int] = None,
    ) -> int:
        """Append one turn atomically, optionally chunking large copy jobs."""
        if not messages:
            return 0
        if chunk_rows is not None:
            if chunk_rows <= 0:
                raise ValueError("chunk_rows must be positive")
            if len(messages) > chunk_rows:
                inserted_total = 0
                for start in range(0, len(messages), chunk_rows):
                    inserted_total += await self.append_messages_batch(
                        session_id,
                        messages[start : start + chunk_rows],
                        compression_lock_holder=compression_lock_holder,
                    )
                return inserted_total

        async def _append_batch(connection):
            await self._check_transcript_write_guards(
                connection, session_id, compression_lock_holder
            )
            inserted, tool_calls_total = await self._insert_message_rows(
                connection, session_id, messages
            )
            if tool_calls_total:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + ?, "
                    "tool_call_count = tool_call_count + ? WHERE id = ?",
                    (inserted, tool_calls_total, session_id),
                )
            else:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + ? WHERE id = ?",
                    (inserted, session_id),
                )
            return inserted

        return await self._write(_append_batch)

    async def end_session(self, session_id: str, end_reason: str) -> None:
        async def _end(connection):
            await connection.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )

        await self._write(_end)

    async def try_acquire_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Atomically acquire an expiring compression lease."""
        if not session_id or not holder:
            return False
        now = time.time()

        async def _acquire(connection):
            row = await (
                await connection.execute(
                    "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                    (session_id,),
                )
            ).fetchone()
            if row is not None and (
                float(row["expires_at"]) < now
                or _compression_lock_holder_process_is_dead(row["holder"])
            ):
                await connection.execute(
                    "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?",
                    (session_id, row["holder"]),
                )
            await connection.execute(
                "INSERT OR IGNORE INTO compression_locks "
                "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                (session_id, holder, now, now + ttl_seconds),
            )
            owner = await (
                await connection.execute(
                    "SELECT holder FROM compression_locks WHERE session_id = ?",
                    (session_id,),
                )
            ).fetchone()
            return owner is not None and owner["holder"] == holder

        try:
            return bool(await self._write(_acquire))
        except sqlite3.Error as exc:
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False

    async def refresh_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Extend a lease only while its owner still matches."""
        if not session_id or not holder:
            return False

        async def _refresh(connection):
            cursor = await connection.execute(
                "UPDATE compression_locks SET expires_at = ? "
                "WHERE session_id = ? AND holder = ?",
                (time.time() + ttl_seconds, session_id, holder),
            )
            return cursor.rowcount > 0

        try:
            return bool(await self._write(_refresh))
        except sqlite3.Error as exc:
            logger.warning("refresh_compression_lock(%s) failed: %s", session_id, exc)
            return False

    async def release_compression_lock(self, session_id: str, holder: str) -> None:
        """Release a compression lease iff it is still owned by *holder*."""
        if not session_id or not holder:
            return

        async def _release(connection):
            await connection.execute(
                "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?",
                (session_id, holder),
            )

        try:
            await self._write(_release)
        except sqlite3.Error as exc:
            logger.warning("release_compression_lock(%s) failed: %s", session_id, exc)

    async def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Return the live compression-lease owner, if any."""
        if not session_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT holder FROM compression_locks "
                "WHERE session_id = ? AND expires_at >= ?",
                (session_id, time.time()),
            )
        ).fetchone()
        return row["holder"] if row is not None else None

    async def archive_and_compact(
        self,
        session_id: str,
        compacted_messages: List[Dict[str, Any]],
    ) -> int:
        """Atomically archive live rows and insert the compacted transcript."""
        async def _archive(connection):
            await connection.execute(
                "UPDATE messages SET active = 0, compacted = 1 "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            )
            now_ts = time.time()
            tool_calls_total = 0
            for message in compacted_messages:
                role = message.get("role", "unknown")
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, str):
                    try:
                        tool_calls = json.loads(tool_calls)
                    except (json.JSONDecodeError, TypeError):
                        tool_calls = []
                timestamp = message.get("timestamp", now_ts)
                try:
                    timestamp = float(
                        timestamp.timestamp()
                        if hasattr(timestamp, "timestamp")
                        else timestamp
                    )
                except (TypeError, ValueError):
                    timestamp = now_ts
                reasoning_details = (
                    message.get("reasoning_details") if role == "assistant" else None
                )
                codex_reasoning_items = (
                    message.get("codex_reasoning_items") if role == "assistant" else None
                )
                codex_message_items = (
                    message.get("codex_message_items") if role == "assistant" else None
                )
                await connection.execute(
                    """INSERT INTO messages (session_id, role, content, tool_call_id,
                       tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        role,
                        self._encode_content(message.get("content")),
                        message.get("tool_call_id"),
                        json.dumps(tool_calls) if tool_calls else None,
                        _scrub_surrogates(message.get("tool_name")),
                        message.get("effect_disposition"),
                        timestamp,
                        message.get("token_count"),
                        message.get("finish_reason"),
                        _scrub_surrogates(message.get("reasoning"))
                        if role == "assistant"
                        else None,
                        _scrub_surrogates(message.get("reasoning_content"))
                        if role == "assistant"
                        else None,
                        json.dumps(reasoning_details) if reasoning_details else None,
                        json.dumps(codex_reasoning_items) if codex_reasoning_items else None,
                        json.dumps(codex_message_items) if codex_message_items else None,
                        message.get("platform_message_id") or message.get("message_id"),
                        1 if message.get("observed") else 0,
                        1,
                        _scrub_surrogates(message.get("api_content"))
                        if isinstance(message.get("api_content"), str)
                        else None,
                        _scrub_surrogates(message.get("display_kind"))
                        if isinstance(message.get("display_kind"), str)
                        else None,
                        self._encode_display_metadata(message.get("display_metadata")),
                    ),
                )
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else int(tool_calls is not None)
                )
                now_ts = max(now_ts + 1e-6, timestamp + 1e-6)
            await connection.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (len(compacted_messages), tool_calls_total, session_id),
            )
            return len(compacted_messages)

        return await self._write(_archive)

    async def update_system_prompt(
        self, session_id: str, system_prompt: Optional[str]
    ) -> None:
        """Persist the assembled prompt without a synchronous SQLite call."""
        async def _update(connection):
            system_prompt_hash = await self._store_system_prompt(
                connection, system_prompt
            )
            await connection.execute(
                "UPDATE sessions SET system_prompt = NULL, "
                "system_prompt_hash = ? WHERE id = ?",
                (system_prompt_hash, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update)

    async def update_session_runtime_lock(
        self,
        session_id: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        route_source: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """Merge a runtime lock while preserving unrelated model metadata."""
        lock = {
            "provider": provider or "",
            "model": model or "",
            "model_options": model_options or {},
            "route_source": route_source or "",
            "confirmed": bool(confirmed),
            "updated_at": time.time(),
        }

        async def _update(connection):
            row = await (
                await connection.execute(
                    "SELECT model_config FROM sessions WHERE id = ?", (session_id,)
                )
            ).fetchone()
            if row is None:
                return
            config: Dict[str, Any] = {}
            raw = row["model_config"]
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        config = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            config["browser_model_lock"] = lock
            await connection.execute(
                "UPDATE sessions SET model_config = ?, model = COALESCE(?, model), "
                "system_prompt = NULL, system_prompt_hash = NULL WHERE id = ?",
                (json.dumps(config), model, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update)

    async def set_latest_user_api_content(
        self, session_id: str, content: Any, api_content: str
    ) -> int:
        """Backfill the API sidecar on the newest matching active user row."""
        encoded = self._encode_content(content)

        async def _update(connection):
            cursor = await connection.execute(
                "UPDATE messages SET api_content = ? WHERE id = ("
                "SELECT id FROM messages "
                "WHERE session_id = ? AND role = 'user' AND active = 1 "
                "ORDER BY id DESC LIMIT 1"
                ") AND content IS ?",
                (_scrub_surrogates(api_content), session_id, encoded),
            )
            return cursor.rowcount

        return await self._write(_update)

    async def record_auxiliary_usage(
        self,
        session_id: str,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        """Persist one auxiliary-model usage delta without sync SQLite I/O."""
        if not session_id or not task:
            return
        await self.create_session(session_id, "unknown")
        now = time.time()

        async def _record(connection):
            await connection.execute(
                """INSERT INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   ) VALUES (?, ?, ?, ?, '', ?, 1, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)
                   ON CONFLICT(session_id, model, billing_provider, billing_base_url,
                               billing_mode, task)
                   DO UPDATE SET
                       api_call_count = api_call_count + 1,
                       input_tokens = input_tokens + excluded.input_tokens,
                       output_tokens = output_tokens + excluded.output_tokens,
                       cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                       cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                       reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                       estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                       last_seen = excluded.last_seen""",
                (
                    session_id,
                    model or "unknown",
                    billing_provider or "",
                    billing_base_url or "",
                    task,
                    input_tokens or 0,
                    output_tokens or 0,
                    cache_read_tokens or 0,
                    cache_write_tokens or 0,
                    reasoning_tokens or 0,
                    float(estimated_cost_usd or 0.0),
                    now,
                    now,
                ),
            )

        await self._write(_record)

    async def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Persist one token-accounting update through the async connection."""
        await self.create_session(session_id, "unknown", model=model)
        has_accounted_usage = bool(
            input_tokens or output_tokens or cache_read_tokens
            or cache_write_tokens or reasoning_tokens or api_call_count
            or estimated_cost_usd or actual_cost_usd
        )
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?, output_tokens = ?,
                   cache_read_tokens = ?, cache_write_tokens = ?,
                   reasoning_tokens = ?, estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE WHEN ? IS NULL THEN actual_cost_usd ELSE ? END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?), api_call_count = ?
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?, output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = COALESCE(api_call_count, 0) + ?
                   WHERE id = ?"""
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None,
            model if has_accounted_usage else None,
            api_call_count,
            session_id,
        )

        async def _update(connection):
            await connection.execute(sql, params)
            if absolute or not has_accounted_usage:
                return
            await connection.execute(
                """INSERT INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, model, billing_provider, billing_base_url,
                               billing_mode, task)
                   DO UPDATE SET
                       api_call_count = api_call_count + excluded.api_call_count,
                       input_tokens = input_tokens + excluded.input_tokens,
                       output_tokens = output_tokens + excluded.output_tokens,
                       cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                       cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                       reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                       estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                       actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                       cost_status = COALESCE(excluded.cost_status, cost_status),
                       cost_source = COALESCE(excluded.cost_source, cost_source),
                       last_seen = excluded.last_seen""",
                (
                    session_id,
                    model or "unknown",
                    billing_provider or "",
                    billing_base_url or "",
                    billing_mode or "",
                    api_call_count or 0,
                    input_tokens or 0,
                    output_tokens or 0,
                    cache_read_tokens or 0,
                    cache_write_tokens or 0,
                    reasoning_tokens or 0,
                    float(estimated_cost_usd or 0.0),
                    float(actual_cost_usd or 0.0),
                    cost_status,
                    cost_source,
                    time.time(),
                    time.time(),
                ),
            )

        await self._write(_update)

    async def update_session_billing_route(
        self,
        session_id: str,
        *,
        provider: str,
        base_url: str,
        billing_mode: Optional[str] = None,
    ) -> None:
        """Persist a model-route change through the native async connection."""
        async def _update_route(connection):
            await connection.execute(
                """UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (provider, base_url, billing_mode, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update_route)

    async def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the durable ``state_meta`` store."""
        connection = await self._get_connection()
        cursor = await connection.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row is not None else None

    async def set_meta(self, key: str, value: str, *, cursor: Any = None) -> None:
        """Atomically upsert a value in the durable ``state_meta`` store."""

        if cursor is not None:
            await cursor.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            return

        async def _set(connection):
            await connection.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

        await self._write(_set)

    async def delete_meta(self, key: str) -> bool:
        """Delete one metadata key and report whether a row was removed."""

        async def _delete(connection):
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key = ?", (key,)
            )
            return cursor.rowcount > 0

        return await self._write(_delete)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return one session row through the adapter's async connection."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) "
                "AS _system_prompt_resolved "
                "FROM sessions s LEFT JOIN system_prompts sp "
                "ON sp.hash = s.system_prompt_hash WHERE s.id = ?",
                (session_id,),
            )
        ).fetchone()
        return self._session_row_dict(row) if row is not None else None

    @staticmethod
    async def _remove_session_files(
        sessions_dir: Optional[Path], session_id: str
    ) -> None:
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            try:
                await aiofiles.os.remove(sessions_dir / f"{session_id}{suffix}")
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            names = await aiofiles.os.listdir(sessions_dir)
        except OSError:
            return
        prefix = f"request_dump_{session_id}_"
        for name in names:
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            try:
                await aiofiles.os.remove(sessions_dir / name)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    async def get_session_delete_targets(self, session_id: str) -> List[str]:
        connection = await self._get_connection()
        exists = await (
            await connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            )
        ).fetchone()
        if exists is None:
            return []
        delegate_ids = await _collect_delegate_child_ids(connection, [session_id])
        return [session_id, *sorted(delegate_ids)]

    async def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        removed_delegate_ids: List[str] = []
        expected_ids = (
            set(expected_delete_ids) if expected_delete_ids is not None else None
        )

        async def _delete(connection):
            exists = await (
                await connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
                )
            ).fetchone()
            if exists is None:
                return False
            if expected_ids is not None:
                actual_ids = {
                    session_id,
                    *(await _collect_delegate_child_ids(connection, [session_id])),
                }
                if actual_ids != expected_ids:
                    return False
            removed_delegate_ids.extend(
                await _delete_delegate_children(connection, [session_id])
            )
            await connection.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            await connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            await connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await self._delete_unreferenced_system_prompts(connection)
            return True

        deleted = bool(await self._write(_delete))
        if deleted:
            for delegate_id in removed_delegate_ids:
                await self._remove_session_files(sessions_dir, delegate_id)
            await self._remove_session_files(sessions_dir, session_id)
        return deleted

    async def delete_session_if_empty(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        async def _delete(connection):
            cursor = await connection.execute(
                """DELETE FROM sessions
                   WHERE id = ?
                     AND title IS NULL
                     AND NOT EXISTS (
                         SELECT 1 FROM messages
                         WHERE messages.session_id = sessions.id
                     )
                     AND NOT EXISTS (
                         SELECT 1 FROM sessions child
                         WHERE child.parent_session_id = sessions.id
                     )""",
                (session_id,),
            )
            if cursor.rowcount > 0:
                await self._delete_unreferenced_system_prompts(connection)
            return cursor.rowcount > 0

        deleted = bool(await self._write(_delete))
        if deleted:
            await self._remove_session_files(sessions_dir, session_id)
        return deleted

    async def delete_sessions(
        self,
        session_ids: List[str],
        sessions_dir: Optional[Path] = None,
    ) -> int:
        ids = list(
            {
                session_id
                for session_id in session_ids
                if isinstance(session_id, str) and session_id
            }
        )
        if not ids:
            return 0
        removed_ids: List[str] = []
        removed_delegate_ids: List[str] = []

        async def _delete(connection):
            placeholders = ",".join("?" for _ in ids)
            rows = await (
                await connection.execute(
                    f"SELECT id FROM sessions WHERE id IN ({placeholders})", ids
                )
            ).fetchall()
            existing = [row["id"] for row in rows]
            if not existing:
                return 0
            existing_placeholders = ",".join("?" for _ in existing)
            removed_delegate_ids.extend(
                await _delete_delegate_children(connection, existing)
            )
            await connection.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({existing_placeholders})",
                existing,
            )
            await connection.execute(
                f"DELETE FROM messages "
                f"WHERE session_id IN ({existing_placeholders})",
                existing,
            )
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({existing_placeholders})",
                existing,
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(existing)
            return len(existing)

        count = int(await self._write(_delete))
        for delegate_id in removed_delegate_ids:
            await self._remove_session_files(sessions_dir, delegate_id)
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    async def delete_empty_sessions(
        self, sessions_dir: Optional[Path] = None
    ) -> int:
        removed_ids: List[str] = []

        async def _delete(connection):
            rows = await (
                await connection.execute(
                    "SELECT id FROM sessions WHERE message_count = 0 "
                    "AND ended_at IS NOT NULL AND archived = 0"
                )
            ).fetchall()
            session_ids = [row["id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            await connection.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})",
                session_ids,
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(session_ids)
            return len(session_ids)

        count = int(await self._write(_delete))
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    @staticmethod
    def _prune_filter_where(
        *,
        last_active_before: Optional[float] = None,
        last_active_after: Optional[float] = None,
        started_before: Optional[float] = None,
        started_after: Optional[float] = None,
        source: Optional[str] = None,
        title_like: Optional[str] = None,
        end_reason: Optional[str] = None,
        cwd_prefix: Optional[str] = None,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None,
        archived: Optional[bool] = None,
        model_like: Optional[str] = None,
        provider: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        branch_like: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_tool_calls: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
    ) -> Tuple[str, list]:
        clauses = ["s.ended_at IS NOT NULL"]
        params: list[Any] = []
        if last_active_before is not None:
            clauses.append(
                "COALESCE((SELECT MAX(m.timestamp) FROM messages m "
                "WHERE m.session_id = s.id), s.started_at) < ?"
            )
            params.append(last_active_before)
        if last_active_after is not None:
            clauses.append(
                "COALESCE((SELECT MAX(m.timestamp) FROM messages m "
                "WHERE m.session_id = s.id), s.started_at) >= ?"
            )
            params.append(last_active_after)
        if started_before is not None:
            clauses.append("s.started_at < ?")
            params.append(started_before)
        if started_after is not None:
            clauses.append("s.started_at >= ?")
            params.append(started_after)
        if source:
            clauses.append("s.source = ?")
            params.append(source)
        if title_like:
            clauses.append("LOWER(COALESCE(s.title, '')) LIKE ?")
            params.append(f"%{title_like.lower()}%")
        if end_reason:
            clauses.append("s.end_reason = ?")
            params.append(end_reason)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            clauses.append(clause)
            params.extend(clause_params)
        if min_messages is not None:
            clauses.append("s.message_count >= ?")
            params.append(min_messages)
        if max_messages is not None:
            clauses.append("s.message_count <= ?")
            params.append(max_messages)
        if model_like:
            clauses.append("LOWER(COALESCE(s.model, '')) LIKE ?")
            params.append(f"%{model_like.lower()}%")
        if provider:
            clauses.append("LOWER(COALESCE(s.billing_provider, '')) = ?")
            params.append(provider.lower())
        if user_id:
            clauses.append("s.user_id = ?")
            params.append(user_id)
        if chat_id:
            clauses.append("s.chat_id = ?")
            params.append(chat_id)
        if chat_type:
            clauses.append("s.chat_type = ?")
            params.append(chat_type)
        if branch_like:
            clauses.append("LOWER(COALESCE(s.git_branch, '')) LIKE ?")
            params.append(f"%{branch_like.lower()}%")
        if min_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + "
                "COALESCE(s.output_tokens, 0)) >= ?"
            )
            params.append(min_tokens)
        if max_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + "
                "COALESCE(s.output_tokens, 0)) <= ?"
            )
            params.append(max_tokens)
        if min_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) >= ?"
            )
            params.append(min_cost)
        if max_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) <= ?"
            )
            params.append(max_cost)
        if min_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) >= ?")
            params.append(min_tool_calls)
        if max_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) <= ?")
            params.append(max_tool_calls)
        if archived is True:
            clauses.append("s.archived = 1")
        elif archived is False:
            clauses.append("s.archived = 0")
        return " AND ".join(clauses), params

    async def prune_sessions(
        self,
        older_than_days: Optional[float] = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
        **filters,
    ) -> int:
        if (
            filters.get("last_active_before") is None
            and filters.get("started_before") is None
            and older_than_days is not None
        ):
            filters["last_active_before"] = time.time() - (
                float(older_than_days) * 86_400
            )
        where, params = self._prune_filter_where(source=source, **filters)
        removed_ids: List[str] = []

        async def _delete(connection):
            rows = await (
                await connection.execute(
                    f"SELECT s.id FROM sessions s WHERE {where}", params
                )
            ).fetchall()
            session_ids = [row["id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            await connection.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})",
                session_ids,
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(session_ids)
            return len(session_ids)

        count = int(await self._write(_delete))
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    async def prune_empty_ghost_sessions(
        self, sessions_dir: Optional[Path] = None
    ) -> int:
        cutoff = time.time() - 86_400
        removed_ids: List[str] = []

        async def _delete(connection):
            rows = await (
                await connection.execute(
                    "SELECT id FROM sessions WHERE source = 'tui' "
                    "AND title IS NULL AND ended_at IS NOT NULL "
                    "AND started_at < ? AND NOT EXISTS ("
                    "SELECT 1 FROM messages "
                    "WHERE messages.session_id = sessions.id)",
                    (cutoff,),
                )
            ).fetchall()
            session_ids = [row["id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(session_ids)
            return len(session_ids)

        count = int(await self._write(_delete))
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    @staticmethod
    def _decode_message_row(row) -> Dict[str, Any]:
        """Decode one SQLite message row without touching ``SessionDB``."""
        message = dict(row)
        if "content" in message:
            message["content"] = SessionDB._decode_content(message["content"])
        if message.get("tool_calls"):
            try:
                message["tool_calls"] = json.loads(message["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to deserialize tool_calls in async session read; "
                    "falling back to []"
                )
                message["tool_calls"] = []
        if message.get("display_metadata") is not None:
            message["display_metadata"] = SessionDB._decode_display_metadata(
                message["display_metadata"]
            )
        return message

    async def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Load a session's messages through the native async connection."""
        connection = await self._get_connection()
        active_clause = "" if include_inactive else " AND active = 1"
        query = (
            "SELECT * FROM messages WHERE session_id = ?"
            f"{active_clause} ORDER BY id"
        )
        params: list[Any] = [session_id]
        if limit is not None or offset:
            query += " LIMIT ? OFFSET ?"
            params.extend([-1 if limit is None else limit, offset])
        cursor = await connection.execute(query, params)
        return [self._decode_message_row(row) for row in await cursor.fetchall()]

    async def rewind_to_message(
        self, session_id: str, target_message_id: int
    ) -> Dict[str, Any]:
        """Soft-delete the target user message and every later active row."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (target_message_id, session_id),
            )
        ).fetchone()
        if row is None:
            raise ValueError(
                f"message {target_message_id} not found in session {session_id}"
            )
        target = self._decode_message_row(row)
        if target.get("role") != "user":
            raise ValueError(
                "rewind target must be a 'user' message "
                f"(got role={target.get('role')!r}, id={target_message_id})"
            )

        async def _rewind(conn):
            cursor = await conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 1",
                (session_id, target_message_id),
            )
            ids = [item[0] for item in await cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await conn.execute(
                    f"UPDATE messages SET active = 0 WHERE id IN ({placeholders})",
                    ids,
                )
            await conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            head = await (
                await conn.execute(
                    "SELECT MAX(id) FROM messages "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )
            ).fetchone()
            return ids, head[0] if head and head[0] is not None else None

        rewound, new_head_id = await self._write(_rewind)
        return {
            "rewound_count": len(rewound),
            "target_message": target,
            "new_head_id": new_head_id,
        }

    async def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Return an async anchored message window with boundary counts."""
        connection = await self._get_connection()
        window = max(0, int(window))
        anchor = await (
            await connection.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            )
        ).fetchone()
        if anchor is None:
            return {"window": [], "messages_before": 0, "messages_after": 0}
        before = await (
            await connection.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            )
        ).fetchall()
        after = await (
            await connection.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            )
        ).fetchall()
        rows = list(reversed(before)) + list(after)
        return {
            "window": [self._decode_message_row(row) for row in rows],
            "messages_before": max(0, len(before) - 1),
            "messages_after": len(after),
        }

    async def get_message_storage_state(
        self, message_id: int
    ) -> Optional[Dict[str, Any]]:
        """Return storage flags for one message id without a raw connection."""
        if not message_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT session_id, active, compacted FROM messages WHERE id = ?",
                (message_id,),
            )
        ).fetchone()
        return dict(row) if row is not None else None

    async def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant"),
    ) -> Dict[str, Any]:
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
        starts: list[Dict[str, Any]] = []
        ends: list[Dict[str, Any]] = []
        if bookend:
            connection = await self._get_connection()
            role_sql = ""
            role_params: list[Any] = []
            if keep_roles is not None:
                role_sql = " AND role IN (" + ",".join("?" for _ in keep_roles) + ")"
                role_params = list(keep_roles)
            start_rows = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND id < ?"
                    f"{role_sql} AND length(content) > 0 ORDER BY id ASC LIMIT ?",
                    (session_id, rows[0]["id"], *role_params, bookend),
                )
            ).fetchall()
            end_rows = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND id > ?"
                    f"{role_sql} AND length(content) > 0 ORDER BY id DESC LIMIT ?",
                    (session_id, rows[-1]["id"], *role_params, bookend),
                )
            ).fetchall()
            starts = [self._decode_message_row(row) for row in start_rows]
            ends = [self._decode_message_row(row) for row in reversed(end_rows)]
        return {
            "window": filtered,
            "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": starts,
            "bookend_end": ends,
        }

    async def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session title through the async connection."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) "
                "AS _system_prompt_resolved "
                "FROM sessions s LEFT JOIN system_prompts sp "
                "ON sp.hash = s.system_prompt_hash WHERE s.title = ?",
                (title,),
            )
        ).fetchone()
        return self._session_row_dict(row) if row is not None else None

    async def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve an exact or numbered continuation title asynchronously."""
        exact = await self.get_session_by_title(title)
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\' "
                "ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
        ).fetchall()
        if rows:
            return rows[0]["id"]
        return exact["id"] if exact else None

    async def fts_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """Return deferred FTS rebuild progress without sync metadata access."""
        high_water = await self.get_meta("fts_rebuild_high_water")
        if high_water is None:
            return None
        try:
            total = int(high_water)
            indexed = int(await self.get_meta("fts_rebuild_progress") or 0)
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

    async def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
        fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search message text natively, preferring FTS5 with LIKE fallback."""
        result_fields = self._search_message_fields(fields)
        if not query or not str(query).strip():
            return []
        connection = await self._get_connection()
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        sanitized = self._sanitize_fts5_query(str(query))
        if not sanitized:
            return []
        where: list[str] = []
        params: list[Any] = [sanitized]
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter:
            where.append("s.source IN (" + ",".join("?" for _ in source_filter) + ")")
            params.extend(source_filter)
        if exclude_sources:
            where.append("s.source NOT IN (" + ",".join("?" for _ in exclude_sources) + ")")
            params.extend(exclude_sources)
        if role_filter:
            where.append("m.role IN (" + ",".join("?" for _ in role_filter) + ")")
            params.extend(role_filter)
        where_sql = " AND ".join(where)
        order_sql = "ORDER BY rank"
        if sort == "newest":
            order_sql = "ORDER BY m.timestamp DESC, rank"
        elif sort == "oldest":
            order_sql = "ORDER BY m.timestamp ASC, rank"
        try:
            query_sql = f"""
                SELECT m.id, m.session_id, m.role,
                       snippet(messages_fts, -1, '>>>', '<<<', '...', 40) AS snippet,
                       m.content, m.timestamp, m.tool_name,
                       s.source, s.model, s.started_at AS session_started
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                JOIN sessions s ON s.id = m.session_id
                WHERE messages_fts MATCH ?
                {(' AND ' + where_sql) if where_sql else ''}
                {order_sql} LIMIT ? OFFSET ?
            """
            cursor = await connection.execute(query_sql, [*params, limit, offset])
            rows = await cursor.fetchall()
        except sqlite3.OperationalError:
            # Older or partially migrated databases may not have the FTS table.
            # Keep recall useful with a parameterized text search; no sync
            # fallback is involved.
            like = f"%{str(query).strip()}%"
            like_where = ["(m.content LIKE ? OR m.tool_name LIKE ? OR m.tool_calls LIKE ?)"]
            like_params: list[Any] = [like, like, like]
            if not include_inactive:
                like_where.append("(m.active = 1 OR m.compacted = 1)")
            if source_filter:
                like_where.append("s.source IN (" + ",".join("?" for _ in source_filter) + ")")
                like_params.extend(source_filter)
            if exclude_sources:
                like_where.append("s.source NOT IN (" + ",".join("?" for _ in exclude_sources) + ")")
                like_params.extend(exclude_sources)
            if role_filter:
                like_where.append("m.role IN (" + ",".join("?" for _ in role_filter) + ")")
                like_params.extend(role_filter)
            fallback_sql = f"""
                SELECT m.id, m.session_id, m.role, m.content, m.timestamp,
                       m.tool_name, s.source, s.model,
                       s.started_at AS session_started
                FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE {' AND '.join(like_where)}
                ORDER BY m.timestamp {'' if sort == 'oldest' else 'DESC'}
                LIMIT ? OFFSET ?
            """
            cursor = await connection.execute(
                fallback_sql, [*like_params, limit, offset]
            )
            rows = await cursor.fetchall()
        results: list[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["content"] = self._decode_message_row(row).get("content")
            item.setdefault("snippet", item.get("content") or "")
            item.pop("content", None)
            results.append(item)
        if result_fields is not None:
            results = [
                {field: item[field] for field in result_fields if field in item}
                for item in results
            ]
        return results

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
        cls, fields: Optional[Collection[str]]
    ) -> Optional[Tuple[str, ...]]:
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

    async def list_sessions_rich(
        self,
        source: str = None,
        sources: List[str] = None,
        exclude_sources: List[str] = None,
        cwd_prefix: str = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        min_message_count: int = 0,
        project_compression_tips: bool = True,
        order_by_last_active: bool = False,
        include_archived: bool = False,
        archived_only: bool = False,
        compact_rows: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        """Return lightweight recent-session rows for async browse surfaces."""
        connection = await self._get_connection()
        where: list[str] = []
        params: list[Any] = []
        if not include_children:
            where.extend([_LISTABLE_CHILD_SQL, f"{_delegate_from_json('s.model_config')} IS NULL"])
        include_sources = [source] if source else list(sources or [])
        if include_sources:
            where.append("s.source IN (" + ",".join("?" for _ in include_sources) + ")")
            params.extend(include_sources)
        if exclude_sources:
            where.append("s.source NOT IN (" + ",".join("?" for _ in exclude_sources) + ")")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where.append(clause)
            params.extend(clause_params)
        if min_message_count:
            where.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where.append("s.archived = 1")
        elif not include_archived:
            where.append("s.archived = 0")
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        order_sql = "last_active DESC, s.started_at DESC, s.id DESC" if order_by_last_active else "s.started_at DESC, s.id DESC"
        prompt_select = (
            "NULL AS _system_prompt_resolved"
            if compact_rows
            else "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            "" if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )
        query_sql = f"""
            SELECT s.*,
                   {prompt_select},
                   COALESCE((SELECT MAX(m.timestamp) FROM messages m
                             WHERE m.session_id = s.id), s.started_at) AS last_active,
                   (SELECT m.content FROM messages m
                    WHERE m.session_id = s.id AND m.role = 'user' AND m.active = 1
                    ORDER BY m.id LIMIT 1) AS _preview_raw
            FROM sessions s
            {prompt_join}
            {where_sql}
            ORDER BY {order_sql} LIMIT ? OFFSET ?
        """
        cursor = await connection.execute(query_sql, [*params, max(0, int(limit)), max(0, int(offset))])
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            item = self._session_row_dict(row)
            item["preview"] = _shape_preview(self._decode_content(item.pop("_preview_raw", "")))
            if compact_rows:
                item.pop("system_prompt", None)
                item.pop("system_prompt_hash", None)
            result.append(item)
        return result

    async def get_compression_failure_cooldown(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_failure_cooldown_until, "
                "compression_failure_error FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        if row is None or row["compression_failure_cooldown_until"] is None:
            return None
        cooldown_until = float(row["compression_failure_cooldown_until"])
        remaining_seconds = cooldown_until - time.time()
        if remaining_seconds <= 0:
            return None
        return {
            "cooldown_until": cooldown_until,
            "remaining_seconds": remaining_seconds,
            "error": row["compression_failure_error"],
        }

    async def get_compression_failure_cooldown_row(
        self, session_id: str
    ) -> Dict[str, Any]:
        """Return the exact stored cooldown columns without expiry filtering."""
        if not session_id:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_failure_cooldown_until, "
                "compression_failure_error FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        if row is None:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        deadline = row["compression_failure_cooldown_until"]
        return {
            "session_exists": True,
            "cooldown_until": float(deadline) if deadline is not None else None,
            "error": row["compression_failure_error"],
        }

    async def restore_compression_failure_cooldown_row(
        self, session_id: str, snapshot: Dict[str, Any]
    ) -> None:
        """Restore and verify an exact cooldown-row snapshot."""
        expected_exists = bool(snapshot.get("session_exists", False))
        if not expected_exists:
            actual = await self.get_compression_failure_cooldown_row(session_id)
            if actual.get("session_exists", False):
                raise RuntimeError(
                    "cannot restore absent compression cooldown row: session now exists"
                )
            return

        deadline = snapshot.get("cooldown_until")
        error = snapshot.get("error")

        async def _restore(connection):
            cursor = await connection.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (deadline, error, session_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"compression cooldown rollback session missing: {session_id}"
                )

        await self._write(_restore)
        actual = await self.get_compression_failure_cooldown_row(session_id)
        expected = {
            "session_exists": True,
            "cooldown_until": float(deadline) if deadline is not None else None,
            "error": error,
        }
        if actual != expected:
            raise RuntimeError(
                "compression cooldown rollback verification failed: "
                f"expected={expected!r}, actual={actual!r}"
            )

    async def record_compression_failure_cooldown(
        self, session_id: str, cooldown_until: float, error: Optional[str] = None
    ) -> None:
        if not session_id:
            return

        async def _record(connection):
            await connection.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (cooldown_until, error, session_id),
            )

        await self._write(_record)

    async def clear_compression_failure_cooldown(self, session_id: str) -> None:
        if not session_id:
            return

        async def _clear(connection):
            await connection.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
                "compression_failure_error = NULL WHERE id = ?",
                (session_id,),
            )

        await self._write(_clear)

    async def get_compression_fallback_streak(self, session_id: str) -> int:
        if not session_id:
            return 0
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_fallback_streak FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        try:
            return max(0, int(row["compression_fallback_streak"] or 0)) if row else 0
        except (TypeError, ValueError):
            return 0

    async def set_compression_fallback_streak(
        self, session_id: str, streak: int
    ) -> None:
        if not session_id:
            return

        async def _set(connection):
            await connection.execute(
                "UPDATE sessions SET compression_fallback_streak = ? WHERE id = ?",
                (max(0, int(streak)), session_id),
            )

        await self._write(_set)

    async def get_compression_ineffective_count(self, session_id: str) -> int:
        if not session_id:
            return 0
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_ineffective_count FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        try:
            return max(0, int(row["compression_ineffective_count"] or 0)) if row else 0
        except (TypeError, ValueError):
            return 0

    async def set_compression_ineffective_count(
        self, session_id: str, count: int
    ) -> None:
        if not session_id:
            return

        async def _set(connection):
            await connection.execute(
                "UPDATE sessions SET compression_ineffective_count = ? WHERE id = ?",
                (max(0, int(count)), session_id),
            )

        await self._write(_set)

    async def get_conversation_root(self, session_id: str) -> str:
        """Resolve a session lineage root without using ``SessionDB``'s lock."""
        if not session_id:
            return session_id
        connection = await self._get_connection()
        current = session_id
        seen = set()
        for _ in range(100):
            if not current or current in seen:
                break
            seen.add(current)
            row = await (
                await connection.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                )
            ).fetchone()
            if row is None or not row["parent_session_id"]:
                break
            current = row["parent_session_id"]
        return current or session_id

    async def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
        repair_alternation: bool = False,
        include_row_ids: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load a conversation through the native async SQLite connection."""
        if not session_id:
            return []
        connection = await self._get_connection()
        session_ids = [session_id]
        if include_ancestors:
            current = session_id
            seen = set()
            while current and current not in seen:
                seen.add(current)
                row = await (
                    await connection.execute(
                        "SELECT parent_session_id FROM sessions WHERE id = ?",
                        (current,),
                    )
                ).fetchone()
                parent = row["parent_session_id"] if row is not None else None
                if not parent:
                    break
                session_ids.insert(0, parent)
                current = parent

        placeholders = ",".join("?" for _ in session_ids)
        active_clause = "" if include_inactive else " AND active = 1"
        rows = await (
            await connection.execute(
                f"SELECT {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders})"
                f"{active_clause} ORDER BY id",
                tuple(session_ids),
            )
        ).fetchall()
        return self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=include_ancestors,
            repair_alternation=repair_alternation,
            include_row_ids=include_row_ids,
        )

    async def find_live_compression_child(
        self, parent_session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the one unambiguous live compression child, if it exists."""
        if not parent_session_id:
            return None
        connection = await self._get_connection()
        parent = await (
            await connection.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (parent_session_id,),
            )
        ).fetchone()
        if (
            parent is None
            or parent["ended_at"] is None
            or parent["end_reason"] != "compression"
        ):
            return None
        rows = await (
            await connection.execute(
                """SELECT s.*, COALESCE(sp.prompt, s.system_prompt)
                          AS _system_prompt_resolved
                   FROM sessions s LEFT JOIN system_prompts sp
                     ON sp.hash = s.system_prompt_hash
                   WHERE s.parent_session_id = ?
                     AND s.ended_at IS NULL
                     AND json_extract(COALESCE(s.model_config, '{}'), '$._branched_from') IS NULL
                     AND json_extract(COALESCE(s.model_config, '{}'), '$._delegate_from') IS NULL
                     AND COALESCE(s.source, '') != 'tool'
                   ORDER BY s.started_at ASC
                   LIMIT 2""",
                (parent_session_id,),
            )
        ).fetchall()
        return self._session_row_dict(rows[0]) if len(rows) == 1 else None

    async def get_session_title(self, session_id: str) -> Optional[str]:
        """Read a title without using the synchronous connection."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
        ).fetchone()
        return row["title"] if row is not None else None

    async def get_next_title_in_lineage(self, base_title: str) -> str:
        """Return the next generated continuation title."""
        match = re.match(r"^(.*?) #(\d+)$", base_title)
        base = match.group(1) if match else base_title
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
        ).fetchall()
        titles = [row["title"] for row in rows]
        if not titles:
            return base
        suffixes = [
            int(found.group(1))
            for title in titles
            if (found := re.match(rf"^{re.escape(base)} #(\d+)$", title or ""))
        ]
        return f"{base} #{max(suffixes, default=1) + 1}"

    async def set_session_title(self, session_id: str, title: str) -> bool:
        """Set a generated continuation title using the async write path."""
        async def _set(connection):
            cursor = await connection.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )
            return cursor.rowcount > 0

        return bool(await self._write(_set))

    async def publish_compression_child(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        source: str,
        messages: List[Dict[str, Any]],
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        cwd: str = None,
        profile_name: str = None,
        compression_lock_holder: str = None,
        require_compression_lease: bool = True,
    ) -> None:
        """Publish a compressed continuation without a sync SQLite bridge."""
        if not messages:
            raise RuntimeError("Compression child handoff must not be empty")

        async def _publish(connection):
            lock_row = await (
                await connection.execute(
                    "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                    (parent_session_id,),
                )
            ).fetchone()
            if require_compression_lease and (
                lock_row is None
                or not compression_lock_holder
                or lock_row["holder"] != compression_lock_holder
                or float(lock_row["expires_at"]) <= time.time()
            ):
                raise CompressionSessionBusyError(
                    f"Compression lease lost before publication: {parent_session_id}"
                )
            parent = await (
                await connection.execute(
                    """SELECT ended_at, cwd, git_branch, git_repo_root,
                              user_id, session_key, chat_id, chat_type,
                              thread_id, display_name, origin_json, profile_name
                       FROM sessions WHERE id = ?""",
                    (parent_session_id,),
                )
            ).fetchone()
            if parent is None:
                raise RuntimeError(f"Compression parent not found: {parent_session_id}")
            if parent["ended_at"] is not None:
                raise RuntimeError(
                    f"Compression parent already ended: {parent_session_id}"
                )
            system_prompt_hash = await self._store_system_prompt(
                connection, system_prompt
            )
            await connection.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt,
                   system_prompt_hash,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, started_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    child_session_id,
                    source,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    cwd or parent["cwd"],
                    parent["git_branch"],
                    parent["git_repo_root"],
                    profile_name or parent["profile_name"],
                    parent["user_id"],
                    parent["session_key"],
                    parent["chat_id"],
                    parent["chat_type"],
                    parent["thread_id"],
                    parent["display_name"],
                    parent["origin_json"],
                    time.time(),
                ),
            )
            now_ts = time.time()
            tool_calls_total = 0
            for message in messages:
                role = message.get("role", "unknown")
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, str):
                    try:
                        tool_calls = json.loads(tool_calls)
                    except (json.JSONDecodeError, TypeError):
                        tool_calls = []
                timestamp = message.get("timestamp", now_ts)
                try:
                    timestamp = float(
                        timestamp.timestamp()
                        if hasattr(timestamp, "timestamp")
                        else timestamp
                    )
                except (TypeError, ValueError):
                    timestamp = now_ts
                reasoning_details = (
                    message.get("reasoning_details") if role == "assistant" else None
                )
                codex_reasoning_items = (
                    message.get("codex_reasoning_items") if role == "assistant" else None
                )
                codex_message_items = (
                    message.get("codex_message_items") if role == "assistant" else None
                )
                await connection.execute(
                    """INSERT INTO messages (session_id, role, content, tool_call_id,
                       tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        child_session_id, role,
                        self._encode_content(message.get("content")),
                        message.get("tool_call_id"),
                        json.dumps(tool_calls) if tool_calls else None,
                        _scrub_surrogates(message.get("tool_name")),
                        message.get("effect_disposition"), timestamp,
                        message.get("token_count"), message.get("finish_reason"),
                        _scrub_surrogates(message.get("reasoning")) if role == "assistant" else None,
                        _scrub_surrogates(message.get("reasoning_content")) if role == "assistant" else None,
                        json.dumps(reasoning_details) if reasoning_details else None,
                        json.dumps(codex_reasoning_items) if codex_reasoning_items else None,
                        json.dumps(codex_message_items) if codex_message_items else None,
                        message.get("platform_message_id") or message.get("message_id"),
                        1 if message.get("observed") else 0, 1,
                        _scrub_surrogates(message.get("api_content"))
                        if isinstance(message.get("api_content"), str) else None,
                        _scrub_surrogates(message.get("display_kind"))
                        if isinstance(message.get("display_kind"), str) else None,
                        self._encode_display_metadata(message.get("display_metadata")),
                    ),
                )
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else int(tool_calls is not None)
                )
                now_ts = max(now_ts + 1e-6, timestamp + 1e-6)
            await connection.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (len(messages), tool_calls_total, child_session_id),
            )
            updated = await connection.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'compression' "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), parent_session_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    f"Compression parent changed during publication: {parent_session_id}"
                )

        await self._write(_publish)

    async def flush_token_counts(self, timeout: float = 5.0) -> bool:
        """Token deltas still use their own async accounting path; no turn-blocking drain."""
        del timeout
        return True

    async def close(self) -> None:
        self._closed = True
        connection = self._connection
        self._connection = None
        if connection is not None:
            close_task = asyncio.create_task(connection.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                await asyncio.shield(close_task)
                raise
