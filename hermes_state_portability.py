"""Session portability helpers for the native-async ``SessionDB``.

This retains the upstream mixin path and the session/skill behavior that is
part of the library distribution. The scheduler-only cron-run listing remains
outside this reduced package.

Mixin contract: this class defines no ``__init__`` and owns no runtime state.
Methods use the async connection, transaction, and row helpers supplied by
``hermes_state.SessionDB``. This module must not import ``hermes_state``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent.skill_commands import SKILL_SCAFFOLD_SQL_LIKE
from hermes_state_common import (
    SCHEMA_SQL,
    _PREVIEW_RAW_SELECT,
    _shape_preview,
    _sql_session_last_active,
)


logger = logging.getLogger("hermes_state")


class SessionPortabilityMixin:
    """Native-async retained session export/import behavior for SessionDB."""

    @classmethod
    async def _compact_session_cols(cls) -> str:
        """Return the upstream compact projection without blocking SQLite."""
        if cls._session_compact_cols_sql is None:
            declared = (
                await cls._parse_schema_columns(SCHEMA_SQL)  # type: ignore[unresolved-attribute]
            )["sessions"]
            cls._session_compact_cols_sql = ", ".join(
                f"s.{name}"
                for name in declared
                if name not in cls._SESSION_COMPACT_EXCLUDED  # type: ignore[unresolved-attribute]
            )
        return cls._session_compact_cols_sql

    async def _get_session_rich_rows_batch(
        self,
        session_ids,
        compact_rows: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Fetch enriched session rows in bounded native-async batches."""
        ids = [session_id for session_id in session_ids if session_id]
        if not ids:
            return {}
        chunk_size = 900
        if len(ids) > chunk_size:
            result: dict[str, dict[str, Any]] = {}
            for start in range(0, len(ids), chunk_size):
                result.update(
                    await self._get_session_rich_rows_batch(
                        ids[start : start + chunk_size],
                        compact_rows=compact_rows,
                    )
                )
            return result

        await self.flush_token_counts()  # type: ignore[unresolved-attribute]
        select = await self._compact_session_cols() if compact_rows else "s.*"
        placeholders = ",".join("?" for _ in ids)
        prompt_select = (
            ""
            if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            ""
            if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )
        query = f"""
            SELECT {select}{prompt_select},
                COALESCE(
                    (SELECT {_PREVIEW_RAW_SELECT}
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            {prompt_join}
            WHERE s.id IN ({placeholders})
        """
        rows = await self._read_fetchall(  # type: ignore[unresolved-attribute]
            query,
            ids,
        )
        result = {}
        for row in rows:
            session = self._session_row_dict(  # type: ignore[unresolved-attribute]
                row
            )
            session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
            result[session["id"]] = session
        return result

    async def _get_session_rich_row(
        self,
        session_id: str,
        compact_rows: bool = False,
    ) -> dict[str, Any] | None:
        """Return one enriched row with the same shape as session listings."""
        return (
            await self._get_session_rich_rows_batch(
                [session_id],
                compact_rows=compact_rows,
            )
        ).get(session_id)

    async def get_session_rich_row(
        self,
        session_id: str,
        compact_rows: bool = False,
    ) -> dict[str, Any] | None:
        """Public wrapper for the upstream single-session rich-row API."""
        return await self._get_session_rich_row(
            session_id,
            compact_rows=compact_rows,
        )

    async def distinct_session_cwds(
        self,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Return distinct non-empty session working directories and usage."""
        where = "cwd IS NOT NULL AND TRIM(cwd) != ''"
        if not include_archived:
            where += " AND archived = 0"
        rows = await self._read_fetchall(  # type: ignore[unresolved-attribute]
            "SELECT cwd AS cwd, COUNT(*) AS sessions, "
            "MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
            f"FROM sessions WHERE {where} GROUP BY cwd"
        )
        return [
            {
                "cwd": row["cwd"],
                "sessions": int(row["sessions"] or 0),
                "last_active": float(row["last_active"] or 0),
            }
            for row in rows
        ]

    async def list_skill_scaffolded_sessions(
        self,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return titled sessions whose first user turn invoked a skill."""
        rows = await self._read_fetchall(  # type: ignore[unresolved-attribute]
            """
            SELECT s.id, s.title, m.content
            FROM sessions s
            JOIN messages m ON m.id = (
                SELECT m2.id FROM messages m2
                WHERE m2.session_id = s.id AND m2.role = 'user'
                  AND m2.content IS NOT NULL
                ORDER BY m2.timestamp, m2.id LIMIT 1
            )
            WHERE s.title IS NOT NULL AND m.content LIKE ?
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (SKILL_SCAFFOLD_SQL_LIKE, int(limit)),
        )
        return [dict(row) for row in rows]

    async def get_first_assistant_text(self, session_id: str) -> str:
        """Return the session's first assistant reply as plain text."""
        async with self._read_ctx() as connection:  # type: ignore[unresolved-attribute]
            async with connection.execute(
                "SELECT content FROM messages "
                "WHERE session_id = ? AND role = 'assistant' "
                "AND content IS NOT NULL "
                "ORDER BY timestamp, id LIMIT 1",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return ""
        decoded = self._decode_content(  # type: ignore[unresolved-attribute]
            row["content"]
        )
        return decoded if isinstance(decoded, str) else ""

    async def export_session(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Export a single session with all active messages as a dict."""
        session = await self.get_session(  # type: ignore[unresolved-attribute]
            session_id
        )
        if not session:
            return None
        await self.assert_export_safe(session_id)  # type: ignore[unresolved-attribute]
        messages = await self.get_messages(  # type: ignore[unresolved-attribute]
            session_id
        )
        return {**session, "messages": messages}

    async def export_session_lineage(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Export a compression lineage as one logical session dict."""
        lineage_ids = await self.get_compression_lineage(  # type: ignore[unresolved-attribute]
            session_id
        )
        if not lineage_ids:
            return None
        segments = []
        for lineage_id in lineage_ids:
            segment = await self.export_session(lineage_id)
            if segment:
                segments.append(segment)
        if not segments:
            return None
        base = dict(segments[-1])
        total_messages = sum(
            len(segment.get("messages") or []) for segment in segments
        )
        base["segments"] = segments
        base["lineage_session_ids"] = [segment["id"] for segment in segments]
        base["message_count"] = total_messages
        base["messages"] = [
            message
            for segment in segments
            for message in (segment.get("messages") or [])
        ]
        return base

    async def export_all(
        self,
        source: str | None = None,  # type: ignore[invalid-parameter-default]
    ) -> list[dict[str, Any]]:
        """Export every matching session and its active messages."""
        sessions = await self.search_sessions(  # type: ignore[unresolved-attribute]
            source=source,
            limit=100000,
        )
        results = []
        for session in sessions:
            await self.assert_export_safe(session["id"])  # type: ignore[unresolved-attribute]
            messages = await self.get_messages(  # type: ignore[unresolved-attribute]
                session["id"]
            )
            results.append({**session, "messages": messages})
        return results

    @staticmethod
    def _import_text_or_none(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise ValueError(f"{field} must be a string")

    @staticmethod
    def _import_json_object_or_none(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{field} must be a JSON object")
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object")
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be JSON serializable") from exc

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _import_int_or_none(value: Any, field: str) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc

    @staticmethod
    def _int_or_default(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _reasoning_json_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    @staticmethod
    def _import_error(index: int, session_id: str, error: str) -> dict[str, Any]:
        item: dict[str, Any] = {"index": index, "error": error}
        if session_id:
            item["session_id"] = session_id
        return item

    async def import_sessions(
        self,
        sessions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Import sessions exported by :meth:`export_session` or ``export_all``."""
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        if len(sessions) > self._IMPORT_MAX_SESSIONS:  # type: ignore[unresolved-attribute]
            raise ValueError(
                f"sessions must contain at most {self._IMPORT_MAX_SESSIONS} entries"  # type: ignore[unresolved-attribute]
            )

        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        total_messages = 0
        total_bytes = 0
        session_text_fields = (
            "source",
            "user_id",
            "model",
            "system_prompt",
            "end_reason",
            "cwd",
            "git_branch",
            "git_repo_root",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
        )
        message_text_fields = (
            "role",
            "tool_call_id",
            "tool_name",
            "effect_disposition",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "platform_message_id",
            "message_id",
        )

        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                errors.append(
                    self._import_error(index, "", "session must be an object")
                )
                continue
            session_id = str(raw.get("id") or "").strip()
            if not session_id:
                errors.append(
                    self._import_error(index, "", "session id is required")
                )
                continue
            if session_id in seen_ids:
                errors.append(
                    self._import_error(index, session_id, "duplicate session id")
                )
                continue
            messages = raw.get("messages") or []
            if not isinstance(messages, list):
                errors.append(
                    self._import_error(index, session_id, "messages must be a list")
                )
                continue
            if len(messages) > self._IMPORT_MAX_MESSAGES_PER_SESSION:  # type: ignore[unresolved-attribute]
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages exceeds the per-session import limit",
                    )
                )
                continue
            if any(not isinstance(message, dict) for message in messages):
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages must contain only objects",
                    )
                )
                continue

            try:
                session_bytes = len(
                    json.dumps(
                        raw,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            except (TypeError, ValueError):
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "session must be JSON serializable",
                    )
                )
                continue
            if session_bytes > self._IMPORT_MAX_SESSION_BYTES:  # type: ignore[unresolved-attribute]
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "session exceeds the import size limit",
                    )
                )
                continue
            total_bytes += session_bytes
            if total_bytes > self._IMPORT_MAX_TOTAL_BYTES:  # type: ignore[unresolved-attribute]
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "import exceeds the total size limit",
                    )
                )
                continue

            try:
                clean_session = dict(raw)
                clean_session["id"] = session_id
                clean_session["model_config"] = self._import_json_object_or_none(
                    clean_session.get("model_config"),
                    "model_config",
                )
                clean_session["parent_session_id"] = self._import_text_or_none(
                    clean_session.get("parent_session_id"),
                    "parent_session_id",
                )
                for field in session_text_fields:
                    clean_session[field] = self._import_text_or_none(
                        clean_session.get(field),
                        field,
                    )

                clean_messages: list[dict[str, Any]] = []
                for message_index, message in enumerate(messages):
                    clean_message = dict(message)
                    role = clean_message.get("role")
                    if not isinstance(role, str) or not role:
                        raise ValueError(
                            f"messages[{message_index}].role must be a non-empty string"
                        )
                    for field in message_text_fields:
                        if field == "role":
                            continue
                        clean_message[field] = self._import_text_or_none(
                            clean_message.get(field),
                            field,
                        )
                    clean_message["token_count"] = self._import_int_or_none(
                        clean_message.get("token_count"),
                        "token_count",
                    )
                    clean_messages.append(clean_message)
            except ValueError as exc:
                errors.append(self._import_error(index, session_id, str(exc)))
                continue

            total_messages += len(clean_messages)
            if total_messages > self._IMPORT_MAX_TOTAL_MESSAGES:  # type: ignore[unresolved-attribute]
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages exceeds the total import limit",
                    )
                )
                continue
            seen_ids.add(session_id)
            normalized.append(
                {
                    "index": index,
                    "session": clean_session,
                    "messages": clean_messages,
                }
            )

        if errors:
            return {
                "ok": False,
                "imported": 0,
                "skipped": 0,
                "detached": 0,
                "errors": errors,
            }

        async def _import(connection):
            imported_ids: list[str] = []
            skipped_ids: list[str] = []
            parent_updates: list[tuple[str, str]] = []
            detached = 0

            for item in normalized:
                raw = item["session"]
                messages = item["messages"]
                session_id = str(raw.get("id") or "").strip()
                async with connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                    (session_id,),
                ) as cursor:
                    exists = await cursor.fetchone()
                if exists:
                    skipped_ids.append(session_id)
                    continue

                started_at = self._float_or_none(raw.get("started_at"))
                if started_at is None:
                    started_at = time.time()
                archived = 1 if raw.get("archived") else 0
                system_prompt_hash = await self._store_system_prompt(  # type: ignore[unresolved-attribute]
                    connection,
                    raw.get("system_prompt"),
                )

                await connection.execute(
                    """INSERT INTO sessions (
                           id, source, user_id, model, model_config, system_prompt,
                           system_prompt_hash,
                           parent_session_id, started_at, ended_at, end_reason,
                           message_count, tool_call_count, input_tokens, output_tokens,
                           cache_read_tokens, cache_write_tokens, reasoning_tokens,
                           cwd, git_branch, git_repo_root,
                           billing_provider, billing_base_url, billing_mode,
                           estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                           pricing_version, title, api_call_count, archived
                       )
                       VALUES (
                           :id, :source, :user_id, :model, :model_config,
                           NULL, :system_prompt_hash, NULL, :started_at, :ended_at,
                           :end_reason, 0, 0, :input_tokens, :output_tokens,
                           :cache_read_tokens, :cache_write_tokens,
                           :reasoning_tokens, :cwd, :git_branch, :git_repo_root,
                           :billing_provider, :billing_base_url, :billing_mode,
                           :estimated_cost_usd, :actual_cost_usd, :cost_status,
                           :cost_source, :pricing_version, :title,
                           :api_call_count, :archived
                       )""",
                    {
                        "id": session_id,
                        "source": str(raw.get("source") or "import"),
                        "user_id": raw.get("user_id"),
                        "model": raw.get("model"),
                        "model_config": raw.get("model_config"),
                        "system_prompt_hash": system_prompt_hash,
                        "started_at": started_at,
                        "ended_at": self._float_or_none(raw.get("ended_at")),
                        "end_reason": raw.get("end_reason"),
                        "input_tokens": self._int_or_default(raw.get("input_tokens")),
                        "output_tokens": self._int_or_default(raw.get("output_tokens")),
                        "cache_read_tokens": self._int_or_default(
                            raw.get("cache_read_tokens")
                        ),
                        "cache_write_tokens": self._int_or_default(
                            raw.get("cache_write_tokens")
                        ),
                        "reasoning_tokens": self._int_or_default(
                            raw.get("reasoning_tokens")
                        ),
                        "cwd": raw.get("cwd"),
                        "git_branch": raw.get("git_branch"),
                        "git_repo_root": raw.get("git_repo_root"),
                        "billing_provider": raw.get("billing_provider"),
                        "billing_base_url": raw.get("billing_base_url"),
                        "billing_mode": raw.get("billing_mode"),
                        "estimated_cost_usd": self._float_or_none(
                            raw.get("estimated_cost_usd")
                        ),
                        "actual_cost_usd": self._float_or_none(
                            raw.get("actual_cost_usd")
                        ),
                        "cost_status": raw.get("cost_status"),
                        "cost_source": raw.get("cost_source"),
                        "pricing_version": raw.get("pricing_version"),
                        "title": raw.get("title"),
                        "api_call_count": self._int_or_default(
                            raw.get("api_call_count")
                        ),
                        "archived": archived,
                    },
                )

                sanitized_messages: list[dict[str, Any]] = []
                for message in messages:
                    clean = dict(message)
                    for key in (
                        "reasoning_details",
                        "codex_reasoning_items",
                        "codex_message_items",
                    ):
                        clean[key] = self._reasoning_json_value(clean.get(key))
                    sanitized_messages.append(clean)

                imported_messages, imported_tool_calls = (
                    await self._insert_message_rows(  # type: ignore[unresolved-attribute]
                        connection,
                        session_id,
                        sanitized_messages,
                    )
                )
                await connection.execute(
                    "UPDATE sessions SET message_count = ?, "
                    "tool_call_count = ? WHERE id = ?",
                    (imported_messages, imported_tool_calls, session_id),
                )

                parent_id = str(raw.get("parent_session_id") or "").strip()
                if parent_id:
                    parent_updates.append((session_id, parent_id))
                imported_ids.append(session_id)

            parent_by_child = dict(parent_updates)

            async def _would_create_cycle(session_id: str, parent_id: str) -> bool:
                seen = {session_id}
                current = parent_id
                while current:
                    if current in seen:
                        return True
                    seen.add(current)
                    if current in parent_by_child:
                        current = parent_by_child[current]
                        continue
                    async with connection.execute(
                        "SELECT parent_session_id FROM sessions "
                        "WHERE id = ? LIMIT 1",
                        (current,),
                    ) as cursor:
                        row = await cursor.fetchone()
                    if row is None:
                        return False
                    current = row["parent_session_id"]
                return False

            for session_id, parent_id in parent_updates:
                async with connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                    (parent_id,),
                ) as cursor:
                    parent_exists = await cursor.fetchone()
                if parent_exists and not await _would_create_cycle(
                    session_id,
                    parent_id,
                ):
                    await connection.execute(
                        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                        (parent_id, session_id),
                    )
                else:
                    parent_by_child.pop(session_id, None)
                    detached += 1

            return {
                "ok": True,
                "imported": len(imported_ids),
                "skipped": len(skipped_ids),
                "detached": detached,
                "imported_ids": imported_ids,
                "skipped_ids": skipped_ids,
                "errors": [],
            }

        return await self._execute_write(_import)  # type: ignore[unresolved-attribute]
