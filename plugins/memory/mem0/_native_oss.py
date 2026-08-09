"""Native-async runtime components for the retained Mem0 OSS backend.

The upstream ``mem0ai`` package implements its SQLite history store with
``sqlite3`` and a ``threading.Lock``.  This module keeps that schema and result
contract while moving database lifecycle and queries behind ``aiosqlite``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
import os
import re
from typing import Any, Coroutine
import uuid
import warnings

import aiosqlite

logger = logging.getLogger(__name__)


async def _finish_cleanup(
    cleanup: Coroutine[Any, Any, None], *, error_message: str
) -> None:
    """Finish one owned-resource cleanup before preserving cancellation."""
    cleanup_task = asyncio.create_task(cleanup)
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        try:
            await cleanup_task
        except Exception:
            logger.exception(error_message)
        raise


def _extract_json(text: str) -> str:
    """Mirror ``mem0.memory.utils.extract_json`` without importing sync runtime."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)

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


class OpenAIEmbedding:
    """Native-async equivalent of Mem0 2.0.10's OpenAI embedder."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model = self.config.get("model") or "text-embedding-3-small"
        self._pass_dimensions_to_api = self.config.get("embedding_dims") is not None
        self.embedding_dims = self.config.get("embedding_dims") or 1536
        self._client: Any = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._initialize_lock:
            if self._client is not None:
                return self._client
            if self._closed:
                raise RuntimeError("Cannot use a closed OpenAIEmbedding")

            from openai import AsyncOpenAI

            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            legacy_base_url = os.getenv("OPENAI_API_BASE")
            base_url = (
                self.config.get("openai_base_url")
                or legacy_base_url
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )
            if legacy_base_url:
                warnings.warn(
                    "The environment variable 'OPENAI_API_BASE' is deprecated "
                    "and will be removed in the 0.1.80. Please use "
                    "'OPENAI_BASE_URL' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            return self._client

    async def embed(
        self, text: str, memory_action: str | None = None  # noqa: ARG002
    ) -> list[float]:
        client = await self._get_client()
        request: dict[str, Any] = {
            "input": [text.replace("\n", " ")],
            "model": self.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            request["dimensions"] = self.embedding_dims
        response = await client.embeddings.create(**request)
        return response.data[0].embedding

    async def embed_batch(
        self, texts: list[str], memory_action: str = "add"  # noqa: ARG002
    ) -> list[list[float]]:
        if not texts:
            return []
        client = await self._get_client()
        normalized = [text.replace("\n", " ") for text in texts]
        embeddings: list[list[float]] = []
        for start in range(0, len(normalized), 100):
            request: dict[str, Any] = {
                "input": normalized[start : start + 100],
                "model": self.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                request["dimensions"] = self.embedding_dims
            response = await client.embeddings.create(**request)
            embeddings.extend(
                item.embedding for item in sorted(response.data, key=lambda item: item.index)
            )
        return embeddings

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            if client is not None:
                await _finish_cleanup(
                    client.close(),
                    error_message="Mem0 OpenAI embedder cleanup failed during cancellation",
                )


class OllamaEmbedding:
    """Native-async equivalent of Mem0 2.0.10's Ollama embedder."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model = self.config.get("model") or "nomic-embed-text"
        self.embedding_dims = self.config.get("embedding_dims") or 512
        self._client: Any = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._initialize_lock:
            if self._client is not None:
                return self._client
            if self._closed:
                raise RuntimeError("Cannot use a closed OllamaEmbedding")

            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise ImportError(
                    "The 'ollama' library is required. Install the mem0 extra."
                ) from exc

            client = AsyncClient(host=self.config.get("ollama_base_url"))
            try:
                response = await client.list()
                models = _response_value(response, "models", []) or []
                target = self._normalize_model_name(self.model)
                found = False
                for model in models:
                    name = _response_value(model, "name", "") or _response_value(
                        model, "model", ""
                    )
                    if name and self._normalize_model_name(name) == target:
                        found = True
                        break
                if not found:
                    await client.pull(self.model)
            except BaseException:
                try:
                    await _finish_cleanup(
                        client.close(),
                        error_message=(
                            "Mem0 Ollama embedder cleanup failed during initialization "
                            "cancellation"
                        ),
                    )
                except Exception:
                    logger.exception("Failed to close Mem0 Ollama embedder client")
                raise

            self._client = client
            return client

    async def embed(
        self, text: str, memory_action: str | None = None  # noqa: ARG002
    ) -> list[float]:
        client = await self._get_client()
        response = await client.embed(model=self.model, input=text)
        embeddings = _response_value(response, "embeddings", []) or []
        if not embeddings:
            raise ValueError(
                f"Ollama embed() returned no embeddings for model '{self.model}'"
            )
        return embeddings[0]

    async def embed_batch(
        self, texts: list[str], memory_action: str = "add"  # noqa: ARG002
    ) -> list[list[float]]:
        if not texts:
            return []
        client = await self._get_client()
        response = await client.embed(model=self.model, input=texts)
        embeddings = _response_value(response, "embeddings", []) or []
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama embed() returned {len(embeddings)} embeddings for "
                f"{len(texts)} texts using model '{self.model}'"
            )
        return embeddings

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            if client is not None:
                await _finish_cleanup(
                    client.close(),
                    error_message="Mem0 Ollama embedder cleanup failed during cancellation",
                )


class OpenAILLM:
    """Native-async equivalent of Mem0 2.0.10's OpenAI LLM."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model = self.config.get("model") or "gpt-5-mini"
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.top_p = self.config.get("top_p", 0.1)
        self._client: Any = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    def _is_reasoning_model(self) -> bool:
        explicit = self.config.get("is_reasoning_model")
        if explicit is not None:
            return bool(explicit)
        base_model = self.model.lower().rsplit("/", 1)[-1]
        if base_model in {
            "o1",
            "o1-preview",
            "o3-mini",
            "o3",
            "gpt-5",
            "gpt-5o",
            "gpt-5o-mini",
            "gpt-5o-micro",
        }:
            return True
        return any(
            base_model.startswith(prefix)
            for prefix in ("o1-", "o1.", "o3-", "o3.")
        )

    def _supported_params(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        if self._is_reasoning_model():
            params: dict[str, Any] = {"messages": messages}
            reasoning_effort = self.config.get("reasoning_effort")
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            return params
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            (
                "max_completion_tokens"
                if self.model.lower().rsplit("/", 1)[-1].startswith("gpt-5")
                else "max_tokens"
            ): self.max_tokens,
        }
        params.update(kwargs)
        params["messages"] = messages
        return params

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._initialize_lock:
            if self._client is not None:
                return self._client
            if self._closed:
                raise RuntimeError("Cannot use a closed OpenAILLM")

            from openai import AsyncOpenAI

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                api_key = openrouter_key
                base_url = (
                    self.config.get("openrouter_base_url")
                    or os.getenv("OPENROUTER_API_BASE")
                    or "https://openrouter.ai/api/v1"
                )
            else:
                api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
                base_url = (
                    self.config.get("openai_base_url")
                    or os.getenv("OPENAI_BASE_URL")
                    or "https://api.openai.com/v1"
                )
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            return self._client

    @staticmethod
    def _parse_response(response: Any, tools: list[dict[str, Any]] | None) -> Any:
        message = response.choices[0].message
        if not tools:
            return message.content
        parsed = {"content": message.content, "tool_calls": []}
        for tool_call in message.tool_calls or []:
            parsed["tool_calls"].append(
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(
                        _extract_json(tool_call.function.arguments)
                    ),
                }
            )
        return parsed

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        response_format: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Any:
        callback = self.config.get("response_callback")
        if callback is not None and not inspect.iscoroutinefunction(callback):
            raise RuntimeError(
                "Mem0 OpenAI LLM response_callback must provide a native-async "
                "response_callback."
            )

        params = self._supported_params(messages, **kwargs)
        params.update({"model": self.model, "messages": messages})
        if os.getenv("OPENROUTER_API_KEY"):
            models = self.config.get("models")
            if models:
                params["models"] = models
                params["route"] = self.config.get("route", "fallback")
                params.pop("model")
            if self.config.get("site_url") and self.config.get("app_name"):
                params["extra_headers"] = {
                    "HTTP-Referer": self.config["site_url"],
                    "X-Title": self.config["app_name"],
                }
        elif self.config.get("store") is not None:
            params["store"] = self.config["store"]
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        client = await self._get_client()
        response = await client.chat.completions.create(**params)
        parsed = self._parse_response(response, tools)
        if callback is not None:
            try:
                await callback(self, response, params)
            except Exception:
                logger.exception("Error in Mem0 OpenAI LLM response callback")
        return parsed

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            if client is not None:
                await _finish_cleanup(
                    client.close(),
                    error_message="Mem0 OpenAI LLM cleanup failed during cancellation",
                )


class OllamaLLM:
    """Native-async equivalent of Mem0 2.0.10's Ollama LLM."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model = self.config.get("model") or "llama3.1:70b"
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.top_p = self.config.get("top_p", 0.1)
        self._client: Any = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._initialize_lock:
            if self._client is not None:
                return self._client
            if self._closed:
                raise RuntimeError("Cannot use a closed OllamaLLM")
            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise ImportError(
                    "The 'ollama' library is required. Install the mem0 extra."
                ) from exc
            self._client = AsyncClient(host=self.config.get("ollama_base_url"))
            return self._client

    @staticmethod
    def _parse_response(response: Any, tools: list[dict[str, Any]] | None) -> Any:
        message = _response_value(response, "message", {})
        content = _response_value(message, "content")
        if not tools:
            return content
        parsed = {"content": content, "tool_calls": []}
        for tool_call in _response_value(message, "tool_calls", []) or []:
            function = _response_value(tool_call, "function", {})
            arguments = _response_value(function, "arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(_extract_json(arguments))
            parsed["tool_calls"].append(
                {
                    "name": _response_value(function, "name", ""),
                    "arguments": arguments,
                }
            )
        return parsed

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        response_format: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        request_messages = messages
        params: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
        }
        if response_format and response_format.get("type") == "json_object":
            params["format"] = "json"
            request_messages = [dict(message) for message in messages]
            if request_messages and request_messages[-1]["role"] == "user":
                request_messages[-1]["content"] += (
                    "\n\nPlease respond with valid JSON only."
                )
            else:
                request_messages.append(
                    {
                        "role": "user",
                        "content": "Please respond with valid JSON only.",
                    }
                )
            params["messages"] = request_messages
        params["options"] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "top_p": self.top_p,
        }
        if tools:
            params["tools"] = tools

        client = await self._get_client()
        response = await client.chat(**params)
        return self._parse_response(response, tools)

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            if client is not None:
                await _finish_cleanup(
                    client.close(),
                    error_message="Mem0 Ollama LLM cleanup failed during cancellation",
                )


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

    async def _initialize(self) -> None:
        async with self._lock:
            await self._connection_locked()

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
                await _finish_cleanup(
                    connection.close(),
                    error_message="Mem0 SQLite cleanup failed during cancellation",
                )
