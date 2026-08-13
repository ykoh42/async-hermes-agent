"""RetainDB memory plugin — MemoryProvider interface.

Cross-session memory via RetainDB cloud API.

Features:
- Correct API routes for all operations
- Durable SQLite write-behind queue (crash-safe, async ingest)
- Semantic search + user profile retrieval
- Context query with deduplication overlay
- Dialectic synthesis (LLM-powered user understanding, prefetched each turn)
- Agent self-model (persona + instructions from SOUL.md, prefetched each turn)
- Shared file store tools (upload, list, read, ingest, delete)
- Explicit memory tools (profile, search, context, remember, forget)

Config (env vars or hermes config.yaml under retaindb:):
  RETAINDB_API_KEY     — API key (required)
  RETAINDB_BASE_URL    — API endpoint (default: https://api.retaindb.com)
  RETAINDB_PROJECT     — Project identifier (optional — defaults to "default")
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiofiles
import aiofiles.os
import aiosqlite

from agent.file_safety import raise_if_read_blocked
from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from agent.ssl_verify import _create_httpx_client
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.retaindb.com"
_ASYNC_SHUTDOWN = object()


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish an owned task before propagating caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def _load_retaindb_config() -> dict[str, Any]:
    """Return the ``memory.retaindb`` block from config.yaml (empty on error)."""
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        provider_config = (
            memory_config.get("retaindb", {}) if isinstance(memory_config, dict) else {}
        )
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except asyncio.CancelledError:
        raise
    except Exception:
        return {}


def _config_str(value: Any) -> str:
    """Return a stripped string for a config value, else ``""``."""
    return value.strip() if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "retaindb_profile",
    "description": "Get the user's stable profile — preferences, facts, and patterns recalled from long-term memory.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "retaindb_search",
    "description": "Semantic search across stored memories. Returns ranked results with relevance scores.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {
                "type": "integer",
                "description": "Max results (default: 8, max: 20).",
            },
        },
        "required": ["query"],
    },
}

CONTEXT_SCHEMA = {
    "name": "retaindb_context",
    "description": "Synthesized context block — what matters most for the current task, pulled from long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Current task or question."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "retaindb_remember",
    "description": "Persist an explicit fact, preference, or decision to long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "memory_type": {
                "type": "string",
                "enum": [
                    "factual",
                    "preference",
                    "goal",
                    "instruction",
                    "event",
                    "opinion",
                ],
                "description": "Category (default: factual).",
            },
            "importance": {
                "type": "number",
                "description": "Importance 0-1 (default: 0.7).",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "retaindb_forget",
    "description": "Delete a specific memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}

FILE_UPLOAD_SCHEMA = {
    "name": "retaindb_upload_file",
    "description": "Upload a file to the shared RetainDB file store. Returns an rdb:// URI any agent can reference.",
    "parameters": {
        "type": "object",
        "properties": {
            "local_path": {
                "type": "string",
                "description": "Local file path to upload.",
            },
            "remote_path": {
                "type": "string",
                "description": "Destination path, e.g. /reports/q1.pdf",
            },
            "scope": {
                "type": "string",
                "enum": ["USER", "PROJECT", "ORG"],
                "description": "Access scope (default: PROJECT).",
            },
            "ingest": {
                "type": "boolean",
                "description": "Also extract memories from file after upload (default: false).",
            },
        },
        "required": ["local_path"],
    },
}

FILE_LIST_SCHEMA = {
    "name": "retaindb_list_files",
    "description": "List files in the shared file store.",
    "parameters": {
        "type": "object",
        "properties": {
            "prefix": {
                "type": "string",
                "description": "Path prefix to filter by, e.g. /reports/",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 50).",
            },
        },
        "required": [],
    },
}

FILE_READ_SCHEMA = {
    "name": "retaindb_read_file",
    "description": "Read the text content of a stored file by its file ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "File ID returned from upload or list.",
            },
        },
        "required": ["file_id"],
    },
}

FILE_INGEST_SCHEMA = {
    "name": "retaindb_ingest_file",
    "description": "Chunk, embed, and extract memories from a stored file. Makes its contents searchable.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "File ID to ingest."},
        },
        "required": ["file_id"],
    },
}

FILE_DELETE_SCHEMA = {
    "name": "retaindb_delete_file",
    "description": "Delete a stored file.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "File ID to delete."},
        },
        "required": ["file_id"],
    },
}


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class _Client:
    def __init__(self, api_key: str, base_url: str, project: str):
        self.api_key = api_key
        self.base_url = re.sub(r"/+$", "", base_url)
        self.project = project
        self._http_client = None
        self._client_lock = asyncio.Lock()
        self._closed = False

    def _headers(self, path: str) -> dict:
        token = self.api_key.replace("Bearer ", "").strip()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-sdk-runtime": "hermes-plugin",
        }
        if path.startswith(("/v1/memory", "/v1/context")):
            headers["X-API-Key"] = token
        return headers

    async def _get_http_client(self):
        if self._closed:
            raise RuntimeError("RetainDB client is closed")
        if self._http_client is not None:
            return self._http_client
        async with self._client_lock:
            if self._closed:
                raise RuntimeError("RetainDB client is closed")
            if self._http_client is None:
                self._http_client = await _create_httpx_client(follow_redirects=True)
        return self._http_client

    async def request(
        self,
        method: str,
        path: str,
        *,
        params=None,
        json_body=None,
        timeout: float = 8.0,  # noqa: ASYNC109
    ) -> Any:
        client = await self._get_http_client()
        method_upper = method.upper()
        response = await client.request(
            method_upper,
            f"{self.base_url}{path}",
            params=params,
            json=json_body if method_upper not in {"GET", "DELETE"} else None,
            headers=self._headers(path),
            timeout=timeout,
        )
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        if response.status_code >= 400:
            message = ""
            if isinstance(payload, dict):
                message = str(payload.get("message") or payload.get("error") or "")
            raise RuntimeError(
                f"RetainDB {method} {path} failed "
                f"({response.status_code}): {message or payload}"
            )
        return payload

    async def query_context(
        self,
        user_id: str,
        session_id: str,
        query: str,
        max_tokens: int = 1200,
    ) -> dict:
        return await self.request(
            "POST",
            "/v1/context/query",
            json_body={
                "project": self.project,
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "include_memories": True,
                "max_tokens": max_tokens,
            },
        )

    async def search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int = 8,
    ) -> dict:
        return await self.request(
            "POST",
            "/v1/memory/search",
            json_body={
                "project": self.project,
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "top_k": top_k,
                "include_pending": True,
            },
        )

    async def get_profile(self, user_id: str) -> dict:
        try:
            return await self.request(
                "GET",
                f"/v1/memory/profile/{quote(user_id, safe='')}",
                params={"project": self.project, "include_pending": "true"},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return await self.request(
                "GET",
                "/v1/memories",
                params={
                    "project": self.project,
                    "user_id": user_id,
                    "limit": "200",
                },
            )

    async def add_memory(
        self,
        user_id: str,
        session_id: str,
        content: str,
        memory_type: str = "factual",
        importance: float = 0.7,
    ) -> dict:
        body = {
            "project": self.project,
            "content": content,
            "memory_type": memory_type,
            "user_id": user_id,
            "session_id": session_id,
            "importance": importance,
        }
        try:
            return await self.request(
                "POST",
                "/v1/memory",
                json_body={**body, "write_mode": "sync"},
                timeout=5.0,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return await self.request(
                "POST",
                "/v1/memories",
                json_body=body,
                timeout=5.0,
            )

    async def delete_memory(self, memory_id: str) -> dict:
        try:
            return await self.request(
                "DELETE",
                f"/v1/memory/{quote(memory_id, safe='')}",
                timeout=5.0,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return await self.request(
                "DELETE",
                f"/v1/memories/{quote(memory_id, safe='')}",
                timeout=5.0,
            )

    async def ingest_session(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        timeout: float = 15.0,  # noqa: ASYNC109
    ) -> dict:
        return await self.request(
            "POST",
            "/v1/memory/ingest/session",
            json_body={
                "project": self.project,
                "session_id": session_id,
                "user_id": user_id,
                "messages": messages,
                "write_mode": "sync",
            },
            timeout=timeout,
        )

    async def ask_user(
        self,
        user_id: str,
        query: str,
        reasoning_level: str = "low",
    ) -> dict:
        return await self.request(
            "POST",
            f"/v1/memory/profile/{quote(user_id, safe='')}/ask",
            json_body={
                "project": self.project,
                "query": query,
                "reasoning_level": reasoning_level,
            },
            timeout=8.0,
        )

    async def get_agent_model(self, agent_id: str) -> dict:
        return await self.request(
            "GET",
            f"/v1/memory/agent/{quote(agent_id, safe='')}/model",
            params={"project": self.project},
            timeout=4.0,
        )

    async def seed_agent_identity(
        self,
        agent_id: str,
        content: str,
        source: str = "soul_md",
    ) -> dict:
        return await self.request(
            "POST",
            f"/v1/memory/agent/{quote(agent_id, safe='')}/seed",
            json_body={
                "project": self.project,
                "content": content,
                "source": source,
            },
            timeout=20.0,
        )

    async def upload_file(
        self,
        data: bytes,
        filename: str,
        remote_path: str,
        mime_type: str,
        scope: str,
        project_id: str | None,
    ) -> dict:
        client = await self._get_http_client()
        token = self.api_key.replace("Bearer ", "").strip()
        headers = {
            "Authorization": f"Bearer {token}",
            "x-sdk-runtime": "hermes-plugin",
        }
        fields = {"path": remote_path, "scope": scope.upper()}
        if project_id:
            fields["project_id"] = project_id
        response = await client.post(
            f"{self.base_url}/v1/files",
            files={"file": (filename, data, mime_type)},
            data=fields,
            headers=headers,
            timeout=30,
        )
        if response.status_code >= 400:
            response.raise_for_status()
        return response.json()

    async def list_files(
        self,
        prefix: str | None = None,
        limit: int = 50,
    ) -> dict:
        params: dict = {"limit": limit}
        if prefix:
            params["prefix"] = prefix
        return await self.request("GET", "/v1/files", params=params)

    async def get_file(self, file_id: str) -> dict:
        return await self.request(
            "GET",
            f"/v1/files/{quote(file_id, safe='')}",
        )

    async def read_file_content(self, file_id: str) -> bytes:
        client = await self._get_http_client()
        token = self.api_key.replace("Bearer ", "").strip()
        response = await client.get(
            f"{self.base_url}/v1/files/{quote(file_id, safe='')}/content",
            headers={
                "Authorization": f"Bearer {token}",
                "x-sdk-runtime": "hermes-plugin",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            response.raise_for_status()
        return response.content

    async def ingest_file(
        self,
        file_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        body: dict = {}
        if user_id:
            body["user_id"] = user_id
        if agent_id:
            body["agent_id"] = agent_id
        return await self.request(
            "POST",
            f"/v1/files/{quote(file_id, safe='')}/ingest",
            json_body=body,
            timeout=60.0,
        )

    async def delete_file(self, file_id: str) -> dict:
        return await self.request(
            "DELETE",
            f"/v1/files/{quote(file_id, safe='')}",
            timeout=5.0,
        )

    async def close(self) -> None:
        async with self._client_lock:
            self._closed = True
            client = self._http_client
            self._http_client = None
        if client is not None:
            await _finish_owned_task(
                asyncio.create_task(
                    client.aclose(),
                    name="retaindb-http-client-close",
                )
            )


# ---------------------------------------------------------------------------
# Durable write-behind queue
# ---------------------------------------------------------------------------


class _WriteQueue:
    """SQLite-backed async write queue. Pending rows replay on startup."""

    def __init__(self, client: _Client, db_path: Path):
        self._client = client
        self._db_path = db_path
        self._q: asyncio.Queue = asyncio.Queue()
        self._conn: aiosqlite.Connection | None = None
        self._db_lock = asyncio.Lock()
        self._writer_task: asyncio.Task[None] | None = None
        self._shutting_down = asyncio.Event()
        self._shutdown_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        await aiofiles.os.makedirs(self._db_path.parent, exist_ok=True)
        conn = await aiosqlite.connect(str(self._db_path), timeout=30)
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        try:
            await conn.execute(
                """CREATE TABLE IF NOT EXISTS pending (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT, session_id TEXT, messages_json TEXT,
                    created_at TEXT, last_error TEXT
                )"""
            )
            await conn.commit()
            rows = await self._pending_rows()
            recovered = [
                (row_id, user_id, session_id, json.loads(messages_json))
                for row_id, user_id, session_id, messages_json in rows
            ]
        except BaseException:
            self._conn = None
            await conn.close()
            raise
        self._writer_task = asyncio.create_task(
            self._loop(),
            name="retaindb-writer",
        )
        for item in recovered:
            self._q.put_nowait(item)

    async def _pending_rows(self) -> list:
        if self._conn is None:
            return []
        async with self._db_lock:
            cursor = await self._conn.execute(
                "SELECT id, user_id, session_id, messages_json "
                "FROM pending ORDER BY id ASC LIMIT 200"
            )
            try:
                return await cursor.fetchall()
            finally:
                await cursor.close()

    async def enqueue(
        self,
        user_id: str,
        session_id: str,
        messages: list,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        async with self._db_lock:
            if self._shutting_down.is_set():
                raise RuntimeError("RetainDB write queue is shutting down")
            conn = self._conn
            if conn is None:
                raise RuntimeError("RetainDB write queue is not initialized")
            cursor = await conn.execute(
                "INSERT INTO pending "
                "(user_id, session_id, messages_json, created_at) "
                "VALUES (?,?,?,?)",
                (
                    user_id,
                    session_id,
                    json.dumps(messages, ensure_ascii=False),
                    now,
                ),
            )
            row_id = cursor.lastrowid
            await cursor.close()
            await conn.commit()
            self._q.put_nowait((row_id, user_id, session_id, messages))

    async def _flush_row(
        self,
        row_id: int,
        user_id: str,
        session_id: str,
        messages: list,
    ) -> None:
        if self._conn is None:
            return
        try:
            await self._client.ingest_session(user_id, session_id, messages)
            async with self._db_lock:
                await self._conn.execute(
                    "DELETE FROM pending WHERE id = ?",
                    (row_id,),
                )
                await self._conn.commit()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("RetainDB ingest failed (will retry): %s", exc)
            async with self._db_lock:
                await self._conn.execute(
                    "UPDATE pending SET last_error = ? WHERE id = ?",
                    (str(exc), row_id),
                )
                await self._conn.commit()
            await asyncio.sleep(2)

    async def _loop(self) -> None:
        while True:
            item = await self._q.get()
            try:
                if item is _ASYNC_SHUTDOWN:
                    return
                try:
                    await self._flush_row(*item)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("RetainDB writer error: %s", exc)
            finally:
                self._q.task_done()

    async def _shutdown(self) -> None:
        writer = self._writer_task
        if writer is not None and not writer.done():
            async with self._db_lock:
                self._q.put_nowait(_ASYNC_SHUTDOWN)
            try:
                await asyncio.wait_for(asyncio.shield(writer), timeout=10.0)
            except TimeoutError:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
        elif writer is not None:
            await asyncio.gather(writer, return_exceptions=True)
        self._writer_task = None

        while True:
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._q.task_done()

        async with self._db_lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                await conn.close()

    async def shutdown(self) -> None:
        self._shutting_down.set()
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._shutdown(),
                name="retaindb-queue-shutdown",
            )
        await _finish_owned_task(self._shutdown_task)


# ---------------------------------------------------------------------------
# Overlay formatter
# ---------------------------------------------------------------------------


def _build_overlay(
    profile: dict,
    query_result: dict,
    local_entries: list[str] | None = None,
) -> str:
    def _compact(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()[:320]

    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", _compact(value).lower())

    seen: list[str] = [_norm(entry) for entry in (local_entries or []) if _norm(entry)]
    profile_items: list[str] = []
    for memory in list((profile or {}).get("memories") or [])[:5]:
        content = _compact((memory or {}).get("content") or "")
        normalized = _norm(content)
        if content and normalized not in seen:
            seen.append(normalized)
            profile_items.append(content)

    query_items: list[str] = []
    for result in list((query_result or {}).get("results") or [])[:5]:
        content = _compact((result or {}).get("content") or "")
        normalized = _norm(content)
        if content and normalized not in seen:
            seen.append(normalized)
            query_items.append(content)

    if not profile_items and not query_items:
        return ""

    lines = ["[RetainDB Context]", "Profile:"]
    lines += [f"- {item}" for item in profile_items] or ["- None"]
    lines.append("Relevant memories:")
    lines += [f"- {item}" for item in query_items] or ["- None"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main plugin class
# ---------------------------------------------------------------------------


class RetainDBMemoryProvider(MemoryProvider):
    """RetainDB cloud memory with durable queue and shared files."""

    def __init__(self):
        self._client: _Client | None = None
        self._queue: _WriteQueue | None = None
        self._user_id = "default"
        self._session_id = ""
        self._agent_id = "hermes"
        self._lock = asyncio.Lock()

        self._context_result = ""
        self._dialectic_result = ""
        self._agent_model: dict = {}

        self._prefetch_tasks: list[asyncio.Task[None]] = []
        self._owned_tasks: set[asyncio.Task[None]] = set()
        self._shutting_down = asyncio.Event()
        self._shutdown_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return "retaindb"

    async def is_available(self) -> bool:
        return bool(get_secret("RETAINDB_API_KEY"))

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "RetainDB API key",
                "secret": True,
                "required": True,
                "env_var": "RETAINDB_API_KEY",
                "url": "https://retaindb.com",
            },
            {
                "key": "base_url",
                "description": "API endpoint",
                "default": _DEFAULT_BASE_URL,
            },
            {
                "key": "project",
                "description": "Project identifier (optional — uses 'default' project if not set)",
                "default": "",
            },
        ]

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._owned_tasks.add(task)
        task.add_done_callback(self._owned_tasks.discard)

    async def initialize(self, session_id: str, **kwargs) -> None:
        provider_config = await _load_retaindb_config()
        api_key = get_secret("RETAINDB_API_KEY", "") or ""
        base_url_raw = (
            get_secret("RETAINDB_BASE_URL")
            or _config_str(provider_config.get("base_url"))
            or _DEFAULT_BASE_URL
        )
        base_url = re.sub(r"/+$", "", base_url_raw)

        explicit = get_secret("RETAINDB_PROJECT") or _config_str(
            provider_config.get("project")
        )
        if explicit:
            project = explicit
        else:
            hermes_home = str(kwargs.get("hermes_home", ""))
            profile_name = os.path.basename(hermes_home) if hermes_home else ""
            project = (
                f"hermes-{profile_name}"
                if profile_name and profile_name not in {"", ".hermes"}
                else "default"
            )

        from hermes_constants import get_hermes_home

        hermes_home_path = get_hermes_home()
        soul_content = ""
        soul_path = hermes_home_path / "SOUL.md"
        if await aiofiles.os.path.exists(soul_path):
            async with aiofiles.open(
                soul_path,
                encoding="utf-8",
                errors="replace",
            ) as handle:
                soul_content = (await handle.read()).strip()

        client = _Client(api_key, base_url, project)
        self._session_id = session_id
        self._user_id = kwargs.get("user_id", "default") or "default"
        self._agent_id = kwargs.get("agent_id", "hermes") or "hermes"

        queue = _WriteQueue(client, hermes_home_path / "retaindb_queue.db")
        try:
            await queue.initialize()
        except BaseException:
            await _finish_owned_task(
                asyncio.create_task(
                    client.close(),
                    name="retaindb-initialize-cleanup",
                )
            )
            raise
        self._client = client
        self._queue = queue

        if soul_content:
            task = asyncio.create_task(
                self._seed_soul(soul_content),
                name="retaindb-soul-seed",
            )
            self._track_task(task)

    async def _seed_soul(self, content: str) -> None:
        try:
            if self._client is not None:
                await self._client.seed_agent_identity(
                    self._agent_id,
                    content,
                    source="soul_md",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("RetainDB soul seed failed: %s", exc)

    def system_prompt_block(self) -> str:
        project = self._client.project if self._client else "retaindb"
        return (
            "# RetainDB Memory\n"
            f"Active. Project: {project}.\n"
            "Use retaindb_search to find memories, retaindb_remember to store facts, "
            "retaindb_profile for a user overview, retaindb_context for current-task context."
        )

    async def queue_prefetch(
        self,
        query: str,
        *,
        session_id: str = "",
    ) -> None:
        if not self._client or self._shutting_down.is_set():
            return
        for task in tuple(self._prefetch_tasks):
            await asyncio.wait((task,), timeout=2.0)
            if task.done():
                await asyncio.gather(task, return_exceptions=True)
        if not self._client or self._shutting_down.is_set():
            return

        tasks = [
            asyncio.create_task(
                self._prefetch_context(query),
                name="retaindb-ctx",
            ),
            asyncio.create_task(
                self._prefetch_dialectic(query),
                name="retaindb-dialectic",
            ),
            asyncio.create_task(
                self._prefetch_agent_model(),
                name="retaindb-agent-model",
            ),
        ]
        self._prefetch_tasks = tasks
        for task in tasks:
            self._track_task(task)

    async def _prefetch_context(self, query: str) -> None:
        try:
            if self._client is None:
                return
            query_result = await self._client.query_context(
                self._user_id,
                self._session_id,
                query,
            )
            profile = await self._client.get_profile(self._user_id)
            overlay = _build_overlay(profile, query_result)
            async with self._lock:
                self._context_result = overlay
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("RetainDB context prefetch failed: %s", exc)

    async def _prefetch_dialectic(self, query: str) -> None:
        try:
            if self._client is None:
                return
            result = await self._client.ask_user(
                self._user_id,
                query,
                reasoning_level=self._reasoning_level(query),
            )
            answer = str(result.get("answer") or "")
            if answer:
                async with self._lock:
                    self._dialectic_result = answer
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("RetainDB dialectic prefetch failed: %s", exc)

    async def _prefetch_agent_model(self) -> None:
        try:
            if self._client is None:
                return
            model = await self._client.get_agent_model(self._agent_id)
            if model.get("memory_count", 0) > 0:
                async with self._lock:
                    self._agent_model = model
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("RetainDB agent model prefetch failed: %s", exc)

    @staticmethod
    def _reasoning_level(query: str) -> str:
        length = len(query)
        if length < 120:
            return "low"
        if length < 400:
            return "medium"
        return "high"

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        async with self._lock:
            context = self._context_result
            dialectic = self._dialectic_result
            agent_model = self._agent_model
            self._context_result = ""
            self._dialectic_result = ""
            self._agent_model = {}

        parts: list[str] = []
        if context:
            parts.append(context)
        if dialectic:
            parts.append(f"[RetainDB User Synthesis]\n{dialectic}")
        if agent_model and agent_model.get("memory_count", 0) > 0:
            model_lines: list[str] = []
            if agent_model.get("persona"):
                model_lines.append(f"Persona: {agent_model['persona']}")
            if agent_model.get("persistent_instructions"):
                model_lines.append(
                    "Instructions:\n"
                    + "\n".join(
                        f"- {instruction}"
                        for instruction in agent_model["persistent_instructions"]
                    )
                )
            if agent_model.get("working_style"):
                model_lines.append(f"Working style: {agent_model['working_style']}")
            if model_lines:
                parts.append("[RetainDB Agent Self-Model]\n" + "\n".join(model_lines))

        return "\n\n".join(parts)

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if not self._queue or not user_content:
            return
        now = datetime.now(UTC).isoformat()
        await self._queue.enqueue(
            self._user_id,
            session_id or self._session_id,
            [
                {"role": "user", "content": user_content, "timestamp": now},
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "timestamp": now,
                },
            ],
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            PROFILE_SCHEMA,
            SEARCH_SCHEMA,
            CONTEXT_SCHEMA,
            REMEMBER_SCHEMA,
            FORGET_SCHEMA,
            FILE_UPLOAD_SCHEMA,
            FILE_LIST_SCHEMA,
            FILE_READ_SCHEMA,
            FILE_INGEST_SCHEMA,
            FILE_DELETE_SCHEMA,
        ]

    async def handle_tool_call(
        self,
        tool_name: str,
        args: dict,
        **kwargs,
    ) -> str:
        if not self._client:
            return tool_error("RetainDB not initialized")
        try:
            return json.dumps(await self._dispatch(tool_name, args))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return tool_error(str(exc))

    async def _dispatch(self, tool_name: str, args: dict) -> Any:
        client = self._client
        if client is None:
            return {"error": "RetainDB not initialized"}

        if tool_name == "retaindb_profile":
            return await client.get_profile(self._user_id)

        if tool_name == "retaindb_search":
            query = args.get("query", "")
            if not query:
                return {"error": "query is required"}
            return await client.search(
                self._user_id,
                self._session_id,
                query,
                top_k=min(int(args.get("top_k", 8)), 20),
            )

        if tool_name == "retaindb_context":
            query = args.get("query", "")
            if not query:
                return {"error": "query is required"}
            query_result = await client.query_context(
                self._user_id,
                self._session_id,
                query,
            )
            profile = await client.get_profile(self._user_id)
            overlay = _build_overlay(profile, query_result)
            return {"context": overlay, "raw": query_result}

        if tool_name == "retaindb_remember":
            content = args.get("content", "")
            if not content:
                return {"error": "content is required"}
            return await client.add_memory(
                self._user_id,
                self._session_id,
                content,
                memory_type=args.get("memory_type", "factual"),
                importance=float(args.get("importance", 0.7)),
            )

        if tool_name == "retaindb_forget":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return {"error": "memory_id is required"}
            return await client.delete_memory(memory_id)

        if tool_name == "retaindb_upload_file":
            local_path = args.get("local_path", "")
            if not local_path:
                return {"error": "local_path is required"}
            path_obj = Path(local_path)
            if not await aiofiles.os.path.exists(path_obj):
                return {"error": f"File not found: {local_path}"}
            try:
                await raise_if_read_blocked(str(path_obj))
            except ValueError as exc:
                return {"error": str(exc)}
            async with aiofiles.open(path_obj, "rb") as handle:
                data = await handle.read()
            mime = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
            remote_path = args.get("remote_path") or f"/{path_obj.name}"
            result = await client.upload_file(
                data,
                path_obj.name,
                remote_path,
                mime,
                args.get("scope", "PROJECT"),
                None,
            )
            if args.get("ingest") and result.get("file", {}).get("id"):
                result["ingest"] = await client.ingest_file(
                    result["file"]["id"],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                )
            return result

        if tool_name == "retaindb_list_files":
            return await client.list_files(
                prefix=args.get("prefix"),
                limit=int(args.get("limit", 50)),
            )

        if tool_name == "retaindb_read_file":
            file_id = args.get("file_id", "")
            if not file_id:
                return {"error": "file_id is required"}
            meta = await client.get_file(file_id)
            file_info = meta.get("file") or {}
            mime = (file_info.get("mime_type") or "").lower()
            raw = await client.read_file_content(file_id)
            text_extensions = (
                ".txt",
                ".md",
                ".json",
                ".csv",
                ".yaml",
                ".yml",
                ".xml",
                ".html",
            )
            if not (
                mime.startswith("text/")
                or any(
                    file_info.get("name", "").endswith(ext) for ext in text_extensions
                )
            ):
                return {
                    "file_id": file_id,
                    "rdb_uri": file_info.get("rdb_uri"),
                    "name": file_info.get("name"),
                    "content": None,
                    "note": "Binary file — use retaindb_ingest_file to extract text into memory.",
                }
            text = raw.decode("utf-8", errors="replace")
            return {
                "file_id": file_id,
                "rdb_uri": file_info.get("rdb_uri"),
                "name": file_info.get("name"),
                "content": text[:32000],
                "truncated": len(text) > 32000,
            }

        if tool_name == "retaindb_ingest_file":
            file_id = args.get("file_id", "")
            if not file_id:
                return {"error": "file_id is required"}
            return await client.ingest_file(
                file_id,
                user_id=self._user_id,
                agent_id=self._agent_id,
            )

        if tool_name == "retaindb_delete_file":
            file_id = args.get("file_id", "")
            if not file_id:
                return {"error": "file_id is required"}
            return await client.delete_file(file_id)

        return {"error": f"Unknown tool: {tool_name}"}

    async def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        if action != "add" or not content or not self._client:
            return
        try:
            memory_type = "preference" if target == "user" else "factual"
            await self._client.add_memory(
                self._user_id,
                self._session_id,
                content,
                memory_type=memory_type,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("RetainDB memory mirror failed: %s", exc)

    async def _shutdown(self) -> None:
        tasks = tuple(self._owned_tasks)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=3.0)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._owned_tasks.clear()
        self._prefetch_tasks.clear()

        queue = self._queue
        self._queue = None
        client = self._client
        self._client = None
        try:
            if queue is not None:
                await queue.shutdown()
        finally:
            if client is not None:
                await client.close()

    async def shutdown(self) -> None:
        self._shutting_down.set()
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._shutdown(),
                name="retaindb-shutdown",
            )
        await _finish_owned_task(self._shutdown_task)


def register(ctx) -> None:
    """Register RetainDB as a memory provider plugin."""
    ctx.register_memory_provider(RetainDBMemoryProvider())
