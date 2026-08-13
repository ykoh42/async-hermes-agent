"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search, and automatic deduplication
via the Mem0 Platform API (cloud) or OSS (self-hosted) via Memory.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Configuration
-------------
Secret (lives in $HERMES_HOME/.env or the environment):
  MEM0_API_KEY       — Mem0 Platform API key (required for platform mode)
  MEM0_HOST          — Base URL of a self-hosted Mem0 server. When set, the
                       plugin talks to that server directly over HTTP
                       (X-API-Key auth) instead of the cloud API.

Behavioral settings (live in $HERMES_HOME/mem0.json):
  mode               — Backend mode: "platform" (default) or "oss"
  host               — Self-hosted Mem0 server URL (alt: MEM0_HOST env var).
                       When set, routes to the self-hosted HTTP backend.
  user_id            — Canonical user identifier. When set, it is applied
                       uniformly across sessions so the same user gets one
                       merged memory store. When unset, a caller-provided
                       session identity may be used instead.
  agent_id           — Agent identifier (default: hermes)

The matching MEM0_MODE / MEM0_USER_ID / MEM0_AGENT_ID environment variables are
still read as a backward-compatible fallback, but mem0.json is the canonical
home for these non-secret settings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import secrets
import time
from typing import Any, Dict, List

import aiofiles
import aiofiles.os

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from hermes_cli.async_source_loader import _locate_source_module
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_PREFETCH_WAIT_SECS = 3

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")

# Sentinel returned when neither MEM0_USER_ID nor a caller-provided identity is
# available. Treated as "no operator-configured user_id" by initialize() so
# legacy mem0.json files written by the upstream setup wizard (which used this
# placeholder) still allow caller-provided identities to flow through.
_DEFAULT_USER_ID = "hermes-user"


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one owned Mem0 task through repeated caller cancellation."""
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


async def _collect_owned_task(task: asyncio.Task[Any]) -> None:
    """Collect a cancelled Mem0 task without propagating its terminal state."""
    async def collect() -> None:
        await asyncio.gather(task, return_exceptions=True)

    await _finish_owned_task(asyncio.create_task(collect()))


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    etype = type(exc).__name__
    if etype in _CLIENT_ERROR_TYPES:
        return True
    err_str = str(exc).lower()
    return "404" in err_str or "not found" in err_str or "valid uuid" in err_str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

async def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "mode": get_secret("MEM0_MODE", "platform"),
        "api_key": get_secret("MEM0_API_KEY", ""),
        "host": get_secret("MEM0_HOST", ""),
        "agent_id": get_secret("MEM0_AGENT_ID", "hermes"),
        "oss": {},
    }
    # Only carry user_id when the operator explicitly configured one (env or
    # mem0.json). An absent key tells initialize() to fall back to the
    # caller-provided id from kwargs instead of overriding it with a placeholder.
    env_user_id = get_secret("MEM0_USER_ID")
    if env_user_id:
        config["user_id"] = env_user_id

    config_path = get_hermes_home() / "mem0.json"
    try:
        async with aiofiles.open(config_path, encoding="utf-8") as config_file:
            file_cfg = json.loads(await config_file.read())
        if isinstance(file_cfg, dict):
            config.update(
                {k: v for k, v in file_cfg.items() if v is not None and v != ""}
            )
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: false, platform mode only)."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts you've "
        "already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed "
        "or was wrong — correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search "
        "result). Use when a stored fact is obsolete or the user asks you to "
        "forget it; prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search.

    Supports Platform API (cloud) and OSS (self-hosted) modes via MEM0_MODE.
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._backend = None
        self._mode = "platform"
        self._api_key = ""
        self._host = ""
        self._user_id = _DEFAULT_USER_ID
        self._agent_id = "hermes"
        self._rerank_default = False
        self._channel = "cli"  # gateway channel name (cli/telegram/discord/...)
        self._prefetch_task: asyncio.Task[str] | None = None
        self._prefetch_query = ""
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    async def is_available(self) -> bool:
        cfg = await _load_config()
        mode = cfg.get("mode", "platform")
        if mode == "oss":
            return bool(
                cfg.get("oss", {}).get("vector_store")
                and await _locate_source_module("mem0")
            )
        # Platform needs an api_key; self-hosted needs a host (api_key optional
        # when the server runs with AUTH_DISABLED).
        return bool(cfg.get("host") or cfg.get("api_key"))

    async def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        try:
            async with aiofiles.open(config_path, encoding="utf-8") as config_file:
                loaded = json.loads(await config_file.read())
            if isinstance(loaded, dict):
                existing = loaded
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass
        existing.update(values)
        await aiofiles.os.makedirs(config_path.parent, exist_ok=True)
        temporary = config_path.with_name(
            f".{config_path.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            async with aiofiles.open(
                temporary,
                "x",
                encoding="utf-8",
                opener=lambda path, flags: os.open(path, flags, 0o600),
            ) as config_file:
                await config_file.write(
                    json.dumps(existing, indent=2, ensure_ascii=False) + "\n"
                )
                await config_file.flush()
            await aiofiles.os.replace(temporary, config_path)
        except BaseException:
            try:
                await aiofiles.os.remove(temporary)
            except FileNotFoundError:
                pass
            raise

    def get_config_schema(self):
        cfg = self._config or {"mode": get_secret("MEM0_MODE", "platform")}
        mode = cfg.get("mode", "platform")
        api_key_required = mode != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def _create_backend(self):
        try:
            if self._mode == "oss":
                from ._backend import OSSBackend

                return OSSBackend(self._config.get("oss", {}))
            if self._host:
                from ._backend import SelfHostedBackend

                return SelfHostedBackend(self._api_key, self._host)
            from ._backend import PlatformBackend

            return PlatformBackend(self._api_key)
        except Exception as exc:
            logger.error(
                "Mem0 backend failed to initialize (%s mode): %s",
                self._mode,
                exc,
            )
            self._init_error = str(exc)
            return None

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if self._mode == "oss":
            err_str = str(exc).lower()
            if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
                vs = self._config.get("oss", {}).get("vector_store", {})
                msg += f" (check that {vs.get('provider', 'vector store')} is running)"
        return msg

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        count = self._consecutive_failures
        if count >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        else:
            count = 0
        if count >= _BREAKER_THRESHOLD:
            hint = ""
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "unknown")
                hint = f" Check that your {provider} vector store is running and reachable."
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.%s",
                count, _BREAKER_COOLDOWN_SECS, hint,
            )

    async def initialize(self, session_id: str, **kwargs) -> None:
        self._config = await _load_config()
        self._mode = self._config.get("mode", "platform")
        self._api_key = self._config.get("api_key", "")
        self._host = self._config.get("host", "")
        # Resolution order for user_id:
        #   1. Operator-configured MEM0_USER_ID (env or $HERMES_HOME/mem0.json) —
        #      the canonical principal across sessions.
        #   2. Caller-provided user_id from kwargs — preserves application-level
        #      identity isolation when no override is configured.
        #   3. Hardcoded fallback _DEFAULT_USER_ID for callers with no identity.
        # The literal _DEFAULT_USER_ID string is treated as unset so users who
        # used the upstream setup wizard's suggested default still get
        # caller-provided ids instead of being silently bucketed together.
        configured = self._config.get("user_id")
        if configured == _DEFAULT_USER_ID:
            configured = None
        self._user_id = configured or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._agent_id = self._config.get("agent_id", "hermes")
        # Persisted rerank preference from mem0.json. Used as the
        # DEFAULT for mem0_search when the model doesn't pass ``rerank``
        # explicitly; per-call args still win. Platform-only feature — other
        # backends accept-and-ignore the flag.
        _rr = self._config.get("rerank", False)
        self._rerank_default = (
            _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        )
        self._channel = kwargs.get("platform") or "cli"
        self._backend = self._create_backend()
        if self._backend is not None:
            from ._backend import OSSBackend, PlatformBackend

            backend = self._backend
            needs_initialization = (
                isinstance(backend, OSSBackend)
                or (
                    isinstance(backend, PlatformBackend)
                    and self._mode != "oss"
                    and not self._host
                )
            )
            if needs_initialization:
                try:
                    await backend._initialize()
                except asyncio.CancelledError:
                    self._backend = None
                    await _finish_owned_task(
                        asyncio.create_task(backend.close())
                    )
                    raise
                except Exception as exc:
                    self._backend = None
                    try:
                        await _finish_owned_task(
                            asyncio.create_task(backend.close())
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                    logger.error(
                        "Mem0 backend failed to initialize (%s mode): %s",
                        self._mode,
                        exc,
                    )
                    self._init_error = str(exc)

    def _read_filters(self) -> Dict[str, Any]:
        # Scoped to user_id only — by design — so recall surfaces memories
        # written from any gateway/agent under this principal. Writes attach
        # agent_id (and metadata.channel) so per-agent / per-channel views are
        # still possible at query time when needed; reads default to the wider
        # cross-agent recall.
        return {"user_id": self._user_id}

    def _write_metadata(self) -> Dict[str, Any]:
        # Tag every write with the gateway channel so the dashboard can offer
        # per-channel filtered views without coupling identity to the channel.
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        # Mirror the precedence in _create_backend (oss > host > platform) so
        # the label always names the backend that actually runs. Checking
        # ``host`` first here would mislabel an ``oss``+``host`` config as
        # self-hosted HTTP even though OSS wins the routing.
        if self._mode == "oss":
            mode_label = "OSS (self-hosted)"
        elif self._host:
            mode_label = "self-hosted (HTTP API)"
        else:
            mode_label = "platform (cloud API)"
        # Rerank is a Mem0 Platform feature only.
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        return (
            "# Mem0 Memory\n"
            f"Active. Mode: {mode_label}. User: {self._user_id}.\n"
            "You have persistent memory of this user from past conversations. "
            "You should call mem0_search before answering anything that could depend "
            "on prior context (the user's preferences, facts, history, people, "
            "projects, or earlier decisions) — do not rely on the chat window "
            "alone, and do not assume you have no memory.\n"
            "For multi-part or multi-hop questions, run several searches with "
            "different wording/angles and follow-up searches on what the first "
            "results surface; one search is rarely enough. Keep searching until "
            "you have every fact the question needs before you answer.\n"
            "Tools: mem0_search to find memories, mem0_add to store facts, "
            f"mem0_update and mem0_delete to manage by ID.{rerank_note}"
        )

    async def on_turn_start(
        self,
        turn_number: int,
        message: str,
        **kwargs,
    ) -> None:
        await self._start_prefetch(message)

    async def _fetch_prefetch(self, query: str) -> str:
        backend = self._backend
        if backend is None:
            return ""
        try:
            results = await backend.search(
                query,
                filters=self._read_filters(),
                top_k=10,
                rerank=False,
            )
            lines = [
                result.get("memory", "")
                for result in (results or [])
                if result.get("memory")
            ]
            self._record_success()
            if lines:
                return "## Mem0 Memory\n" + "\n".join(f"- {line}" for line in lines)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_failure()
            logger.debug("Mem0 prefetch failed: %s", exc)
        return ""

    async def _start_prefetch(self, query: str) -> None:
        if not query or self._backend is None or self._is_breaker_open():
            return
        if self._prefetch_query == query and self._prefetch_task is not None:
            return
        if self._prefetch_task is not None and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            await _collect_owned_task(self._prefetch_task)
        self._prefetch_query = query
        self._prefetch_task = asyncio.create_task(
            self._fetch_prefetch(query),
            name="mem0-prefetch",
        )

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        await self._start_prefetch(query)
        task = self._prefetch_task if self._prefetch_query == query else None
        if task is not None:
            try:
                async with asyncio.timeout(_PREFETCH_WAIT_SECS):
                    result = await asyncio.shield(task)
                if self._prefetch_task is task:
                    self._prefetch_task = None
                return result
            except TimeoutError:
                pass
        # Slow backend: skip injection; mem0_search tool remains the backstop.
        return ""

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return
        try:
            turn_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            await self._backend.add(
                turn_messages,
                user_id=self._user_id,
                agent_id=self._agent_id,
                infer=True,
                metadata=self._write_metadata(),
            )
            self._record_success()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_failure()
            logger.warning("Mem0 sync failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    async def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            hint = ""
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "vector store")
                hint = f" Check that {provider} is running and reachable."
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{hint}"})

        if self._is_breaker_open():
            msg = "Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically."
            if self._mode == "oss":
                vs = self._config.get("oss", {}).get("vector_store", {})
                msg += f" Check that your {vs.get('provider', 'vector store')} is running."
            return json.dumps({"error": msg})

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                top_k = max(1, min(int(args.get("top_k", 10)), 50))
                rerank_raw = args.get("rerank", getattr(self, "_rerank_default", False))
                if isinstance(rerank_raw, str):
                    rerank = rerank_raw.lower() not in ("false", "0", "no")
                else:
                    rerank = bool(rerank_raw)
                results = await self._backend.search(
                    query,
                    filters=self._read_filters(),
                    top_k=top_k,
                    rerank=rerank,
                )
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"id": r.get("id"), "memory": r.get("memory", ""),
                          "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                if not _is_client_error(e):
                    self._record_failure()
                return tool_error(self._format_error("Search failed", e))

        elif tool_name == "mem0_add":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            try:
                result = await self._backend.add(
                    [{"role": "user", "content": content}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                    metadata=self._write_metadata(),
                )
                self._record_success()
                event_id = result.get("event_id") if isinstance(result, dict) else None
                # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
                msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
                return json.dumps({"result": msg, "event_id": event_id})
            except Exception as e:
                self._record_failure()
                return tool_error(self._format_error("Failed to store", e))

        elif tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            text = args.get("text", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            if not text:
                return tool_error("Missing required parameter: text")
            try:
                result = await self._backend.update(memory_id, text)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return tool_error(self._format_error("Update failed", e))

        elif tool_name == "mem0_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                result = await self._backend.delete(memory_id)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return tool_error(self._format_error("Delete failed", e))

        return tool_error(f"Unknown tool: {tool_name}")

    async def shutdown(self) -> None:
        await _finish_owned_task(asyncio.create_task(self._shutdown_owned()))

    async def _shutdown_owned(self) -> None:
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()
            await _collect_owned_task(self._prefetch_task)
            self._prefetch_task = None
        if self._backend is not None:
            backend = self._backend
            self._backend = None
            try:
                await backend.close()
            except Exception:
                pass


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
