"""Supermemory memory plugin using the MemoryProvider interface.

Provides semantic long-term memory with profile recall, semantic search,
explicit memory tools, cleaned turn capture, and session-end conversation ingest.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiofiles.os
import aiofiles.tempfile

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from agent.ssl_verify import _create_httpx_client
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_TAG = "hermes"
_DEFAULT_MAX_RECALL_RESULTS = 10
_DEFAULT_PROFILE_FREQUENCY = 50
_DEFAULT_CAPTURE_MODE = "all"
_DEFAULT_SEARCH_MODE = "hybrid"
_VALID_SEARCH_MODES = ("hybrid", "memories", "documents")
_DEFAULT_API_TIMEOUT = 5.0
_MIN_CAPTURE_LENGTH = 10
_MAX_ENTITY_CONTEXT_LENGTH = 1500
_DEFAULT_BASE_URL = "https://api.supermemory.ai"
_API_KEY_URL = "http://app.supermemory.ai/integrations?connect=hermes"
_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$",
    re.IGNORECASE,
)
_CONTEXT_STRIP_RE = re.compile(
    r"<supermemory-context>[\s\S]*?</supermemory-context>\s*", re.DOTALL
)
_CONTAINERS_STRIP_RE = re.compile(
    r"<supermemory-containers>[\s\S]*?</supermemory-containers>\s*", re.DOTALL
)
_DEFAULT_ENTITY_CONTEXT = (
    "User-assistant conversation. Format: [role: user]...[user:end] and "
    "[role: assistant]...[assistant:end].\n\n"
    "Only extract things useful in future conversations. Most messages are not worth remembering.\n\n"
    "Remember lasting personal facts, preferences, routines, tools, ongoing projects, working context, "
    "and explicit requests to remember something.\n\n"
    "Do not remember temporary intents, one-time tasks, assistant actions, implementation details, or in-progress status.\n\n"
    "When in doubt, store less."
)


def _default_config() -> dict:
    return {
        "container_tag": _DEFAULT_CONTAINER_TAG,
        "auto_recall": True,
        "auto_capture": True,
        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,
        "profile_frequency": _DEFAULT_PROFILE_FREQUENCY,
        "capture_mode": _DEFAULT_CAPTURE_MODE,
        "search_mode": _DEFAULT_SEARCH_MODE,
        "entity_context": _DEFAULT_ENTITY_CONTEXT,
        "api_timeout": _DEFAULT_API_TIMEOUT,
        "base_url": "",
        "enable_custom_container_tags": False,
        "custom_containers": [],
        "custom_container_instructions": "",
    }


def _sanitize_tag(raw: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_]", "_", raw or "")
    tag = re.sub(r"_+", "_", tag)
    return tag.strip("_") or _DEFAULT_CONTAINER_TAG


def _resolve_base_url(config_value: Any = "") -> str:
    """Resolve the API base URL: config > SUPERMEMORY_BASE_URL env var > default.

    Supports self-hosted Supermemory servers (e.g. http://localhost:6767).
    """
    raw = (
        str(config_value or "").strip()
        or os.environ.get("SUPERMEMORY_BASE_URL", "").strip()
    )
    return (raw or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL


def _clamp_entity_context(text: str) -> str:
    if not text:
        return _DEFAULT_ENTITY_CONTEXT
    text = text.strip()
    return text[:_MAX_ENTITY_CONTEXT_LENGTH]


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


async def _load_supermemory_config(hermes_home: str) -> dict:
    config = _default_config()
    config_path = Path(hermes_home) / "supermemory.json"
    if await aiofiles.os.path.exists(config_path):
        try:
            async with aiofiles.open(config_path, encoding="utf-8") as handle:
                raw = json.loads(await handle.read())
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    # Keep raw container_tag — template variables like {identity} are resolved
    # in initialize(), and _sanitize_tag runs AFTER resolution.
    raw_tag = str(config.get("container_tag", _DEFAULT_CONTAINER_TAG)).strip()
    config["container_tag"] = raw_tag if raw_tag else _DEFAULT_CONTAINER_TAG
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)
    try:
        config["max_recall_results"] = max(1, min(20, int(config.get("max_recall_results", _DEFAULT_MAX_RECALL_RESULTS))))
    except Exception:
        config["max_recall_results"] = _DEFAULT_MAX_RECALL_RESULTS
    try:
        config["profile_frequency"] = max(1, min(500, int(config.get("profile_frequency", _DEFAULT_PROFILE_FREQUENCY))))
    except Exception:
        config["profile_frequency"] = _DEFAULT_PROFILE_FREQUENCY
    config["capture_mode"] = "everything" if config.get("capture_mode") == "everything" else "all"
    raw_search_mode = str(config.get("search_mode", _DEFAULT_SEARCH_MODE)).strip().lower()
    config["search_mode"] = raw_search_mode if raw_search_mode in _VALID_SEARCH_MODES else _DEFAULT_SEARCH_MODE
    config["entity_context"] = _clamp_entity_context(str(config.get("entity_context", _DEFAULT_ENTITY_CONTEXT)))
    try:
        config["api_timeout"] = max(0.5, min(15.0, float(config.get("api_timeout", _DEFAULT_API_TIMEOUT))))
    except Exception:
        config["api_timeout"] = _DEFAULT_API_TIMEOUT
    config["base_url"] = str(config.get("base_url", "") or "").strip()

    # Multi-container support
    config["enable_custom_container_tags"] = _as_bool(config.get("enable_custom_container_tags"), False)
    raw_containers = config.get("custom_containers", [])
    if isinstance(raw_containers, list):
        config["custom_containers"] = [_sanitize_tag(str(t)) for t in raw_containers if t]
    else:
        config["custom_containers"] = []
    config["custom_container_instructions"] = str(config.get("custom_container_instructions", "")).strip()

    return config


async def _save_supermemory_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "supermemory.json"
    existing = {}
    if await aiofiles.os.path.exists(config_path):
        try:
            async with aiofiles.open(config_path, encoding="utf-8") as handle:
                raw = json.loads(await handle.read())
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    await aiofiles.os.makedirs(config_path.parent, exist_ok=True)
    target_path = str(config_path)
    if await aiofiles.os.path.islink(target_path):
        target_path = await aiofiles.os.wrap(os.path.realpath)(target_path)
    original_owner: tuple[int, int] | None = None
    if os.name == "posix":
        try:
            target_stat = await aiofiles.os.stat(target_path)
            original_owner = (target_stat.st_uid, target_stat.st_gid)
        except OSError:
            pass
    temporary = ""
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            if hasattr(os, "fchmod"):
                await aiofiles.os.wrap(os.fchmod)(handle.fileno(), 0o600)
            await handle.write(
                json.dumps(
                    existing,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            await handle.flush()
            await aiofiles.os.wrap(os.fsync)(handle.fileno())
        try:
            await aiofiles.os.replace(temporary, target_path)
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EBUSY):
                raise
            async with (
                aiofiles.open(temporary, "rb") as source,
                aiofiles.open(target_path, "wb") as destination,
            ):
                while chunk := await source.read(1024 * 1024):
                    await destination.write(chunk)
                await destination.flush()
                await aiofiles.os.wrap(os.fsync)(destination.fileno())
            await aiofiles.os.remove(temporary)
        if original_owner is not None and hasattr(os, "chown"):
            try:
                await aiofiles.os.wrap(os.chown)(
                    target_path,
                    original_owner[0],
                    original_owner[1],
                )
            except OSError:
                pass
        try:
            await aiofiles.os.wrap(os.chmod)(target_path, 0o600)
        except OSError:
            pass
        temporary = ""
    finally:
        if temporary and await aiofiles.os.path.exists(temporary):
            await aiofiles.os.remove(temporary)


def _detect_category(text: str) -> str:
    lowered = text.lower()
    if re.search(r"prefer|like|love|hate|want", lowered):
        return "preference"
    if re.search(r"decided|will use|going with", lowered):
        return "decision"
    if re.search(r"\bis\b|\bare\b|\bhas\b|\bhave\b", lowered):
        return "fact"
    return "other"


def _format_relative_time(iso_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        seconds = (now - dt).total_seconds()
        if seconds < 1800:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        if seconds < 604800:
            return f"{int(seconds / 86400)}d ago"
        if dt.year == now.year:
            return dt.strftime("%d %b")
        return dt.strftime("%d %b %Y")
    except Exception:
        return ""


def _deduplicate_recall(static_facts: list, dynamic_facts: list, search_results: list) -> tuple[list, list, list]:
    seen = set()
    out_static, out_dynamic, out_search = [], [], []
    for fact in static_facts or []:
        if fact and fact not in seen:
            seen.add(fact)
            out_static.append(fact)
    for fact in dynamic_facts or []:
        if fact and fact not in seen:
            seen.add(fact)
            out_dynamic.append(fact)
    for item in search_results or []:
        memory = item.get("memory", "")
        if memory and memory not in seen:
            seen.add(memory)
            out_search.append(item)
    return out_static, out_dynamic, out_search


def _format_prefetch_context(static_facts: list, dynamic_facts: list, search_results: list, max_results: int) -> str:
    statics, dynamics, search = _deduplicate_recall(static_facts, dynamic_facts, search_results)
    statics = statics[:max_results]
    dynamics = dynamics[:max_results]
    search = search[:max_results]
    if not statics and not dynamics and not search:
        return ""

    sections = []
    if statics:
        sections.append("## User Profile (Persistent)\n" + "\n".join(f"- {item}" for item in statics))
    if dynamics:
        sections.append("## Recent Context\n" + "\n".join(f"- {item}" for item in dynamics))
    if search:
        lines = []
        for item in search:
            memory = item.get("memory", "")
            if not memory:
                continue
            similarity = item.get("similarity")
            updated = item.get("updated_at") or item.get("updatedAt") or ""
            prefix_bits = []
            rel = _format_relative_time(updated)
            if rel:
                prefix_bits.append(f"[{rel}]")
            if similarity is not None:
                try:
                    prefix_bits.append(f"[{round(float(similarity) * 100)}%]")
                except Exception:
                    pass
            prefix = " ".join(prefix_bits)
            lines.append(f"- {prefix} {memory}".strip())
        if lines:
            sections.append("## Relevant Memories\n" + "\n".join(lines))
    if not sections:
        return ""

    intro = (
        "The following is background context from long-term memory. Use it silently when relevant. "
        "Do not force memories into the conversation."
    )
    body = "\n\n".join(sections)
    return f"<supermemory-context>\n{intro}\n\n{body}\n</supermemory-context>"


def _clean_text_for_capture(text: str) -> str:
    text = _CONTEXT_STRIP_RE.sub("", text or "")
    text = _CONTAINERS_STRIP_RE.sub("", text)
    return text.strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


class _SupermemoryClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str,
                 search_mode: str = "hybrid", base_url: str = ""):
        self._api_key = api_key
        self._container_tag = container_tag
        self._search_mode = search_mode if search_mode in _VALID_SEARCH_MODES else _DEFAULT_SEARCH_MODE
        self._timeout = timeout
        self._base_url = _resolve_base_url(base_url)
        self._client: Any = None
        self._http_client: Any = None

    async def initialize(self) -> None:
        """Create the SDK's native async transport at an awaited boundary."""
        from supermemory import AsyncSupermemory

        http_client = await _create_httpx_client(timeout=self._timeout)
        try:
            client = AsyncSupermemory(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=0,
                default_headers={"x-sm-source": "hermes"},
                http_client=http_client,
            )
        except BaseException:
            await http_client.aclose()
            raise
        self._http_client = http_client
        self._client = client

    def _merge_metadata(self, metadata: Optional[dict]) -> dict:
        # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory
        # app so the user can filter / bulk-manage them per source agent. This is a
        # functional routing key for the user, not vendor telemetry.
        merged = {"sm_source": "hermes", **(metadata or {})}
        legacy_source = merged.pop("source", None)
        if legacy_source and "type" not in merged:
            merged["type"] = str(legacy_source)
        return merged

    async def add_memory(self, content: str, metadata: Optional[dict] = None, *,
                         entity_context: str = "", container_tag: Optional[str] = None,
                         custom_id: Optional[str] = None) -> dict:
        tag = container_tag or self._container_tag
        kwargs: dict[str, Any] = {
            "content": content.strip(),
            "container_tags": [tag],
        }
        if metadata:
            kwargs["metadata"] = self._merge_metadata(metadata)
        if entity_context:
            kwargs["entity_context"] = _clamp_entity_context(entity_context)
        if custom_id:
            kwargs["custom_id"] = custom_id
        result = await self._client.documents.add(**kwargs)
        return {"id": getattr(result, "id", "")}

    async def search_memories(self, query: str, *, limit: int = 5,
                              container_tag: Optional[str] = None,
                              search_mode: Optional[str] = None) -> list[dict]:
        tag = container_tag or self._container_tag
        mode = search_mode or self._search_mode
        kwargs: dict[str, Any] = {"q": query, "container_tag": tag, "limit": limit}
        if mode in _VALID_SEARCH_MODES:
            kwargs["search_mode"] = mode
        response = await self._client.search.memories(**kwargs)
        results = []
        for item in (getattr(response, "results", None) or []):
            results.append({
                "id": getattr(item, "id", ""),
                "memory": getattr(item, "memory", "") or "",
                "similarity": getattr(item, "similarity", None),
                "updated_at": getattr(item, "updated_at", None) or getattr(item, "updatedAt", None),
                "metadata": getattr(item, "metadata", None),
            })
        return results

    async def get_profile(self, query: Optional[str] = None, *,
                          container_tag: Optional[str] = None) -> dict:
        tag = container_tag or self._container_tag
        kwargs: dict[str, Any] = {"container_tag": tag}
        if query:
            kwargs["q"] = query
        response = await self._client.profile(**kwargs)
        profile_data = getattr(response, "profile", None)
        search_data = getattr(response, "search_results", None) or getattr(response, "searchResults", None)
        static = getattr(profile_data, "static", []) or [] if profile_data else []
        dynamic = getattr(profile_data, "dynamic", []) or [] if profile_data else []
        raw_results = getattr(search_data, "results", None) or search_data or []
        search_results = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    search_results.append(item)
                else:
                    search_results.append({
                        "memory": getattr(item, "memory", ""),
                        "updated_at": getattr(item, "updated_at", None) or getattr(item, "updatedAt", None),
                        "similarity": getattr(item, "similarity", None),
                    })
        return {"static": static, "dynamic": dynamic, "search_results": search_results}

    async def forget_memory(self, memory_id: str, *, container_tag: Optional[str] = None) -> None:
        tag = container_tag or self._container_tag
        await self._client.memories.forget(container_tag=tag, id=memory_id)

    async def forget_by_query(self, query: str, *, container_tag: Optional[str] = None) -> dict:
        results = await self.search_memories(query, limit=5, container_tag=container_tag)
        if not results:
            return {"success": False, "message": "No matching memory found to forget."}
        target = results[0]
        memory_id = target.get("id", "")
        if not memory_id:
            return {"success": False, "message": "Best matching memory has no id."}
        await self.forget_memory(memory_id, container_tag=container_tag)
        preview = (target.get("memory") or "")[:100]
        return {"success": True, "message": f'Forgot: "{preview}"', "id": memory_id}

    async def ingest_conversation(
        self,
        session_id: str,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> None:
        payload: dict = {
            "conversationId": session_id,
            "messages": messages,
            "containerTags": [self._container_tag],
        }
        if metadata:
            payload["metadata"] = self._merge_metadata(metadata)

        response = await self._http_client.post(
            f"{self._base_url}/v4/conversations",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "x-sm-source": "hermes",
            },
            timeout=self._timeout + 3,
        )
        response.raise_for_status()

    async def close(self) -> None:
        client = self._client
        http_client = self._http_client
        self._client = None
        self._http_client = None
        try:
            if client is not None:
                await client.close()
        finally:
            if http_client is not None and not http_client.is_closed:
                await http_client.aclose()


async def _resolve_container_tag_for_setup(
    hermes_home: str,
    *,
    identity: str = "default",
) -> str:
    config = await _load_supermemory_config(hermes_home)
    env_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip()
    raw_tag = env_tag or config["container_tag"]
    return _sanitize_tag(raw_tag.replace("{identity}", identity))


async def _probe_supermemory_connection(
    api_key: str,
    hermes_home: str,
    *,
    identity: str = "default",
) -> dict:
    config = await _load_supermemory_config(hermes_home)
    base_url = _resolve_base_url(config["base_url"])
    status = {
        "ok": False,
        "error": "",
        "container_tag": await _resolve_container_tag_for_setup(
            hermes_home,
            identity=identity,
        ),
        "profile_facts": 0,
        "auto_recall": bool(config["auto_recall"]),
        "auto_capture": bool(config["auto_capture"]),
    }
    if not (api_key or "").strip():
        status["error"] = "SUPERMEMORY_API_KEY not set"
        return status
    try:
        client = _SupermemoryClient(
            api_key=api_key.strip(),
            timeout=config["api_timeout"],
            container_tag=status["container_tag"],
            search_mode=config["search_mode"],
            base_url=base_url,
        )
        await client.initialize()
        profile = await client.get_profile()
        facts = [
            fact for fact in (profile.get("static") or []) + (profile.get("dynamic") or [])
            if fact and str(fact).strip()
        ]
        status["profile_facts"] = len(facts)
        status["ok"] = True
    except ModuleNotFoundError as exc:
        if exc.name == "supermemory":
            status["error"] = "supermemory package not installed"
        else:
            status["error"] = str(exc).strip()[:160] or "connection failed"
    except Exception as exc:
        status["error"] = str(exc).strip()[:160] or "connection failed"
    finally:
        if "client" in locals():
            await client.close()
    return status


def _format_connection_summary(status: dict) -> str:
    recall = "on" if status.get("auto_recall") else "off"
    capture = "on" if status.get("auto_capture") else "off"
    container = status.get("container_tag") or _DEFAULT_CONTAINER_TAG
    if status.get("ok"):
        facts = int(status.get("profile_facts") or 0)
        fact_label = "fact" if facts == 1 else "facts"
        return (
            f"✓ Connected · container: {container} · {facts} profile {fact_label} · "
            f"auto_recall {recall} · auto_capture {capture}"
        )
    err = status.get("error") or "connection failed"
    return f"✗ {err} · container: {container} · auto_recall {recall} · auto_capture {capture}"


STORE_SCHEMA = {
    "name": "supermemory_store",
    "description": "Store an explicit memory for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "metadata": {"type": "object", "description": "Optional metadata attached to the memory."},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "supermemory_search",
    "description": "Search long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 20."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "supermemory_forget",
    "description": "Forget a memory by exact id or by best-match query.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Exact memory id to delete."},
            "query": {"type": "string", "description": "Query used to find the memory to forget."},
        },
    },
}

PROFILE_SCHEMA = {
    "name": "supermemory_profile",
    "description": "Retrieve persistent profile facts and recent memory context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional query to focus the profile response."},
        },
    },
}


class SupermemoryMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = _default_config()
        self._api_key = ""
        self._client: Optional[_SupermemoryClient] = None
        self._container_tag = _DEFAULT_CONTAINER_TAG
        self._session_id = ""
        self._turn_count = 0
        self._auto_recall = True
        self._auto_capture = True
        self._max_recall_results = _DEFAULT_MAX_RECALL_RESULTS
        self._profile_frequency = _DEFAULT_PROFILE_FREQUENCY
        self._capture_mode = _DEFAULT_CAPTURE_MODE
        self._search_mode = _DEFAULT_SEARCH_MODE
        self._entity_context = _DEFAULT_ENTITY_CONTEXT
        self._api_timeout = _DEFAULT_API_TIMEOUT
        self._base_url = _DEFAULT_BASE_URL
        self._hermes_home = ""
        self._write_enabled = True
        self._active = False
        # Multi-container support
        self._enable_custom_containers = False
        self._custom_containers: List[str] = []
        self._custom_container_instructions = ""
        self._allowed_containers: List[str] = []
        self._session_turns: List[Dict[str, str]] = []

    @property
    def name(self) -> str:
        return "supermemory"

    async def is_available(self) -> bool:
        # Key presence only — no SDK import or network call. The optional
        # dependency is validated at the awaited initialization boundary.
        # Mirrors honcho/mem0, which check configuration only.
        return bool(get_secret("SUPERMEMORY_API_KEY", ""))

    def get_config_schema(self):
        # Only prompt for the API key during `hermes memory setup`.
        # All other options are documented for $HERMES_HOME/supermemory.json
        # or the SUPERMEMORY_CONTAINER_TAG env var.
        return [
            {"key": "api_key", "description": "Supermemory API key", "secret": True, "required": True, "env_var": "SUPERMEMORY_API_KEY", "url": _API_KEY_URL},
        ]

    async def save_config(self, values, hermes_home):
        sanitized = dict(values or {})
        if "container_tag" in sanitized:
            sanitized["container_tag"] = _sanitize_tag(str(sanitized["container_tag"]))
        if "entity_context" in sanitized:
            sanitized["entity_context"] = _clamp_entity_context(str(sanitized["entity_context"]))
        await _save_supermemory_config(sanitized, hermes_home)

    async def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id
        self._turn_count = 0
        self._config = await _load_supermemory_config(self._hermes_home)
        self._api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""

        # Resolve container tag: env var > config > default.
        # Supports {identity} template for profile-scoped containers.
        env_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip()
        raw_tag = env_tag or self._config["container_tag"]
        identity = kwargs.get("agent_identity", "default")
        self._container_tag = _sanitize_tag(raw_tag.replace("{identity}", identity))

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_recall_results = self._config["max_recall_results"]
        self._profile_frequency = self._config["profile_frequency"]
        self._capture_mode = self._config["capture_mode"]
        self._search_mode = self._config["search_mode"]
        self._entity_context = self._config["entity_context"]
        self._api_timeout = self._config["api_timeout"]
        # Base URL: config > SUPERMEMORY_BASE_URL env var > api.supermemory.ai.
        # Supports self-hosted Supermemory servers.
        self._base_url = _resolve_base_url(self._config["base_url"])
        self._enable_custom_containers = self._config["enable_custom_container_tags"]
        self._custom_containers = self._config["custom_containers"]
        self._custom_container_instructions = self._config["custom_container_instructions"]
        self._allowed_containers = [self._container_tag] + list(self._custom_containers)

        self._session_turns = []

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}
        self._active = bool(self._api_key)
        self._client = None
        if self._active:
            client: Optional[_SupermemoryClient] = None
            try:
                client = _SupermemoryClient(
                    api_key=self._api_key,
                    timeout=self._api_timeout,
                    container_tag=self._container_tag,
                    search_mode=self._search_mode,
                    base_url=self._base_url,
                )
                await client.initialize()
                self._client = client
            except asyncio.CancelledError:
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        logger.debug(
                            "Supermemory cleanup after initialization cancellation failed",
                            exc_info=True,
                        )
                self._active = False
                self._client = None
                raise
            except Exception:
                logger.warning("Supermemory initialization failed", exc_info=True)
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        logger.debug(
                            "Supermemory cleanup after initialization failure failed",
                            exc_info=True,
                        )
                self._active = False
                self._client = None

    async def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        lines = [
            "# Supermemory",
            f"Active. Container: {self._container_tag}.",
            "Use supermemory-search, supermemory-save, supermemory-forget, and supermemory-profile (aliases: supermemory_search, supermemory_store, supermemory_forget, supermemory_profile).",
        ]
        if self._enable_custom_containers and self._custom_containers:
            tags_str = ", ".join(self._allowed_containers)
            lines.append(f"\nMulti-container mode enabled. Available containers: {tags_str}.")
            lines.append("Pass an optional container_tag to supermemory_search, supermemory_store, supermemory_forget, and supermemory_profile to target a specific container.")
            if self._custom_container_instructions:
                lines.append(f"\n{self._custom_container_instructions}")
        return "\n".join(lines)

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        try:
            profile = await self._client.get_profile(query=query[:200])
            include_profile = self._turn_count <= 1 or (self._turn_count % self._profile_frequency == 0)
            context = _format_prefetch_context(
                static_facts=profile["static"] if include_profile else [],
                dynamic_facts=profile["dynamic"] if include_profile else [],
                search_results=profile["search_results"],
                max_results=self._max_recall_results,
            )
            return context
        except Exception:
            logger.debug("Supermemory prefetch failed", exc_info=True)
            return ""

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._client:
            return

        clean_user = _clean_text_for_capture(user_content)
        clean_assistant = _clean_text_for_capture(assistant_content)
        if not clean_user and not clean_assistant:
            return

        # Buffer every turn for the single full-session document written at end/switch/shutdown
        self._session_turns.append({"user": clean_user, "assistant": clean_assistant})

    async def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active or not self._write_enabled or not self._client or not self._session_id:
            return
        cleaned = []
        for message in messages or []:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = _clean_text_for_capture(str(message.get("content", "")))
            if content:
                cleaned.append({"role": role, "content": content})
        if not cleaned:
            return
        if len(cleaned) == 1 and len(cleaned[0].get("content", "")) < 20:
            return
        try:
            await self._client.ingest_conversation(
                self._session_id,
                cleaned,
                metadata={
                    "type": "full_session",
                    "session_id": self._session_id,
                    "message_count": len(cleaned),
                },
            )
        except Exception:
            logger.warning("Supermemory session ingest failed", exc_info=True)

        # Clear buffer so shutdown() doesn't duplicate on normal exit
        self._session_turns = []

    async def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Flush any buffered turns from the old session as one document, then reset for the new session."""
        if not self._active or not self._write_enabled or not self._client:
            self._session_id = str(new_session_id or "").strip() or self._session_id
            self._session_turns = []
            return

        old_session_id = self._session_id
        old_turns = list(self._session_turns)

        # Flush previous session via conversations ingest (with metadata)
        if old_turns and old_session_id:
            messages: list[dict] = []
            for turn in old_turns:
                if turn.get("user"):
                    messages.append({"role": "user", "content": turn["user"]})
                if turn.get("assistant"):
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            try:
                await self._client.ingest_conversation(
                    old_session_id,
                    messages,
                    metadata={
                        "type": "full_session",
                        "session_id": old_session_id,
                        "message_count": len(old_turns) * 2,
                        "partial": not reset,
                    },
                )
            except Exception:
                logger.debug("Supermemory session-switch ingest failed", exc_info=True)

        # Reset for new session
        self._session_id = str(new_session_id or "").strip() or old_session_id
        self._session_turns = []
        self._turn_count = 0

    async def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        if not self._active or not self._write_enabled or not self._client:
            return
        if action != "add" or not (content or "").strip():
            return

        try:
            await self._client.add_memory(
                content.strip(),
                metadata={"target": target, "type": "explicit_memory"},
                entity_context=self._entity_context,
            )
        except Exception:
            logger.debug("Supermemory on_memory_write failed", exc_info=True)

    async def shutdown(self) -> None:
        client = self._client
        cancellation: asyncio.CancelledError | None = None
        try:
            # Emergency fallback (crashes only). Buffer is cleared on normal on_session_end().
            if self._active and self._write_enabled and client and self._session_turns and self._session_id:
                logger.warning("Supermemory: Saving session via shutdown (session=%s, turns=%d)", self._session_id, len(self._session_turns))

                messages: list[dict] = []
                for turn in self._session_turns:
                    if turn.get("user"):
                        messages.append({"role": "user", "content": turn["user"]})
                    if turn.get("assistant"):
                        messages.append({"role": "assistant", "content": turn["assistant"]})

                try:
                    await client.ingest_conversation(
                        self._session_id,
                        messages,
                        metadata={
                            "type": "full_session",
                            "session_id": self._session_id,
                            "message_count": len(self._session_turns) * 2,
                            "partial": True,
                        },
                    )
                except Exception:
                    logger.debug("Supermemory shutdown ingest failed", exc_info=True)
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - close before re-raise
            cancellation = exc

        close_error: Exception | None = None
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                close_error = exc
        self._client = None
        self._active = False

        if cancellation is not None:
            if close_error is not None:
                logger.debug(
                    "Supermemory close failed while propagating cancellation",
                    exc_info=(
                        type(close_error),
                        close_error,
                        close_error.__traceback__,
                    ),
                )
            raise cancellation
        if close_error is not None:
            raise close_error

    def _resolve_tool_container_tag(self, args: dict) -> Optional[str]:
        """Validate and resolve container_tag from tool call args.

        Returns None (use primary) if multi-container is disabled or no tag provided.
        Returns the validated tag if it's in the allowed list.
        Raises ValueError if the tag is not whitelisted.
        """
        if not self._enable_custom_containers:
            return None
        tag = str(args.get("container_tag") or "").strip()
        if not tag:
            return None
        sanitized = _sanitize_tag(tag)
        if sanitized not in self._allowed_containers:
            raise ValueError(
                f"Container tag '{sanitized}' is not allowed. "
                f"Allowed: {', '.join(self._allowed_containers)}"
            )
        return sanitized

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        def with_kebab_aliases(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            aliases = {
                "supermemory_store": "supermemory-save",
                "supermemory_search": "supermemory-search",
                "supermemory_forget": "supermemory-forget",
                "supermemory_profile": "supermemory-profile",
            }
            expanded = list(schemas)
            for schema in schemas:
                kebab = aliases.get(schema.get("name", ""))
                if not kebab:
                    continue
                copy = json.loads(json.dumps(schema))
                copy["name"] = kebab
                expanded.append(copy)
            return expanded

        if not self._enable_custom_containers:
            return with_kebab_aliases([STORE_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, PROFILE_SCHEMA])

        # When multi-container is enabled, add optional container_tag to relevant tools
        container_param = {
            "type": "string",
            "description": f"Optional container tag. Allowed: {', '.join(self._allowed_containers)}. Defaults to primary ({self._container_tag}).",
        }
        schemas = []
        for base in [STORE_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, PROFILE_SCHEMA]:
            schema = json.loads(json.dumps(base))  # deep copy
            schema["parameters"]["properties"]["container_tag"] = container_param
            schemas.append(schema)
        return with_kebab_aliases(schemas)

    async def _tool_store(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        metadata = args.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("type", _detect_category(content))
        metadata.pop("source", None)
        try:
            result = await self._client.add_memory(
                content,
                metadata=metadata,
                entity_context=self._entity_context,
                container_tag=tag,
            )
            preview = content[:80] + ("..." if len(content) > 80 else "")
            resp: dict[str, Any] = {"saved": True, "id": result.get("id", ""), "preview": preview}
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Failed to store memory: {exc}")

    async def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            limit = max(1, min(20, int(args.get("limit", 5) or 5)))
        except Exception:
            limit = 5
        try:
            results = await self._client.search_memories(
                query,
                limit=limit,
                container_tag=tag,
            )
            formatted = []
            for item in results:
                entry: dict[str, Any] = {"id": item.get("id", ""), "content": item.get("memory", "")}
                if item.get("similarity") is not None:
                    try:
                        entry["similarity"] = round(float(item["similarity"]) * 100)
                    except Exception:
                        pass
                formatted.append(entry)
            resp: dict[str, Any] = {"results": formatted, "count": len(formatted)}
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Search failed: {exc}")

    async def _tool_forget(self, args: dict) -> str:
        memory_id = str(args.get("id") or "").strip()
        query = str(args.get("query") or "").strip()
        if not memory_id and not query:
            return tool_error("Provide either id or query")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            if memory_id:
                await self._client.forget_memory(memory_id, container_tag=tag)
                return json.dumps({"forgotten": True, "id": memory_id})
            return json.dumps(
                await self._client.forget_by_query(query, container_tag=tag)
            )
        except Exception as exc:
            return tool_error(f"Forget failed: {exc}")

    async def _tool_profile(self, args: dict) -> str:
        query = str(args.get("query") or "").strip() or None
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            profile = await self._client.get_profile(
                query=query,
                container_tag=tag,
            )
            sections = []
            if profile["static"]:
                sections.append("## User Profile (Persistent)\n" + "\n".join(f"- {item}" for item in profile["static"]))
            if profile["dynamic"]:
                sections.append("## Recent Context\n" + "\n".join(f"- {item}" for item in profile["dynamic"]))
            resp: dict[str, Any] = {
                "profile": "\n\n".join(sections),
                "static_count": len(profile["static"]),
                "dynamic_count": len(profile["dynamic"]),
            }
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Profile failed: {exc}")

    async def handle_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs,
    ) -> str:
        if not self._active or not self._client:
            return tool_error("Supermemory is not configured")
        aliases = {
            "supermemory-save": "supermemory_store",
            "supermemory-search": "supermemory_search",
            "supermemory-forget": "supermemory_forget",
            "supermemory-profile": "supermemory_profile",
        }
        tool_name = aliases.get(tool_name, tool_name)
        if tool_name == "supermemory_store":
            return await self._tool_store(args)
        if tool_name == "supermemory_search":
            return await self._tool_search(args)
        if tool_name == "supermemory_forget":
            return await self._tool_forget(args)
        if tool_name == "supermemory_profile":
            return await self._tool_profile(args)
        return tool_error(f"Unknown tool: {tool_name}")


def register(ctx):
    ctx.register_memory_provider(SupermemoryMemoryProvider())
