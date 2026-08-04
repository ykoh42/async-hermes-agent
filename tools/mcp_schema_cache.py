"""Persistent MCP tool-schema cache for lazy server startup.

Stores per-server tool manifests on disk so Hermes can register MCP tools
into the agent snapshot without spawning the stdio child process at idle
dashboard startup. Cache entries are keyed by server name + a fingerprint
of the connection config (command/args/url/tools filters).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "mcp_schema_cache.json"
_cache_lock = asyncio.Lock()


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / _CACHE_FILENAME


def config_fingerprint(config: dict) -> str:
    """Stable hash of the connection-defining parts of an MCP server config."""
    tools_filter = config.get("tools") or {}
    payload = {
        "command": config.get("command"),
        "args": config.get("args") or [],
        "url": config.get("url"),
        "transport": config.get("transport"),
        "tools_include": sorted(tools_filter.get("include") or []),
        "tools_exclude": sorted(tools_filter.get("exclude") or []),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def _load_all() -> Dict[str, Any]:
    path = _cache_path()
    if not await aiofiles.os.path.exists(path):
        return {}
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Could not read MCP schema cache %s: %s", path, exc)
        return {}


async def _save_all(data: Dict[str, Any]) -> None:
    """Atomically persist the user-writable cache without blocking the loop."""
    path = _cache_path()
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        async with aiofiles.open(temporary, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(data, ensure_ascii=False, indent=2))
            await handle.flush()
            await aiofiles.os.wrap(os.fsync)(handle.fileno())
        await aiofiles.os.wrap(os.chmod)(temporary, 0o600)
        await aiofiles.os.replace(temporary, path)
    finally:
        try:
            if await aiofiles.os.path.exists(temporary):
                await aiofiles.os.remove(temporary)
        except OSError:
            pass


async def get_cached_entry(server_name: str, fingerprint: str) -> Optional[dict]:
    """Return cached entry when fingerprint matches, else None."""
    async with _cache_lock:
        entry = (await _load_all()).get(server_name)
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != fingerprint:
        return None
    return entry


async def has_cached_entry(server_name: str, fingerprint: str) -> bool:
    return await get_cached_entry(server_name, fingerprint) is not None


async def write_cache_entry(
    server_name: str,
    fingerprint: str,
    *,
    tools: List[dict],
    utility_tools: Optional[List[dict]] = None,
) -> None:
    """Persist tool schemas after a successful live connect."""
    entry = {
        "fingerprint": fingerprint,
        "tools": tools,
        "utility_tools": utility_tools or [],
    }
    async with _cache_lock:
        data = await _load_all()
        # Write-through fires on every registration (reconnects,
        # list_changed refreshes); skip the load-all+rewrite churn when the
        # entry is byte-identical to what is already on disk.
        if data.get(server_name) == entry:
            return
        data[server_name] = entry
        await _save_all(data)


async def clear_cache_entry(server_name: str) -> None:
    async with _cache_lock:
        data = await _load_all()
        if server_name in data:
            del data[server_name]
            await _save_all(data)


def tools_from_cache_entry(entry: dict) -> List[dict]:
    """Return cached MCP tool dicts (name, description, inputSchema)."""
    tools = entry.get("tools")
    return list(tools) if isinstance(tools, list) else []


def utility_tools_from_cache_entry(entry: dict) -> List[dict]:
    util = entry.get("utility_tools")
    return list(util) if isinstance(util, list) else []
