"""ByteRover memory plugin — MemoryProvider interface.

Persistent memory via the ByteRover CLI (``brv``). Organizes knowledge into
a hierarchical context tree with tiered retrieval (fuzzy text → LLM-driven
search). Local-first with optional cloud sync.

Original PR #3499 by hieuntg81, adapted to MemoryProvider ABC.

Requires: ``brv`` CLI installed (npm install -g byterover-cli or
curl -fsSL https://byterover.dev/install.sh | sh).

Config via environment variables (profile-scoped via each profile's .env):
  BRV_API_KEY   — ByteRover API key (for cloud features, optional for local)

Config via config.yaml:
  memory:
    byterover:
      auto_extract: false  # disable automatic brv curate hooks

Working directory: $HERMES_HOME/byterover/ (profile-scoped context tree)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles.os

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Timeouts
_QUERY_TIMEOUT = 10   # brv query — should be fast
_CURATE_TIMEOUT = 120  # brv curate — may involve LLM processing

# Minimum lengths to filter noise
_MIN_QUERY_LEN = 10
_MIN_OUTPUT_LEN = 20


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


async def _load_plugin_config() -> Dict[str, Any]:
    """Read ByteRover's profile-scoped memory config.

    New memory-provider setup stores non-secret provider settings under
    ``memory.<provider>``.  Some users also set ``memory.provider_config`` from
    early docs/issues, so accept it as a compatibility fallback.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        memory_config = config.get("memory", {})
        if not isinstance(memory_config, dict):
            return {}

        provider_config = memory_config.get("byterover", {})
        if isinstance(provider_config, dict) and provider_config:
            return dict(provider_config)

        legacy_config = memory_config.get("provider_config", {})
        if isinstance(legacy_config, dict):
            return dict(legacy_config)
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# brv binary resolution (cached)
# ---------------------------------------------------------------------------

_cached_brv_path: Optional[str] = None


def _resolve_brv_path() -> Optional[str]:
    """Find the brv binary on PATH or well-known install locations."""
    global _cached_brv_path
    if _cached_brv_path is not None:
        return _cached_brv_path if _cached_brv_path != "" else None

    found = shutil.which("brv")
    if not found:
        home = Path.home()
        candidates = [
            home / ".brv-cli" / "bin" / "brv",
            Path("/usr/local/bin/brv"),
            home / ".npm-global" / "bin" / "brv",
        ]
        for c in candidates:
            if c.exists():
                found = str(c)
                break

    _cached_brv_path = found or ""
    return found


async def _run_brv(
    args: List[str],
    timeout: int = _QUERY_TIMEOUT,
    cwd: Optional[str] = None,
) -> dict:
    """Run a brv CLI command. Returns {success, output, error}."""
    brv_path = _resolve_brv_path()
    if not brv_path:
        return {"success": False, "error": "brv CLI not found. Install: npm install -g byterover-cli"}

    cmd = [brv_path] + args
    effective_cwd = cwd or str(_get_brv_cwd())
    await aiofiles.os.makedirs(effective_cwd, exist_ok=True)

    env = os.environ.copy()
    brv_bin_dir = str(Path(brv_path).parent)
    env["PATH"] = brv_bin_dir + os.pathsep + env.get("PATH", "")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=effective_cwd,
            env=env,
        )
        try:
            async with asyncio.timeout(timeout):
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            process.kill()
            await process.wait()
            return {"success": False, "error": f"brv timed out after {timeout}s"}
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if process.returncode == 0:
            return {"success": True, "output": stdout}
        return {
            "success": False,
            "error": stderr or stdout or f"brv exited {process.returncode}",
        }

    except FileNotFoundError:
        global _cached_brv_path
        _cached_brv_path = None
        return {"success": False, "error": "brv CLI not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_brv_cwd() -> Path:
    """Profile-scoped working directory for the brv context tree."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "byterover"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

QUERY_SCHEMA = {
    "name": "brv_query",
    "description": (
        "Search ByteRover's persistent knowledge tree for relevant context. "
        "Returns memories, project knowledge, architectural decisions, and "
        "patterns from previous sessions. Use for any question where past "
        "context would help."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

CURATE_SCHEMA = {
    "name": "brv_curate",
    "description": (
        "Store important information in ByteRover's persistent knowledge tree. "
        "Use for architectural decisions, bug fixes, user preferences, project "
        "patterns — anything worth remembering across sessions. ByteRover's LLM "
        "automatically categorizes and organizes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "brv_status",
    "description": "Check ByteRover status — CLI version, context tree stats, cloud sync state.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ByteRoverMemoryProvider(MemoryProvider):
    """ByteRover persistent memory via the brv CLI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else None
        config_snapshot = self._config or {}
        self._auto_extract = _coerce_bool(config_snapshot.get("auto_extract"), True)
        self._cwd = ""
        self._session_id = ""
        self._turn_count = 0

    @property
    def name(self) -> str:
        return "byterover"

    async def is_available(self) -> bool:
        """Check if brv CLI is installed. No network calls."""
        return _resolve_brv_path() is not None

    def get_config_schema(self):
        return [
            {
                "key": "api_key",
                "description": "ByteRover API key (optional, for cloud sync)",
                "secret": True,
                "env_var": "BRV_API_KEY",
                "url": "https://app.byterover.dev",
            },
            {
                "key": "auto_extract",
                "description": "Automatically curate completed turns and compression/memory hooks",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    async def initialize(self, session_id: str, **kwargs) -> None:
        if self._config is None:
            self._config = await _load_plugin_config()
            self._auto_extract = _coerce_bool(
                self._config.get("auto_extract"), True
            )
        self._cwd = str(_get_brv_cwd())
        self._session_id = session_id
        self._turn_count = 0
        await aiofiles.os.makedirs(self._cwd, exist_ok=True)

    def system_prompt_block(self) -> str:
        if not _resolve_brv_path():
            return ""
        return (
            "# ByteRover Memory\n"
            "Active. Persistent knowledge tree with hierarchical context.\n"
            "Use brv_query to search past knowledge, brv_curate to store "
            "important facts, brv_status to check state."
        )

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run brv query before the agent's first LLM call."""
        if not query or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        result = await _run_brv(
            ["query", "--", query.strip()[:5000]],
            timeout=_QUERY_TIMEOUT, cwd=self._cwd,
        )
        if result["success"] and result.get("output"):
            output = result["output"].strip()
            if len(output) > _MIN_OUTPUT_LEN:
                return f"## ByteRover Context\n{output}"
        return ""

    async def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs at turn start."""

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Curate a substantive completed conversation turn."""
        self._turn_count += 1
        if not self._auto_extract:
            logger.debug("ByteRover sync_turn skipped (auto_extract disabled)")
            return

        # Only curate substantive turns
        if len(user_content.strip()) < _MIN_QUERY_LEN:
            return

        combined = (
            f"User: {user_content[:2000]}\n"
            f"Assistant: {assistant_content[:2000]}"
        )
        await _run_brv(
            ["curate", "--", combined],
            timeout=_CURATE_TIMEOUT,
            cwd=self._cwd,
        )

    async def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to ByteRover."""
        if not self._auto_extract:
            logger.debug("ByteRover memory mirror skipped (auto_extract disabled)")
            return
        if action not in {"add", "replace"} or not content:
            return

        label = "User profile" if target == "user" else "Agent memory"
        await _run_brv(
            ["curate", "--", f"[{label}] {content}"],
            timeout=_CURATE_TIMEOUT,
            cwd=self._cwd,
        )

    async def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights before context compression discards turns."""
        if not self._auto_extract:
            logger.debug("ByteRover pre-compression flush skipped (auto_extract disabled)")
            return ""
        if not messages:
            return ""

        # Build a summary of messages about to be compressed
        parts = []
        for msg in messages[-10:]:  # last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip() and role in {"user", "assistant"}:
                parts.append(f"{role}: {content[:500]}")

        if not parts:
            return ""

        combined = "\n".join(parts)

        await _run_brv(
            ["curate", "--", f"[Pre-compression context]\n{combined}"],
            timeout=_CURATE_TIMEOUT,
            cwd=self._cwd,
        )
        logger.info("ByteRover pre-compression flush: %d messages", len(parts))
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [QUERY_SCHEMA, CURATE_SCHEMA, STATUS_SCHEMA]

    async def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "brv_query":
            return await self._tool_query(args)
        elif tool_name == "brv_curate":
            return await self._tool_curate(args)
        elif tool_name == "brv_status":
            return await self._tool_status()
        return tool_error(f"Unknown tool: {tool_name}")

    async def shutdown(self) -> None:
        """ByteRover subprocesses are awaited by their owning operations."""

    # -- Tool implementations ------------------------------------------------

    async def _tool_query(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        result = await _run_brv(
            ["query", "--", query.strip()[:5000]],
            timeout=_QUERY_TIMEOUT, cwd=self._cwd,
        )

        if not result["success"]:
            return tool_error(result.get("error", "Query failed"))

        output = result.get("output", "").strip()
        if not output or len(output) < _MIN_OUTPUT_LEN:
            return json.dumps({"result": "No relevant memories found."})

        # Truncate very long results
        if len(output) > 8000:
            output = output[:8000] + "\n\n[... truncated]"

        return json.dumps({"result": output})

    async def _tool_curate(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        result = await _run_brv(
            ["curate", "--", content],
            timeout=_CURATE_TIMEOUT, cwd=self._cwd,
        )

        if not result["success"]:
            return tool_error(result.get("error", "Curate failed"))

        return json.dumps({"result": "Memory curated successfully."})

    async def _tool_status(self) -> str:
        result = await _run_brv(["status"], timeout=15, cwd=self._cwd)
        if not result["success"]:
            return tool_error(result.get("error", "Status check failed"))
        return json.dumps({"status": result.get("output", "")})


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register ByteRover as a memory provider plugin."""
    ctx.register_memory_provider(ByteRoverMemoryProvider())
