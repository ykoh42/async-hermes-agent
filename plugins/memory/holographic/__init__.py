"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import aiofiles.os
import aiofiles.tempfile
import yaml

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from utils import is_truthy_value
from .store import MemoryStore, _finish_owned_task
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

async def _load_plugin_config(hermes_home: str | Path | None = None) -> dict:
    token = None
    try:
        if hermes_home is not None:
            from hermes_constants import set_hermes_home_override

            token = set_hermes_home_override(hermes_home)
        # Canonical loader: behavioral read now honors the managed-scope
        # overlay + ${VAR} expansion (e.g. an api key template) too.
        from hermes_cli.config import load_config_readonly
        all_config = await load_config_readonly()
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}
    finally:
        if token is not None:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)


async def _remove_config_temporary_file(path: str) -> None:
    try:
        await aiofiles.os.remove(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config
        self._store = None
        self._retriever = None
        self._min_trust = float((config or {}).get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    async def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    async def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            # Write-back round-trip: raw read is correct (merged defaults
            # must not be persisted back into the user's file).
            from hermes_cli.config import read_user_config_raw

            existing = await read_user_config_raw(config_path)
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            await aiofiles.os.makedirs(config_path.parent, exist_ok=True)
            target_path = await aiofiles.os.wrap(os.path.realpath)(config_path)
            target = Path(target_path)
            original_mode = None
            original_owner = None
            try:
                target_stat = await aiofiles.os.stat(target)
                original_mode = stat.S_IMODE(target_stat.st_mode)
                if os.name == "posix":
                    original_owner = (target_stat.st_uid, target_stat.st_gid)
            except OSError:
                pass
            temporary = ""
            try:
                async with aiofiles.tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=target.parent,
                    prefix=f".{target.stem}_",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = handle.name
                    if hasattr(os, "fchmod"):
                        await aiofiles.os.wrap(os.fchmod)(handle.fileno(), 0o600)
                    await handle.write(
                        yaml.dump(
                            existing,
                            default_flow_style=False,
                        )
                    )
                    await handle.flush()
                    await aiofiles.os.wrap(os.fsync)(handle.fileno())
                await aiofiles.os.replace(temporary, target)
                temporary = ""
                if original_owner is not None and hasattr(os, "chown"):
                    try:
                        await aiofiles.os.wrap(os.chown)(
                            target,
                            original_owner[0],
                            original_owner[1],
                        )
                    except OSError:
                        pass
                if original_mode is not None:
                    try:
                        await aiofiles.os.wrap(os.chmod)(target, original_mode)
                    except OSError:
                        pass
            except BaseException:
                if temporary:
                    cleanup_task = asyncio.create_task(
                        _remove_config_temporary_file(temporary)
                    )
                    await _finish_owned_task(cleanup_task)
                raise
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    async def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        if self._config is None:
            self._config = await _load_plugin_config(_hermes_home)
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        await self._store.initialize()
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        total = self._store._current_fact_count()
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = await self._retriever.search(
                query,
                min_trust=self._min_trust,
                limit=5,
            )
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    async def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Holographic memory stores explicit facts via tools, not auto-sync.
        # The on_session_end hook handles auto-extraction if configured.
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return await self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return await self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    async def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # is_truthy_value: the config schema declares auto_extract as a string
        # enum ("false"/"true"), and a plain truthiness check treats the string
        # "false" as enabled (#57682).
        if not is_truthy_value(self._config.get("auto_extract", False)):
            return
        if not self._store or not messages:
            return
        await self._auto_extract_facts(messages)

    async def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                await self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    async def shutdown(self) -> None:
        # Release the shared SQLite connection deterministically on the
        # caller's thread. Dropping the reference alone leaves fd finalization
        # to GC, which keeps the connection (and its write lock) alive on a
        # long-running gateway and prolongs the "database is locked" contention
        # this store's shared-connection refcounting is meant to eliminate.
        # close() is idempotent and refcount-guarded, so siblings stay safe.
        if self._store is not None:
            try:
                await self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    async def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = await store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = await retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = await retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = await retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = await retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = await retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = await store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = await store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = await store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    async def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = await self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    async def _auto_extract_facts(self, messages: list) -> None:
        # Local import (pattern used in initialize()): the compressor module is
        # heavier than this plugin and is only needed when auto_extract is on.
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            is_compaction_summary_message,
        )

        def _pre_delimiter_user_segment(msg: dict):
            """Return the genuine user text preceding a merged-into-tail
            compaction summary, or None when the whole message is a summary.

            Merge-into-tail messages (agent/context_compressor.py ~3163-3190)
            wrap real prior tail content BEFORE ``_MERGED_SUMMARY_DELIMITER``,
            prefixed with ``_MERGED_PRIOR_CONTEXT_HEADER``, then append the
            generated handoff summary AFTER the delimiter. Dropping the whole
            row (as ``is_compaction_summary_message`` alone would suggest)
            discards that genuine pre-delimiter content too (#57690 review).
            Only the summary suffix must be excluded from harvesting.
            """
            content = msg.get("content", "")
            if not isinstance(content, str) or _MERGED_SUMMARY_DELIMITER not in content:
                return None
            pre = content.split(_MERGED_SUMMARY_DELIMITER, 1)[0]
            if pre.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
                pre = pre[len(_MERGED_PRIOR_CONTEXT_HEADER):]
            pre = pre.strip()
            return pre or None

        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            # Compaction handoff summaries can be inserted as role="user"
            # messages; their prose reliably matches the decision patterns, so
            # without this guard the compactor's own output is stored as a
            # durable "fact" on every rollover (#57682). A merge-into-tail
            # summary also carries genuine pre-delimiter user content in the
            # SAME row; harvest that segment instead of dropping the whole
            # message (#57690 review).
            pre_delimiter_segment = _pre_delimiter_user_segment(msg)
            if pre_delimiter_segment is not None:
                content = pre_delimiter_segment
            elif is_compaction_summary_message(msg):
                continue
            else:
                content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        await self._store.add_fact(
                            content[:400],
                            category="user_pref",
                        )
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        await self._store.add_fact(
                            content[:400],
                            category="project",
                        )
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    ctx.register_memory_provider(HolographicMemoryProvider())
