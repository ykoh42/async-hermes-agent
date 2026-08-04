#!/usr/bin/env python3
"""
Model Tools Module

Thin orchestration layer over the tool registry. Each tool file in tools/
self-registers its schema, handler, and metadata via tools.registry.register().
This module triggers discovery (by importing all tool modules), then provides
the public API that run_agent.py, cli.py, batch_runner.py, and the RL
environments consume.

Public API (signatures preserved from the original 2,400-line version):
    get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode) -> list
    handle_function_call(function_name, function_args, task_id, user_task) -> str
    TOOL_TO_TOOLSET_MAP: dict          (for batch_runner.py)
    TOOLSET_REQUIREMENTS: dict         (for cli.py, doctor.py)
    get_all_tool_names() -> list
    get_toolset_for_tool(name) -> str
    get_available_toolsets() -> dict
    check_toolset_requirements() -> dict
    check_tool_availability(quiet) -> tuple
"""

import os
import json
import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import discover_builtin_tools, registry, tool_error
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)

# Tracks platform-bundle names already flagged in disabled_toolsets so the
# advisory (#33924) is logged once per name, not on every tool recompute.
_WARNED_DISABLED_BUNDLES: set = set()


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


# =============================================================================
# Native async contract
# =============================================================================





# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

discover_builtin_tools()

# MCP discovery is deliberately lazy.  A conversation's async turn prologue
# awaits it after the agent has entered its own event loop, avoiding import-time
# network I/O and preserving per-conversation prompt-cache construction.

# This training runtime accepts capabilities through configured external MCP
# servers and file-based Skills.  Do not import the separate Hermes plugin
# runtime here: plugins can register arbitrary additional model tools as an
# import side effect and would defeat the fixed training tool surface.


# =============================================================================
# Backward-compat constants  (built once after discovery)
# =============================================================================

TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# =============================================================================
# Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
# =============================================================================

_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions  (the main schema provider)
# =============================================================================

# Module-level memoization for get_tool_definitions(). Keyed on
# (frozenset(enabled_toolsets), frozenset(disabled_toolsets), registry._generation).
# Hot callers (gateway runner, AIAgent.__init__) invoke this on every turn
# with quiet_mode=True; caching avoids ~7 ms of registry walking + schema
# filtering + check_fn probing per call. Only active when quiet_mode=True
# because quiet_mode=False has stdout side effects (tool-selection prints).
#
# Invalidation happens transparently via the registry's _generation counter,
# which bumps on register() / deregister() / register_toolset_alias(). The
# inner check_fn TTL cache in registry.py handles environment drift (Docker
# daemon start/stop, env var changes, etc.) on a 30 s horizon.
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}

# Hard cap on memoized get_tool_definitions() results. A long-lived Gateway
# process sees many distinct toolset/config fingerprints over its lifetime
# (per-session toolset sets, config edits, kanban-task toggles); without a
# bound the cache grows unboundedly. 8 comfortably covers the warm working
# set (the handful of distinct platform/toolset combos a gateway actually
# serves) while keeping the cap small. (#19251)
_TOOL_DEFS_CACHE_MAX = 8


def _clear_tool_defs_cache() -> None:
    """Drop memoized get_tool_definitions() results."""
    _tool_defs_cache.clear()


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
    probe_availability: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get tool definitions for model API calls with toolset-based filtering.

    All tools must be part of a toolset to be accessible.

    Args:
        enabled_toolsets: Only include tools from these toolsets.
        disabled_toolsets: Exclude tools from these toolsets (if enabled_toolsets is None).
        quiet_mode: Suppress status prints.
        skip_tool_search_assembly: When True, return the pre-assembly tool list
            (raw schemas for every enabled tool). Used internally by the
            tool_search / tool_describe bridge handlers so they can read the
            real catalog, not the already-collapsed one. Public callers should
            leave this False.

    Returns:
        Filtered list of OpenAI-format tool definitions.
    """
    # Fast path: memoized result when the caller doesn't need stdout prints.
    # The cache key captures every argument-level input; the registry
    # generation captures registry mutations (MCP refresh, plugin load).
    # check_fn results are TTL-cached one level down, inside
    # registry.get_definitions. The config-mtime fingerprint below captures
    # user-visible config edits that affect dynamic schemas without needing an explicit
    # invalidate hook on every config-writer.
    if quiet_mode and probe_availability:
        try:
            from hermes_cli.config import get_config_path
            cfg_path = get_config_path()
            cfg_stat = cfg_path.stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        cache_key = (
            frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
            frozenset(disabled_toolsets) if disabled_toolsets else None,
            registry._generation,
            cfg_fp,
            bool(os.environ.get("HERMES_KANBAN_TASK")),
            bool(skip_tool_search_assembly),
            _is_delegated_child_context(),
        )
        cached = _tool_defs_cache.get(cache_key)
        if cached is not None:
            # Return a shallow copy of the list but share the dict references —
            # schemas are treated as read-only by all known callers.
            return list(cached)

    result = _compute_tool_definitions(
        enabled_toolsets,
        disabled_toolsets,
        quiet_mode,
        skip_tool_search_assembly=skip_tool_search_assembly,
        probe_availability=probe_availability,
    )
    if quiet_mode and probe_availability:
        # Cache the freshly-computed list, but hand callers a shallow copy so
        # downstream mutations (e.g. run_agent appending memory/LCM tool
        # schemas to self.tools) don't poison the cache. Without this, a
        # long-lived Gateway process accumulates duplicate tool names across
        # agent inits and providers that enforce unique tool names
        # (DeepSeek, Xiaomi MiMo, Moonshot Kimi) reject the request with
        # HTTP 400. Mirrors the cache-hit path above. (issue #17335)
        # Bound the cache with LRU eviction so a long-lived Gateway process
        # doesn't accumulate entries unboundedly across the many distinct
        # toolset/config fingerprints it sees over its lifetime (#19251).
        if len(_tool_defs_cache) >= _TOOL_DEFS_CACHE_MAX:
            _tool_defs_cache.pop(next(iter(_tool_defs_cache)))  # evict oldest
        _tool_defs_cache[cache_key] = result
        return list(result)
    return result


def _compute_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
    probe_availability: bool = True,
) -> List[Dict[str, Any]]:
    """Uncached implementation of :func:`get_tool_definitions`."""
    # Determine which tool names the caller wants
    tools_to_include: set = set()

    if enabled_toolsets is not None:
        effective_enabled_toolsets = list(enabled_toolsets)
        if (
            os.environ.get("HERMES_KANBAN_TASK")
            and not _is_delegated_child_context()
            and "kanban" not in effective_enabled_toolsets
        ):
            # Dispatcher-spawned workers are scoped by HERMES_KANBAN_TASK and
            # must always receive the lifecycle handoff tools. Assignee
            # profiles may intentionally restrict their normal chat toolsets
            # (for token/cost reasons), but that should not strip the kanban
            # worker's completion/block/heartbeat surface.
            effective_enabled_toolsets.append("kanban")
        for toolset_name in effective_enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        # Default: start with everything
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # Always apply disabled toolsets as a subtraction step at the end.
    # This ensures that even if a composite toolset (like hermes-cli)
    # is enabled, any tools belonging to a disabled toolset are strictly
    # stripped out. See issue #17309.
    if disabled_toolsets:
        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                from toolsets import bundle_non_core_tools, get_toolset
                if toolset_name.startswith("hermes-") or (get_toolset(toolset_name) or {}).get("posture"):
                    # Platform bundles (hermes-*) include _HERMES_CORE_TOOLS, and
                    # posture toolsets (`posture: True`, e.g. `coding`) re-list
                    # those same core tools without owning them, so subtracting
                    # the whole toolset would strip core tools shared by other
                    # enabled toolsets and empty the tool list (#33924, #57315).
                    # Subtract only the non-core delta; keep core.
                    to_remove = bundle_non_core_tools(toolset_name)
                    tools_to_include.difference_update(to_remove)
                    resolved = sorted(to_remove)
                    if (not quiet_mode and toolset_name.startswith("hermes-")
                            and toolset_name not in _WARNED_DISABLED_BUNDLES):
                        _WARNED_DISABLED_BUNDLES.add(toolset_name)
                        logger.info(
                            "agent.disabled_toolsets contains platform-bundle "
                            "name '%s'; core tools are preserved and only its "
                            "platform-specific tools (%s) are removed. Bundle "
                            "names usually belong in `toolsets:`, not "
                            "`disabled_toolsets` (#33924).",
                            toolset_name,
                            ", ".join(resolved) if resolved else "none",
                        )
                else:
                    resolved = resolve_toolset(toolset_name)
                    tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")

    # Ask the registry for schemas (only returns tools whose check_fn passes)
    filtered_tools = registry.get_definitions(
        tools_to_include,
        quiet=quiet_mode,
        probe_availability=probe_availability,
    )

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    # Sanitize schemas for broad backend compatibility. llama.cpp's
    # json-schema-to-grammar converter (used by its OAI server to build
    # GBNF tool-call parsers) rejects some shapes that cloud providers
    # silently accept — bare "type": "object" with no properties,
    # string-valued schema nodes from malformed MCP servers, etc. This
    # is a no-op for schemas that are already well-formed.
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    return filtered_tools


def _resolve_active_context_length() -> int:
    """Look up the active model's context length for the tool-search gate.

    Returns 0 when the model can't be resolved — ``should_activate`` falls
    back to a fixed token cutoff in that case.
    """
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        model_id = (model_cfg.get("model") or model_cfg.get("default") or "").strip()
        if not model_id:
            return 0
        from agent.model_metadata import get_static_context_length
        # Resolve only from local/static metadata here.  Tool schemas are
        # assembled by the synchronous construction/configuration surface;
        # provider /models probes belong to the deferred async runtime and
        # must never block a conversation turn.
        raw_ctx = model_cfg.get("context_length")
        config_ctx = raw_ctx if isinstance(raw_ctx, int) and raw_ctx > 0 else None
        # This is a synchronous, CPU/config-only gate used while assembling
        # the tool schema.  Runtime credential resolution is deliberately not
        # performed here: ``resolve_runtime_provider`` is an async lifecycle
        # boundary and awaiting it from this helper would reintroduce a
        # hidden sync/async bridge.  Provider-aware static fallbacks still use
        # the configured provider and endpoint below; the live runtime gets
        # its authoritative context window after initialization.
        provider = str(model_cfg.get("provider") or "").strip()
        base_url = str(model_cfg.get("base_url") or "").strip()
        return int(get_static_context_length(
            model_id,
            base_url=base_url,
            config_context_length=config_ctx,
            provider=provider,
        ) or 0)
    except Exception as e:
        logger.debug("Could not resolve active context length: %s", e)
        return 0


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# =========================================================================
# Tool error sanitization
# =========================================================================
#
# Tool exceptions can carry arbitrary text into the model's context as the
# `tool` message content. json.dumps() handles quote/backslash escaping so a
# raw injection of `</tool_call>` won't break message framing, but the model
# still *reads* those tokens and they can confuse downstream tool-call
# parsing or, in adversarial cases, nudge it toward role-confusion framing.
#
# This helper strips structural framing tokens (XML role tags, CDATA,
# markdown code fences) and caps the message at a sane upper bound before it
# becomes part of the conversation. It's defense-in-depth — the json layer
# already prevents framing escape — but cheap and worth having.
#
# Ported from ironclaw#1639.
_TOOL_ERROR_ROLE_TAG_RE = re.compile(
    r'</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>',
    re.IGNORECASE,
)
_TOOL_ERROR_FENCE_OPEN_RE = re.compile(r'^\s*```(?:json|xml|html|markdown)?\s*', re.MULTILINE)
_TOOL_ERROR_FENCE_CLOSE_RE = re.compile(r'\s*```\s*$', re.MULTILINE)
_TOOL_ERROR_CDATA_RE = re.compile(r'<!\[CDATA\[.*?\]\]>', re.DOTALL)
_TOOL_ERROR_MAX_LEN = 2000


def _sanitize_tool_error(error_msg: str) -> str:
    """Strip structural framing tokens from a tool error before showing it to the model.

    See _TOOL_ERROR_ROLE_TAG_RE docstring above for rationale.
    """
    if not error_msg:
        return "[TOOL_ERROR] "
    sanitized = _TOOL_ERROR_ROLE_TAG_RE.sub("", error_msg)
    sanitized = _TOOL_ERROR_FENCE_OPEN_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_FENCE_CLOSE_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_CDATA_RE.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


# =========================================================================
# Tool argument type coercion
# =========================================================================

def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce tool call arguments to match their JSON Schema types.

    LLMs frequently return numbers as strings (``"42"`` instead of ``42``)
    and booleans as strings (``"true"`` instead of ``true``).  This compares
    each argument value against the tool's registered JSON Schema and attempts
    safe coercion when the value is a string but the schema expects a different
    type.  Original values are preserved when coercion fails.

    Handles ``"type": "integer"``, ``"type": "number"``, ``"type": "boolean"``,
    and union types (``"type": ["integer", "string"]``).

    Also wraps bare scalar values in a single-element list when the schema
    declares ``"type": "array"``.  Open-weight models (DeepSeek, Qwen, GLM)
    sometimes emit ``{"urls": "https://a.com"}`` when the tool expects
    ``{"urls": ["https://a.com"]}``; wrapping here avoids a confusing tool
    failure on what is otherwise a well-formed call.
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    # The model saw the SANITIZED schema — property keys violating provider
    # patterns (e.g. Cloudflare's ``issue_class~neq``) were renamed before
    # the request. Map any sanitized keys back to the registry's original
    # wire names before schema lookup / dispatch.
    try:
        from tools.schema_sanitizer import unrename_tool_args
        args = unrename_tool_args(schema.get("parameters"), args)
    except Exception:  # pragma: no cover — never break dispatch
        pass

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")

        # Wrap bare non-list values when the schema declares ``array``.
        # Strings still go through _coerce_value first so JSON-encoded
        # arrays (``'["a","b"]'``) get parsed and nullable ``"null"``
        # becomes ``None`` rather than ``["null"]``.
        # ``None`` itself is preserved — we don't know whether the model
        # meant "omit" or "empty list", and tools with sensible defaults
        # (e.g. read_file's normalize_read_pagination) already handle it.
        if expected == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    # _coerce_value handled it (JSON-parsed list or
                    # nullable "null" → None).
                    args[key] = coerced
                    continue
                # If the string looks like a JSON array but _coerce_value
                # failed to parse it, warn clearly instead of silently wrapping.
                if value.strip().startswith("["):
                    logger.warning(
                        "coerce_tool_args: %s.%s looks like a JSON array string "
                        "but could not be parsed — model may have emitted a "
                        "JSON-encoded string instead of a native array. "
                        "Falling back to single-element list.",
                        tool_name, key,
                    )
                args[key] = [value]
                logger.info(
                    "coerce_tool_args: wrapped bare string in list for %s.%s",
                    tool_name, key,
                )
                continue
            args[key] = [value]
            logger.info(
                "coerce_tool_args: wrapped bare %s in list for %s.%s",
                type(value).__name__, tool_name, key,
            )
            continue

        if not isinstance(value, str):
            # Recurse into already-native containers so JSON-encoded
            # *elements* (array items) and *sub-fields* (nested object
            # properties) get normalized too — e.g. ``todos: ['{"id":...}']``
            # or ``tasks: [{"goal": "..."}]`` where an element was emitted as
            # a JSON string. The top-level coercion above only repairs the
            # outermost value.
            if expected == "array" and isinstance(value, (list, tuple)):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            elif expected == "object" and isinstance(value, dict):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced
            # If we just JSON-parsed a string into a container, recurse so
            # nested JSON-encoded elements/fields get normalized as well.
            if isinstance(coerced, (list, tuple, dict)):
                args[key] = _normalize_json_strings_for_schema(coerced, prop_schema)

    return args


def _schema_accepts_kind(schema: Any, kind: str) -> bool:
    """Return True when *schema* permits a value of JSON type *kind*.

    Looks at ``type`` (string or list) and recurses through
    ``anyOf``/``oneOf``/``allOf`` branches — matching the JSON-Schema shapes
    open-weight models emit against. ``kind`` is ``"array"`` or ``"object"``.
    """
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and any(
            _schema_accepts_kind(b, kind) for b in branches
        ):
            return True
    return False


def _normalize_json_strings_for_schema(value: Any, schema: Any) -> Any:
    """Recursively parse JSON-encoded string values that a schema expects to
    be arrays or objects, including nested array items and object properties.

    Open-weight models (DeepSeek, Qwen, GLM, and others) sometimes emit a
    structured field — or an *element* of a structured field — as a
    JSON-encoded string instead of a native value. The top-level
    :func:`coerce_tool_args` pass repairs the outermost value; this helper
    walks the rest of the tree so cases like::

        {"todos": ["{\\"id\\": \\"1\\", \\"content\\": \\"x\\"}"]}

    (a list whose elements are JSON strings) and nested object sub-fields are
    repaired too. Parsing is schema-guided: a string is only parsed when the
    matching schema position actually expects an array or object, so
    legitimate JSON-looking string fields (``type: string``) are preserved.

    Ported from cline/cline#11803, adapted to hermes-agent's coercion layer.
    Returns the original value object when nothing changed (identity preserved
    so callers can cheaply detect no-ops).
    """
    if not isinstance(schema, dict):
        return value

    # Parse a JSON-encoded string into the container the schema expects.
    if isinstance(value, str):
        trimmed = value.strip()
        expects_array = _schema_accepts_kind(schema, "array")
        expects_object = _schema_accepts_kind(schema, "object")
        if (expects_array and trimmed.startswith("[")) or (
            expects_object and trimmed.startswith("{")
        ):
            try:
                parsed = json.loads(trimmed)
            except (ValueError, TypeError):
                return value
            if isinstance(parsed, list) and expects_array:
                value = parsed
            elif isinstance(parsed, dict) and expects_object:
                value = parsed
            else:
                return value
        else:
            return value

    # Recurse into list items using the ``items`` schema.
    if isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        changed = False
        out = []
        for item in value:
            nxt = _normalize_json_strings_for_schema(item, items_schema)
            changed = changed or (nxt is not item)
            out.append(nxt)
        return out if changed else value

    # Recurse into object properties using each property's schema.
    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        changed = False
        out = dict(value)
        for k, prop_schema in props.items():
            if k not in value or not isinstance(prop_schema, dict):
                continue
            nxt = _normalize_json_strings_for_schema(value[k], prop_schema)
            if nxt is not value[k]:
                out[k] = nxt
                changed = True
        return out if changed else value

    return value


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Attempt to coerce a string *value* to *expected_type*.

    Returns the original string when coercion is not applicable or fails.
    """
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        # Union type — try each in order, return first successful coercion
        for t in expected_type:
            result = _coerce_value(value, t, schema=schema)
            if result is not value:
                return result
        return value

    if expected_type in {"integer", "number"}:
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type == "array":
        return _coerce_json(value, list)
    if expected_type == "object":
        return _coerce_json(value, dict)
    if expected_type == "null" and value.strip().lower() == "null":
        return None
    return value


def _schema_allows_null(schema: dict | None) -> bool:
    """Return True when a JSON Schema fragment explicitly permits null."""
    if not isinstance(schema, dict):
        return False

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    if schema.get("nullable") is True:
        return True

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict) and variant.get("type") == "null":
                return True

    return False


def _coerce_json(value: str, expected_python_type: type):
    """Parse *value* as JSON when the schema expects an array or object.

    Handles model output drift where a complex oneOf/discriminated-union schema
    causes the LLM to emit the array/object as a JSON string instead of a native
    structure.  Returns the original string if parsing fails or yields the wrong
    Python type.
    """
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "coerce_tool_args: failed to parse string as JSON for expected type %s: %s",
            expected_python_type.__name__,
            exc,
        )
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug(
            "coerce_tool_args: coerced string to %s via json.loads",
            expected_python_type.__name__,
        )
        return parsed
    logger.warning(
        "coerce_tool_args: JSON-parsed value is %s, expected %s — skipping coercion",
        type(parsed).__name__,
        expected_python_type.__name__,
    )
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Try to parse *value* as a number.  Returns original string on failure."""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    # Guard against inf/nan — not JSON-serializable, keep original string
    if f != f or f == float("inf") or f == float("-inf"):
        return value
    # If it looks like an integer (no fractional part), return int
    if f == int(f):
        return int(f)
    if integer_only:
        # Schema wants an integer but value has decimals — keep as string
        return value
    return f


def _coerce_boolean(value: str):
    """Try to parse *value* as a boolean.  Returns original string on failure."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def _tool_result_observer_fields(result: Any) -> tuple[str, Optional[str], Optional[str]]:
    try:
        parsed_result = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed_result, dict) and parsed_result.get("error"):
            return "error", "tool_error", str(parsed_result.get("error"))
    except Exception:
        pass
    return "ok", None, None


async def _emit_post_tool_call_hook(
    *,
    function_name: str,
    function_args: Dict[str, Any],
    result: Any,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    duration_ms: int = 0,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit the ``post_tool_call`` observer hook.

    No-ops cheaply when no plugin has registered for ``post_tool_call`` —
    the ``has_hook`` gate skips both the result-field derivation and the
    payload dispatch so the no-listener path costs one dict lookup.  When
    ``status`` is not supplied, the ok/error fields are derived from the
    result *after* the gate (parsing the result is only worth it when a
    listener will actually consume it).
    """
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not has_hook("post_tool_call"):
            return
        if status is None:
            status, error_type, error_message = _tool_result_observer_fields(result)
        await invoke_hook(
            "post_tool_call",
            tool_name=function_name,
            args=function_args,
            result=result,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception as _hook_err:
        logger.debug("post_tool_call hook error: %s", _hook_err)


async def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    *,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False,
    skip_tool_request_middleware: bool = False,
    skip_tool_execution_middleware: bool = False,
    tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    **handler_context: Any,
) -> str | dict:
    """Dispatch one tool through the native async plugin lifecycle.

    Request middleware, pre-tool policy, execution middleware, observer hooks,
    and result transforms keep their established order. Callers that already
    own those phases use the existing ``skip_*`` arguments so each phase fires
    exactly once.
    """
    function_args = coerce_tool_args(function_name, function_args)
    if not isinstance(function_args, dict):
        function_args = {}
    original_args = dict(function_args)
    middleware_trace = list(tool_request_middleware_trace or [])

    if not skip_tool_request_middleware:
        from hermes_cli.middleware import apply_tool_request_middleware

        request_result = await apply_tool_request_middleware(
            function_name,
            function_args,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
        )
        if isinstance(request_result.payload, dict):
            function_args = request_result.payload
        if isinstance(request_result.original_payload, dict):
            original_args = request_result.original_payload
        middleware_trace = list(request_result.trace)

    if not skip_pre_tool_call_hook:
        from hermes_cli.plugins import resolve_pre_tool_block

        block_message = await resolve_pre_tool_block(
            function_name,
            function_args,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            middleware_trace=middleware_trace,
        )
        if block_message is not None:
            result = tool_error(block_message)
            await _emit_post_tool_call_hook(
                function_name=function_name,
                function_args=function_args,
                result=result,
                task_id=task_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                status="blocked",
                error_type="plugin_block",
                error_message=block_message,
                middleware_trace=middleware_trace,
            )
            return result

    if function_name not in _READ_SEARCH_TOOLS:
        try:
            from tools.file_tools import notify_other_tool_call

            notify_other_tool_call(task_id or "default")
        except Exception:
            logger.debug("file-read loop reset failed", exc_info=True)

    async def dispatch(next_args: Dict[str, Any]) -> str | dict:
        return await registry.dispatch(
            function_name,
            next_args,
            task_id=task_id,
            tool_call_id=tool_call_id,
            session_id=session_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            user_task=user_task,
            enabled_tools=enabled_tools,
            **handler_context,
        )

    started = time.monotonic()
    if skip_tool_execution_middleware:
        result = await dispatch(function_args)
    else:
        from hermes_cli.middleware import run_tool_execution_middleware

        result = await run_tool_execution_middleware(
            function_name,
            function_args,
            dispatch,
            original_args=original_args,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
        )
    duration_ms = int((time.monotonic() - started) * 1000)

    await _emit_post_tool_call_hook(
        function_name=function_name,
        function_args=function_args,
        result=result,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        duration_ms=duration_ms,
        middleware_trace=middleware_trace,
    )

    from hermes_cli.lifecycle import has_hook, invoke_hook

    if has_hook("transform_tool_result"):
        status, error_type, error_message = _tool_result_observer_fields(result)
        hook_results = await invoke_hook(
            "transform_tool_result",
            tool_name=function_name,
            args=function_args,
            result=result,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
        )
        for hook_result in hook_results:
            if isinstance(hook_result, str):
                result = hook_result
                break
    return result


# =============================================================================
# Backward-compat wrapper functions
# =============================================================================

def get_all_tool_names() -> List[str]:
    """Return all registered tool names."""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """Return the toolset a tool belongs to."""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Return toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """Return {toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """Return (available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
