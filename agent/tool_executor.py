"""Native async tool-call scheduling for the agent conversation loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles.os

from agent.display import (
    build_tool_preview as _build_tool_preview,
    redact_tool_args_for_display as _redact_tool_args_for_display,
    _detect_tool_failure,
)
from agent.tool_dispatch_helpers import (
    _is_destructive_command,
    _is_multimodal_tool_result,
    _append_subdir_hint_to_multimodal,
    _plan_tool_batch_segments,
    make_tool_result_message,
)
from tools.terminal_tool import (
    get_active_env,
)
from tools.tool_result_storage import (
    maybe_persist_tool_result,
    enforce_turn_budget,
)
from tools.budget_config import BudgetConfig, DEFAULT_BUDGET, budget_for_context_window

logger = logging.getLogger(__name__)

_MAX_TOOL_WORKERS = 8
_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0


def _resolve_concurrent_tool_timeout() -> float | None:
    raw = os.getenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "").strip()
    if not raw:
        return _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "invalid HERMES_CONCURRENT_TOOL_TIMEOUT_S=%r; using %.0fs",
            raw,
            _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S,
        )
        return _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
    if value <= 0:
        return None
    return value


async def _ensure_file_checkpoint(
    agent,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
) -> None:
    """Checkpoint the same workspace path that the file tool will mutate."""
    file_path = function_args.get("path", "")
    if not file_path:
        return

    # File tools resolve relative paths against the task's live/session cwd,
    # which can differ from the Hermes process cwd (notably in Docker).  Resolve
    # through that same path pipeline before asking the checkpoint manager to
    # discover the project root.
    from tools.file_tools import _resolve_path_for_task

    resolved_path = await _resolve_path_for_task(
        file_path,
        effective_task_id or "default",
    )
    work_dir = await agent._checkpoint_mgr.get_working_dir_for_path(
        str(resolved_path)
    )
    await agent._checkpoint_mgr.ensure_checkpoint(
        work_dir,
        f"before {function_name}",
    )


def _budget_for_agent(agent) -> BudgetConfig:
    """Resolve a tool-result BudgetConfig scaled to the agent's context window.

    Large-context models keep the historical 100K/200K char defaults; small
    models (e.g. a 65K-token local model switched into mid-session) get a budget
    proportional to their window so a single large tool result can't push the
    request past the model's limit (#23767). Falls back to the default budget
    when the context length isn't resolvable.
    """
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        return budget_for_context_window(int(ctx)) if ctx else DEFAULT_BUDGET
    except Exception:
        return DEFAULT_BUDGET


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, str | None]:
    """Parse model-emitted arguments without repairing or coercing them."""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        return arguments, None
    return {}, json.dumps(
        {
            "error": "Invalid tool arguments",
            "message": (
                "Tool arguments must be a valid JSON object; tool was not executed."
            ),
        },
        ensure_ascii=False,
    )










async def _emit_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    result: Any,
    effective_task_id: str,
    tool_call_id: str,
    duration_ms: int = 0,
    status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
) -> None:
    try:
        from model_tools import _emit_post_tool_call_hook
        await _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception:
        pass


def _cancelled_tool_result(reason: str = "user interrupt") -> str:
    return json.dumps(
        {
            "error": f"Tool execution cancelled by {reason}",
            "status": "cancelled",
        },
        ensure_ascii=False,
    )


async def _tool_search_scoped_names(agent) -> frozenset[str]:
    """Return the deferrable tool names available to this agent session."""
    try:
        import model_tools
        from tools import tool_search
        from tools.registry import registry
    except Exception:
        return frozenset()

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    cache_key = (
        getattr(registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    try:
        scoped_defs = await model_tools.get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = tool_search.scoped_deferrable_names(scoped_defs or [])
    except Exception:
        names = frozenset()
    try:
        agent._tool_search_scope_cache = (cache_key, names)
    except Exception:
        pass
    return names


async def _emit_cancelled_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    start_time: float,
    reason: str = "user interrupt",
    error_type: str = "keyboard_interrupt",
    middleware_trace: list[dict[str, Any]] | None = None,
) -> str:
    result = _cancelled_tool_result(reason)
    await _emit_terminal_post_tool_call(
        agent,
        function_name=function_name,
        function_args=function_args,
        result=result,
        effective_task_id=effective_task_id,
        tool_call_id=tool_call_id,
        duration_ms=int((time.time() - start_time) * 1000),
        status="cancelled",
        error_type=error_type,
        error_message=f"Tool execution cancelled by {reason}",
        middleware_trace=list(middleware_trace or []),
    )
    return result




@dataclass
class _ManagedToolResult:
    result: Any
    args: dict[str, Any]
    middleware_trace: list[dict[str, Any]]
    blocked: bool


class _ConcurrentToolAuthorizationGate:
    """Serialize policy prompts and exclude their queue from batch deadlines."""

    def __init__(self) -> None:
        self._serialization_lock = asyncio.Lock()
        self._pending = 0
        self._window_started: float | None = None
        self._excluded_seconds = 0.0

    async def run(self, callback):
        now = time.monotonic()
        if self._pending == 0:
            self._window_started = now
        self._pending += 1
        try:
            async with self._serialization_lock:
                return await callback()
        finally:
            now = time.monotonic()
            self._pending -= 1
            if self._pending == 0:
                if self._window_started is not None:
                    self._excluded_seconds += max(
                        0.0, now - self._window_started
                    )
                self._window_started = None

    def excluded_seconds(self) -> float:
        excluded = self._excluded_seconds
        if self._window_started is not None:
            excluded += max(0.0, time.monotonic() - self._window_started)
        return excluded


class _OrderedToolStartGate:
    """Run parallel tool preflight callbacks in model emission order."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._next_order = 0

    async def advance(self, order: int, callback=None) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: order == self._next_order)
            try:
                if callback is not None:
                    await callback()
            finally:
                self._next_order += 1
                self._condition.notify_all()








async def _run_agent_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
    scope_block: str | None = None,
    display_index: int | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
    begin_execution=None,
    start_order: int = 0,
    authorization_gate: _ConcurrentToolAuthorizationGate | None = None,
    emit_runtime_post_hook: bool = False,
) -> _ManagedToolResult:
    """Execute one native tool without bypassing the established policy path.

    The old threaded executor owned policy, guardrail, display, and
    post-hook handling.  Keeping those concerns here ensures the coroutine
    scheduler preserves agent semantics instead of treating registry dispatch
    as the whole tool lifecycle.
    """
    from hermes_cli.middleware import (
        apply_tool_request_middleware,
        run_tool_execution_middleware,
    )

    trace = middleware_trace if middleware_trace is not None else []
    request_result = await apply_tool_request_middleware(
        function_name,
        function_args,
        task_id=effective_task_id or "",
        session_id=getattr(agent, "session_id", "") or "",
        tool_call_id=tool_call_id or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
    )
    request_args = (
        request_result.payload
        if isinstance(request_result.payload, dict)
        else function_args
    )
    trace.extend(request_result.trace)
    state = {
        "args": request_args,
        "blocked": False,
        "dispatched": False,
        "started": None,
        "post_hook_emitted": False,
    }
    dispatch_lock = asyncio.Lock()

    async def _authorized_dispatch(next_args: dict[str, Any]) -> Any:
        async with dispatch_lock:
            if state["dispatched"]:
                raise RuntimeError(
                    "Hermes tool execution callback invoked more than once"
                )
            state["dispatched"] = True
            state["args"] = next_args

        block_message = scope_block
        block_error_type = "tool_scope_block"
        if block_message is None:
            block_error_type = "plugin_block"
            from hermes_cli.plugins import resolve_pre_tool_block

            async def _resolve_pre_tool_block():
                return await resolve_pre_tool_block(
                    function_name,
                    next_args,
                    task_id=effective_task_id or "",
                    session_id=getattr(agent, "session_id", "") or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=getattr(agent, "_current_turn_id", "") or "",
                    api_request_id=getattr(
                        agent, "_current_api_request_id", ""
                    )
                    or "",
                    middleware_trace=list(trace),
                )

            try:
                block_message = (
                    await authorization_gate.run(_resolve_pre_tool_block)
                    if authorization_gate is not None
                    else await _resolve_pre_tool_block()
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                block_message = None

        guardrail_decision = None
        if block_message is None:
            guardrail_decision = agent._tool_guardrails.before_call(
                function_name, next_args
            )
            if guardrail_decision.allows_execution:
                guardrail_decision = None

        if block_message is not None or guardrail_decision is not None:
            if begin_execution is not None:
                await begin_execution(start_order)
            state["blocked"] = True
            if block_message is not None:
                result = json.dumps({"error": block_message}, ensure_ascii=False)
                error_type = block_error_type
                error_message = block_message
            else:
                result = agent._guardrail_block_result(guardrail_decision)
                error_type = "guardrail_block"
                error_message = (
                    getattr(guardrail_decision, "message", None)
                    or "Tool blocked by guardrail policy"
                )
            await _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=next_args,
                result=result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="blocked",
                error_type=error_type,
                error_message=error_message,
                middleware_trace=list(trace),
            )
            return result

        if function_name == "memory":
            agent._turns_since_memory = 0
        elif function_name == "skill_manage":
            agent._iters_since_skill = 0

        async def _begin() -> None:
            await _begin_tool_execution(
                agent,
                function_name=function_name,
                function_args=next_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                display_index=display_index,
            )

        if begin_execution is None:
            await _begin()
        else:
            await begin_execution(start_order, _begin)
        started = time.monotonic()
        state["started"] = started
        try:
            result = await execute(next_args, trace)
        except asyncio.CancelledError as exc:
            if not exc.args or exc.args[0] != "tool timeout":
                await _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=next_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    start_time=started,
                    middleware_trace=list(trace),
                )
            raise
        except Exception as exc:
            result = json.dumps(
                {
                    "error": f"Tool execution failed: {type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
            state["post_hook_emitted"] = True
            await _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=next_args,
                result=result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                duration_ms=int((time.monotonic() - started) * 1000),
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                middleware_trace=list(trace),
            )
            return result

        return result

    result = await run_tool_execution_middleware(
        function_name,
        request_args,
        _authorized_dispatch,
        original_args=function_args,
        task_id=effective_task_id or "",
        session_id=getattr(agent, "session_id", "") or "",
        tool_call_id=tool_call_id or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
    )
    if (
        emit_runtime_post_hook
        and state["dispatched"]
        and not state["blocked"]
        and not state["post_hook_emitted"]
    ):
        started = state["started"]
        duration_ms = (
            int((time.monotonic() - started) * 1000)
            if isinstance(started, float)
            else 0
        )
        await _emit_terminal_post_tool_call(
            agent,
            function_name=function_name,
            function_args=state["args"],
            result=result,
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            duration_ms=duration_ms,
            middleware_trace=list(trace),
        )
    return _ManagedToolResult(
        result=result,
        args=state["args"],
        middleware_trace=trace,
        blocked=bool(state["blocked"]),
    )


async def _begin_tool_execution(
    agent,
    *,
    function_name: str,
    function_args: dict[str, Any],
    effective_task_id: str,
    tool_call_id: str,
    display_index: int | None,
) -> None:
    """Run user-visible preflight on final tool arguments."""
    if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
        display_args = (
            _redact_tool_args_for_display(function_name, function_args) or function_args
        )
        args_str = json.dumps(display_args, ensure_ascii=False)
        prefix = f"Tool {display_index}" if display_index is not None else "Tool"
        if agent.verbose_logging:
            print(f"  📞 {prefix}: {function_name}({list(display_args.keys())})")
            print(
                agent._wrap_verbose(
                    "Args: ", json.dumps(display_args, indent=2, ensure_ascii=False)
                )
            )
        else:
            args_preview = (
                args_str[: agent.log_prefix_chars] + "..."
                if len(args_str) > agent.log_prefix_chars
                else args_str
            )
            print(
                f"  📞 {prefix}: {function_name}({list(function_args.keys())}) - "
                f"{args_preview}"
            )

    agent._current_tool = function_name
    touch_activity = getattr(agent, "_touch_activity", None)
    if callable(touch_activity):
        touch_activity(f"executing tool: {function_name}")
    if agent.tool_progress_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            preview = _build_tool_preview(function_name, display_args)
            agent.tool_progress_callback(
                "tool.started", function_name, preview, display_args
            )
        except Exception as callback_error:
            logging.debug("Tool progress callback error: %s", callback_error)

    if agent.tool_start_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            agent.tool_start_callback(
                tool_call_id, function_name, display_args
            )
        except Exception as callback_error:
            logging.debug("Tool start callback error: %s", callback_error)

    if function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
        try:
            await _ensure_file_checkpoint(
                agent,
                function_name,
                function_args,
                effective_task_id,
            )
        except Exception:
            logger.debug("File checkpoint failed", exc_info=True)

    if function_name == "terminal" and agent._checkpoint_mgr.enabled:
        try:
            command = function_args.get("command", "")
            if _is_destructive_command(command):
                from tools.terminal_tool import _get_or_create_environment

                environment = await _get_or_create_environment(effective_task_id)
                cwd = function_args.get("workdir") or environment.cwd
                await agent._checkpoint_mgr.ensure_checkpoint(
                    cwd,
                    f"before terminal: {command[:60]}",
                )
        except Exception:
            logger.debug("Terminal checkpoint failed", exc_info=True)


def _emit_tool_completion(
    agent,
    *,
    tool_call_id: str,
    function_name: str,
    function_args: dict[str, Any],
    result: Any,
    failed: bool,
    duration: float,
    risk_metadata: dict[str, Any] | None = None,
) -> None:
    """Notify observers only after the corresponding result is durable."""
    if agent.tool_progress_callback:
        try:
            agent.tool_progress_callback(
                "tool.completed",
                function_name,
                None,
                None,
                duration=duration,
                is_error=failed,
                result=result,
            )
        except Exception:
            logger.debug(
                "Tool progress callback error for %s", function_name, exc_info=True
            )
    if agent.tool_complete_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            agent.tool_complete_callback(
                tool_call_id,
                function_name,
                display_args,
                result,
            )
        except Exception:
            logger.debug(
                "Tool complete callback error for %s", function_name, exc_info=True
            )
    if (
        risk_metadata is not None
        and risk_metadata.get("risk") != "low"
        and agent.tool_progress_callback
    ):
        try:
            agent.tool_progress_callback(
                "tool.output_risk",
                function_name,
                None,
                None,
                tool_call_id=tool_call_id,
                risk_metadata=risk_metadata,
            )
        except Exception:
            logger.debug(
                "Tool output-risk callback error for %s", function_name,
                exc_info=True,
            )

async def _execute_tool_calls_native(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    segments=None,
    *,
    finalize: bool = True,
) -> None:
    """Execute a batch through native async handlers only.

    The core never dispatches a mixed batch through the old synchronous
    executor: doing that would hide an entire conversation turn behind a
    blocking compatibility bridge. Results still append in model emission
    order.
    """
    from tools.registry import registry
    from model_tools import _TOOL_HANDLER_CONTEXT, handle_function_call

    tool_calls = list(getattr(assistant_message, "tool_calls", None) or [])
    if not tool_calls:
        return
    tool_budget = _budget_for_agent(agent)
    active_env = get_active_env(effective_task_id)

    tool_call_indexes = {id(tool_call): index for index, tool_call in enumerate(tool_calls)}

    if getattr(agent, "_interrupt_requested", False):
        cancelled = [
            make_tool_result_message(
                getattr(tc.function, "name", ""),
                f"[Tool execution cancelled — {getattr(tc.function, 'name', '')} "
                "was skipped due to user interrupt]",
                getattr(tc, "id", "") or "",
                effect_disposition="none",
            )
            for tc in tool_calls
        ]
        messages.extend(cancelled)
        flush = getattr(agent, "_flush_messages_to_session_db", None)
        if callable(flush):
            try:
                if await flush(messages) is False:
                    agent._incremental_persistence_failed = True
            except Exception as exc:
                agent._incremental_persistence_failed = True
                logger.warning("Incremental cancelled tool persistence failed: %s", exc)
        return

    async def _one(
        index,
        tc,
        result_sink,
        completion_sink,
        *,
        begin_execution=None,
        authorization_gate=None,
        runtime_sink=None,
    ):
        tool_started = time.monotonic()
        middleware_trace: list[dict[str, Any]] = []
        name = getattr(tc.function, "name", "")
        args, malformed_args_result = _parse_tool_arguments(
            getattr(tc.function, "arguments", "")
        )
        scope_block = None
        if malformed_args_result is None:
            try:
                from tools import tool_search

                if name == tool_search.TOOL_CALL_NAME:
                    underlying, underlying_args, error = (
                        tool_search.resolve_underlying_call(args)
                    )
                    if not error and underlying:
                        if underlying in await _tool_search_scoped_names(agent):
                            scope_block = tool_search.validate_deferred_call_args(
                                underlying, underlying_args
                            )
                            if scope_block is None:
                                name = underlying
                                args = underlying_args
                        else:
                            scope_block = (
                                f"'{underlying}' is not available in this session. "
                                "Use tool_search to find tools you can call."
                            )
            except Exception:
                pass
        if runtime_sink is not None:
            runtime_sink[index] = (name, args, middleware_trace)
        if malformed_args_result is None:
            memory_manager = getattr(agent, "_memory_manager", None)
            runtime_owns_post_hook = bool(
                name in (getattr(agent, "_context_engine_tool_names", None) or set())
                or (memory_manager and memory_manager.has_tool(name))
            )

            async def _dispatch(next_args, middleware_trace):
                if name in (
                    getattr(agent, "_context_engine_tool_names", None) or set()
                ):
                    return await agent.context_compressor.handle_tool_call(
                        name,
                        next_args,
                        messages=messages,
                    )
                if memory_manager and memory_manager.has_tool(name):
                    return await memory_manager.handle_tool_call(name, next_args)
                dispatch_kwargs = {
                    "tool_call_id": getattr(tc, "id", "") or "",
                    "session_id": getattr(agent, "session_id", "") or "",
                    "turn_id": getattr(agent, "_current_turn_id", "") or "",
                    "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
                    "user_task": getattr(agent, "_current_user_task", None),
                    "enabled_tools": (
                        list(getattr(agent, "valid_tool_names", None) or []) or None
                    ),
                    "enabled_toolsets": getattr(agent, "enabled_toolsets", None),
                    "disabled_toolsets": getattr(agent, "disabled_toolsets", None),
                    "skip_pre_tool_call_hook": True,
                    "skip_tool_request_middleware": True,
                    "skip_tool_execution_middleware": True,
                    "tool_request_middleware_trace": list(middleware_trace),
                }
                entry = registry.get_entry(name)
                if str(getattr(entry, "toolset", "")).startswith("mcp-"):
                    dispatch_kwargs["elicitation_callback"] = getattr(
                        agent, "clarify_callback", None
                    )
                if name == "memory":
                    dispatch_kwargs["store"] = getattr(agent, "_memory_store", None)
                elif name == "todo":
                    dispatch_kwargs["store"] = getattr(agent, "_todo_store", None)
                elif name == "session_search":
                    get_recall_db = getattr(agent, "_get_session_db_for_recall", None)
                    dispatch_kwargs["db"] = (
                        await get_recall_db()
                        if callable(get_recall_db)
                        else getattr(agent, "_session_db", None)
                    )
                    dispatch_kwargs["current_session_id"] = getattr(
                        agent, "session_id", None
                    )
                elif name == "clarify":
                    dispatch_kwargs["callback"] = getattr(agent, "clarify_callback", None)
                elif name == "read_terminal":
                    dispatch_kwargs["callback"] = getattr(
                        agent, "read_terminal_callback", None
                    )
                handler_context = {
                    key: dispatch_kwargs.pop(key)
                    for key in (
                        "elicitation_callback",
                        "store",
                        "db",
                        "current_session_id",
                        "callback",
                    )
                    if key in dispatch_kwargs
                }
                if name == "delegate_task":
                    # Upstream dispatch passes the live AIAgent only into
                    # delegate_task so children inherit provider, tool,
                    # session, and lifecycle state. Keep it context-local
                    # rather than widening handle_function_call's public
                    # signature or exposing the agent to unrelated tools.
                    handler_context["parent_agent"] = agent
                context_token = _TOOL_HANDLER_CONTEXT.set(handler_context)
                try:
                    return await handle_function_call(
                        name, next_args, effective_task_id, **dispatch_kwargs
                    )
                finally:
                    _TOOL_HANDLER_CONTEXT.reset(context_token)

            managed = await _run_agent_tool_execution_middleware(
                agent,
                function_name=name,
                function_args=args,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tc, "id", "") or "",
                display_index=index + 1,
                execute=_dispatch,
                scope_block=scope_block,
                middleware_trace=middleware_trace,
                begin_execution=begin_execution,
                start_order=index,
                authorization_gate=authorization_gate,
                emit_runtime_post_hook=runtime_owns_post_hook,
            )
            result = managed.result
            blocked = managed.blocked
            failed, _ = _detect_tool_failure(name, result)
            duration = time.monotonic() - tool_started
            if not managed.blocked:
                memory_manager = getattr(agent, "_memory_manager", None)
                if name == "memory" and memory_manager:
                    await memory_manager.notify_memory_tool_write(
                        result,
                        managed.args,
                        build_metadata=lambda: agent._build_memory_write_metadata(
                            task_id=effective_task_id,
                            tool_call_id=getattr(tc, "id", None),
                        ),
                    )
                if isinstance(result, str):
                    result = agent._append_guardrail_observation(
                        name,
                        managed.args,
                        result,
                        failed=failed,
                    )
                try:
                    agent._record_file_mutation_result(
                        name, managed.args, result, failed,
                    )
                except Exception:
                    logger.debug(
                        "File-mutation verifier record failed for %s", name,
                        exc_info=True,
                    )

            agent._current_tool = None
            status_suffix = " (error)" if failed else ""
            touch_activity = getattr(agent, "_touch_activity", None)
            if callable(touch_activity):
                touch_activity(
                    f"tool completed: {name} ({duration:.1f}s){status_suffix}"
                )
            display_result = result
            if isinstance(result, str) and not _is_multimodal_tool_result(result):
                result = await maybe_persist_tool_result(
                    content=result,
                    tool_name=name,
                    tool_use_id=getattr(tc, "id", "") or "tool_result",
                    env=active_env,
                    config=tool_budget,
                )

            subdirectory_hints = getattr(agent, "_subdirectory_hints", None)
            if subdirectory_hints is not None:
                check_hints = getattr(subdirectory_hints, "check_tool_call", None)
                hints = (
                    await check_hints(name, managed.args)
                    if callable(check_hints)
                    else ""
                )
                if hints:
                    if _is_multimodal_tool_result(result):
                        _append_subdir_hint_to_multimodal(result, hints)
                    elif isinstance(result, str):
                        result += hints

            format_for_model = getattr(agent, "_tool_result_content_for_active_model", None)
            content_for_model = (
                await format_for_model(name, result)
                if callable(format_for_model)
                else result
            )
            tool_message = make_tool_result_message(
                name,
                content_for_model,
                getattr(tc, "id", "") or "",
                effect_disposition="none" if blocked else None,
            )
            result_sink[index] = tool_message
            if not blocked:
                completion_sink[index] = (
                    getattr(tc, "id", "") or "",
                    name,
                    managed.args,
                    display_result,
                    failed,
                    duration,
                    tool_message.get("_tool_output_risk"),
                )
        else:
            if begin_execution is not None:
                await begin_execution(index)
            result_sink[index] = make_tool_result_message(
                name, malformed_args_result, getattr(tc, "id", "") or ""
            )

    async def _store_timeout_result(
        index,
        tc,
        name,
        args,
        timeout_s,
        middleware_trace,
        result_sink,
        completion_sink,
    ) -> None:
        result = (
            f"Error executing tool '{name}': timed out after {timeout_s:.1f}s"
        )
        await _emit_terminal_post_tool_call(
            agent,
            function_name=name,
            function_args=args,
            result=result,
            effective_task_id=effective_task_id,
            tool_call_id=getattr(tc, "id", "") or "",
            status="timeout",
            error_type="tool_timeout",
            error_message=result,
            middleware_trace=middleware_trace,
        )
        agent._current_tool = None
        touch_activity = getattr(agent, "_touch_activity", None)
        if callable(touch_activity):
            touch_activity(
                f"tool completed: {name} ({timeout_s:.1f}s) (error)"
            )
        display_result = result
        result = await maybe_persist_tool_result(
            content=result,
            tool_name=name,
            tool_use_id=getattr(tc, "id", "") or "tool_result",
            env=active_env,
            config=tool_budget,
        )
        subdirectory_hints = getattr(agent, "_subdirectory_hints", None)
        if subdirectory_hints is not None:
            check_hints = getattr(subdirectory_hints, "check_tool_call", None)
            hints = await check_hints(name, args) if callable(check_hints) else ""
            if hints:
                result += hints
        format_for_model = getattr(
            agent, "_tool_result_content_for_active_model", None
        )
        content_for_model = (
            await format_for_model(name, result)
            if callable(format_for_model)
            else result
        )
        tool_message = make_tool_result_message(
            name,
            content_for_model,
            getattr(tc, "id", "") or "",
            effect_disposition="unknown",
        )
        result_sink[index] = tool_message
        completion_sink[index] = (
            getattr(tc, "id", "") or "",
            name,
            args,
            display_result,
            True,
            float(timeout_s),
            tool_message.get("_tool_output_risk"),
        )

    execution_cwd = None
    if active_env is not None and getattr(active_env, "cwd", None):
        raw_execution_cwd = Path(active_env.cwd)
        execution_cwd = (
            raw_execution_cwd
            if raw_execution_cwd.is_absolute()
            else Path(await aiofiles.os.getcwd()) / raw_execution_cwd
        )
    else:
        # The planner is a synchronous, CPU-only path helper, but its
        # historical default resolves relative paths against the process cwd.
        # Resolve that cwd at the async boundary so an environment-less turn
        # never calls ``Path.cwd()`` while the event loop is running.
        execution_cwd = Path(await aiofiles.os.getcwd())

    # Preserve the established barrier semantics: a later call must never
    # cross an interactive/mutating/conflicting call, while independent
    # read-only calls in the same segment retain native concurrency.
    if segments is None:
        segments = _plan_tool_batch_segments(
            tool_calls, execution_cwd=execution_cwd
        )
    for kind, calls in segments:
        if not calls:
            continue
        if getattr(agent, "_interrupt_requested", False):
            first_call_index = tool_call_indexes[id(calls[0])]
            messages.extend(
                make_tool_result_message(
                    getattr(call.function, "name", ""),
                    "[Tool execution cancelled — call was skipped due to user interrupt]",
                    getattr(call, "id", "") or "",
                    effect_disposition="none",
                )
                for call in tool_calls[first_call_index:]
            )
            flush = getattr(agent, "_flush_messages_to_session_db", None)
            if callable(flush):
                try:
                    if await flush(messages) is False:
                        agent._incremental_persistence_failed = True
                except Exception as exc:
                    agent._incremental_persistence_failed = True
                    logger.warning(
                        "Incremental cancelled tool persistence failed: %s", exc
                    )
            return

        segment_results = [None] * len(calls)
        segment_completions = [None] * len(calls)
        if kind == "sequential":
            try:
                for index, tool_call in enumerate(calls):
                    if getattr(agent, "_interrupt_requested", False):
                        raise asyncio.CancelledError
                    await _one(
                        index,
                        tool_call,
                        segment_results,
                        segment_completions,
                    )
            except asyncio.CancelledError:
                for index, tool_call in enumerate(calls):
                    if segment_results[index] is None:
                        name = getattr(tool_call.function, "name", "")
                        segment_results[index] = make_tool_result_message(
                            name,
                            f"[Tool execution cancelled — {name} was skipped due to user interrupt]",
                            getattr(tool_call, "id", "") or "",
                            effect_disposition="unknown",
                        )
                messages.extend(segment_results)
                messages.extend(
                    make_tool_result_message(
                        getattr(remaining.function, "name", ""),
                        "[Tool execution cancelled — call was skipped due to user interrupt]",
                        getattr(remaining, "id", "") or "",
                        effect_disposition="none",
                    )
                    for remaining in tool_calls[
                        tool_call_indexes[id(calls[-1])] + 1:
                    ]
                )
                flush = getattr(agent, "_flush_messages_to_session_db", None)
                if callable(flush):
                    try:
                        if await flush(messages) is False:
                            agent._incremental_persistence_failed = True
                    except Exception as exc:
                        agent._incremental_persistence_failed = True
                        logger.warning("Incremental cancelled tool persistence failed: %s", exc)
                raise
        else:
            start_gate = _OrderedToolStartGate()
            authorization_gate = _ConcurrentToolAuthorizationGate()
            semaphore = asyncio.Semaphore(_MAX_TOOL_WORKERS)
            runtime_state = [None] * len(calls)

            async def _run_parallel(index, tool_call):
                async with semaphore:
                    await _one(
                        index,
                        tool_call,
                        segment_results,
                        segment_completions,
                        begin_execution=start_gate.advance,
                        authorization_gate=authorization_gate,
                        runtime_sink=runtime_state,
                    )

            tasks = [
                asyncio.create_task(
                    _run_parallel(index, tool_call),
                    name=f"hermes-tool:{getattr(tool_call.function, 'name', '')}",
                )
                for index, tool_call in enumerate(calls)
            ]
            timeout_s = _resolve_concurrent_tool_timeout()
            started = time.monotonic()
            pending = set(tasks)
            timed_out: set[asyncio.Task] = set()
            try:
                while pending:
                    remaining = None
                    if timeout_s is not None:
                        deadline = (
                            started
                            + timeout_s
                            + authorization_gate.excluded_seconds()
                        )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            timed_out = set(pending)
                            break
                    done, pending = await asyncio.wait(
                        pending,
                        timeout=(
                            None if remaining is None else min(5.0, remaining)
                        ),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task.cancelled():
                            continue
                        task_error = task.exception()
                        if task_error is not None:
                            for sibling in pending:
                                sibling.cancel()
                            await asyncio.gather(
                                *pending, return_exceptions=True
                            )
                            await asyncio.gather(
                                *done, return_exceptions=True
                            )
                            raise task_error
                if not timed_out:
                    await asyncio.gather(*tasks)
                else:
                    for task in timed_out:
                        task.cancel("tool timeout")
                    await asyncio.gather(*timed_out, return_exceptions=True)
                    for index, (task, tool_call) in enumerate(zip(tasks, calls)):
                        if task not in timed_out or segment_results[index] is not None:
                            continue
                        runtime = runtime_state[index]
                        if runtime is None:
                            name = getattr(tool_call.function, "name", "")
                            args, _ = _parse_tool_arguments(
                                getattr(tool_call.function, "arguments", "")
                            )
                            middleware_trace = []
                        else:
                            name, args, middleware_trace = runtime
                        await _store_timeout_result(
                            index,
                            tool_call,
                            name,
                            args,
                            float(timeout_s),
                            middleware_trace,
                            segment_results,
                            segment_completions,
                        )
            except asyncio.CancelledError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for index, tool_call in enumerate(calls):
                    if segment_results[index] is None:
                        name = getattr(tool_call.function, "name", "")
                        segment_results[index] = make_tool_result_message(
                            name,
                            f"[Tool execution cancelled — {name} was cancelled before a result was available]",
                            getattr(tool_call, "id", "") or "",
                            effect_disposition="unknown",
                        )
                messages.extend(segment_results)
                messages.extend(
                    make_tool_result_message(
                        getattr(remaining.function, "name", ""),
                        "[Tool execution cancelled — call was skipped due to user interrupt]",
                        getattr(remaining, "id", "") or "",
                        effect_disposition="none",
                    )
                    for remaining in tool_calls[
                        tool_call_indexes[id(calls[-1])] + 1:
                    ]
                )
                flush = getattr(agent, "_flush_messages_to_session_db", None)
                if callable(flush):
                    try:
                        if await flush(messages) is False:
                            agent._incremental_persistence_failed = True
                    except Exception as exc:
                        agent._incremental_persistence_failed = True
                        logger.warning("Incremental cancelled tool persistence failed: %s", exc)
                raise

        messages.extend(segment_results)
        flush = getattr(agent, "_flush_messages_to_session_db", None)
        if callable(flush):
            try:
                if await flush(messages) is False:
                    agent._incremental_persistence_failed = True
                    return
            except Exception as exc:
                agent._incremental_persistence_failed = True
                logger.warning("Incremental tool result persistence failed: %s", exc)
                return

        for completion in segment_completions:
            if completion is None:
                continue
            (
                tool_call_id,
                name,
                args,
                result,
                failed,
                duration,
                risk_metadata,
            ) = completion
            _emit_tool_completion(
                agent,
                tool_call_id=tool_call_id,
                function_name=name,
                function_args=args,
                result=result,
                failed=failed,
                duration=duration,
                risk_metadata=risk_metadata,
            )

    if finalize:
        await enforce_turn_budget(
            messages[-len(tool_calls):],
            env=active_env,
            config=tool_budget,
        )
        agent._apply_pending_steer_to_tool_results(messages, len(tool_calls))


async def execute_tool_calls_concurrent(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    *,
    finalize: bool = True,
) -> None:
    """Execute one parallel-safe tool-call segment in emission order."""
    tool_calls = list(getattr(assistant_message, "tool_calls", None) or [])
    await _execute_tool_calls_native(
        agent,
        assistant_message,
        messages,
        effective_task_id,
        api_call_count,
        segments=[("parallel", tool_calls)],
        finalize=finalize,
    )


async def execute_tool_calls_sequential(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    *,
    finalize: bool = True,
) -> None:
    """Execute one sequential tool-call segment in emission order."""
    tool_calls = list(getattr(assistant_message, "tool_calls", None) or [])
    await _execute_tool_calls_native(
        agent,
        assistant_message,
        messages,
        effective_task_id,
        api_call_count,
        segments=[("sequential", tool_calls)],
        finalize=finalize,
    )


async def execute_tool_calls_segmented(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    segments=None,
) -> None:
    """Execute a mixed batch as ordered parallel and sequential segments."""
    await _execute_tool_calls_native(
        agent,
        assistant_message,
        messages,
        effective_task_id,
        api_call_count,
        segments=segments,
    )


__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
    "execute_tool_calls_segmented",
]
