"""Native async tool-call scheduling for the agent conversation loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    build_tool_label as _build_tool_label,
    get_cute_tool_message as _get_cute_tool_message_impl,
    get_tool_emoji as _get_tool_emoji,
    redact_tool_args_for_display as _redact_tool_args_for_display,
    _detect_tool_failure,
)
from agent.tool_dispatch_helpers import (
    _is_multimodal_tool_result,
    _multimodal_text_summary,
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


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
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
    middleware_trace: Optional[list[dict[str, Any]]] = None,
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
    middleware_trace: Optional[list[dict[str, Any]]] = None,
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

    trace: list[dict[str, Any]] = []
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
    state = {"args": request_args, "blocked": False, "dispatched": False}
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

            block_message = await resolve_pre_tool_block(
                function_name,
                next_args,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "")
                or "",
                middleware_trace=list(trace),
            )

        guardrail_decision = None
        if block_message is None:
            guardrail_decision = agent._tool_guardrails.before_call(
                function_name, next_args
            )
            if guardrail_decision.allows_execution:
                guardrail_decision = None

        if block_message is not None or guardrail_decision is not None:
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

        _begin_tool_execution(
            agent,
            function_name=function_name,
            function_args=next_args,
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            display_index=display_index,
        )
        started = time.monotonic()
        try:
            result = await execute(next_args, trace)
        except asyncio.CancelledError:
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
    return _ManagedToolResult(
        result=result,
        args=state["args"],
        middleware_trace=trace,
        blocked=bool(state["blocked"]),
    )


def _begin_tool_execution(
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
    try:
        from tools.environments.base import set_activity_callback

        set_activity_callback(agent._touch_activity)
    except Exception:
        pass

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


def _emit_tool_completion(
    agent,
    *,
    tool_call_id: str,
    function_name: str,
    function_args: dict[str, Any],
    result: Any,
    failed: bool,
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

async def execute_tool_calls_segmented(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    segments=None,
) -> None:
    """Execute a batch through native async handlers only.

    The core never dispatches a mixed batch through the old synchronous
    executor: doing that would hide an entire conversation turn behind a
    blocking compatibility bridge. Results still append in model emission
    order.
    """
    from tools.registry import registry
    from model_tools import handle_function_call

    tool_calls = list(getattr(assistant_message, "tool_calls", None) or [])
    if not tool_calls:
        return
    tool_budget = _budget_for_agent(agent)
    active_env = get_active_env(effective_task_id)
    unsupported = [
        getattr(tc.function, "name", "")
        for tc in tool_calls
        if (
            (entry := registry.get_entry(getattr(tc.function, "name", "")))
            is None
            or not entry.is_async
        )
    ]
    if unsupported:
        from agent.agent_runtime_helpers import AsyncCapabilityError

        names = ", ".join(sorted(set(unsupported)))
        raise AsyncCapabilityError(
            f"Native async handlers are required for: {names}. "
            "Sync tool execution is not available in async-hermes-agent."
        )

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

    async def _one(index, tc, result_sink, completion_sink):
        name = getattr(tc.function, "name", "")
        args, malformed_args_result = _parse_tool_arguments(
            getattr(tc.function, "arguments", "")
        )
        if malformed_args_result is None:
            async def _dispatch(next_args, middleware_trace):
                dispatch_kwargs = {
                    "tool_call_id": getattr(tc, "id", "") or "",
                    "session_id": getattr(agent, "session_id", "") or "",
                    "turn_id": getattr(agent, "_current_turn_id", "") or "",
                    "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
                    "user_task": getattr(agent, "_current_user_task", None),
                    "enabled_tools": (
                        list(getattr(agent, "valid_tool_names", None) or []) or None
                    ),
                    "skip_pre_tool_call_hook": True,
                    "skip_tool_request_middleware": True,
                    "skip_tool_execution_middleware": True,
                    "tool_request_middleware_trace": list(middleware_trace),
                }
                if name == "memory":
                    dispatch_kwargs["store"] = getattr(agent, "_memory_store", None)
                elif name == "todo":
                    dispatch_kwargs["store"] = getattr(agent, "_todo_store", None)
                elif name == "session_search":
                    dispatch_kwargs["db"] = getattr(agent, "_session_db", None)
                    dispatch_kwargs["current_session_id"] = getattr(
                        agent, "session_id", None
                    )
                elif name == "clarify":
                    dispatch_kwargs["callback"] = getattr(agent, "clarify_callback", None)
                elif name == "read_terminal":
                    dispatch_kwargs["callback"] = getattr(
                        agent, "read_terminal_callback", None
                    )
                return await handle_function_call(
                    name, next_args, effective_task_id, **dispatch_kwargs
                )

            managed = await _run_agent_tool_execution_middleware(
                agent,
                function_name=name,
                function_args=args,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tc, "id", "") or "",
                display_index=index + 1,
                execute=_dispatch,
            )
            result = managed.result
            blocked = managed.blocked
            failed, _ = _detect_tool_failure(name, result)
            if not managed.blocked:
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
                touch_activity(f"tool completed: {name}{status_suffix}")
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
                hints = check_hints(name, managed.args) if callable(check_hints) else ""
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
                    tool_message.get("_tool_output_risk"),
                )
        else:
            result_sink[index] = make_tool_result_message(
                name, malformed_args_result, getattr(tc, "id", "") or ""
            )

    execution_cwd = None
    if active_env is not None and getattr(active_env, "cwd", None):
        execution_cwd = Path(active_env.cwd)

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
            try:
                async with asyncio.TaskGroup() as group:
                    for index, tool_call in enumerate(calls):
                        group.create_task(
                            _one(
                                index,
                                tool_call,
                                segment_results,
                                segment_completions,
                            )
                        )
            except asyncio.CancelledError:
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
            tool_call_id, name, args, result, failed, risk_metadata = completion
            _emit_tool_completion(
                agent,
                tool_call_id=tool_call_id,
                function_name=name,
                function_args=args,
                result=result,
                failed=failed,
                risk_metadata=risk_metadata,
            )

    await enforce_turn_budget(
        messages[-len(tool_calls):],
        env=active_env,
        config=tool_budget,
    )
    agent._apply_pending_steer_to_tool_results(messages, len(tool_calls))


__all__ = ["execute_tool_calls_segmented"]
