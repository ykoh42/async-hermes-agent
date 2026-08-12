"""Durability contracts for the native async tool scheduler."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.tool_executor import (
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
    execute_tool_calls_segmented,
)
from agent.agent_runtime_helpers import invoke_tool


def _tool_call(
    name: str,
    call_id: str,
    arguments: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments or {}),
        ),
    )


def _agent(flush):
    guardrails = SimpleNamespace(
        before_call=lambda *_args: SimpleNamespace(allows_execution=True)
    )
    return SimpleNamespace(
        _interrupt_requested=False,
        _incremental_persistence_failed=False,
        _flush_messages_to_session_db=flush,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _tool_guardrails=guardrails,
        _append_guardrail_observation=lambda _name, _args, result, **_kwargs: result,
        _apply_pending_steer_to_tool_results=lambda *_args: None,
        _touch_activity=lambda *_args: None,
        _current_tool=None,
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
        _current_user_task=None,
        session_id="session-1",
        valid_tool_names={"read_file", "terminal"},
        enabled_toolsets=None,
        disabled_toolsets=None,
        _memory_store=None,
        _todo_store=None,
        clarify_callback=None,
        read_terminal_callback=None,
        _turns_since_memory=0,
        quiet_mode=True,
        verbose_logging=False,
        log_prefix_chars=160,
        tool_progress_mode="all",
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        context_compressor=None,
    )


def _native_tool_entry():
    return patch(
        "tools.registry.registry.get_entry",
        return_value=SimpleNamespace(is_async=True, max_result_size_chars=None),
    )


def _native_policy_path():
    return patch(
        "hermes_cli.plugins.resolve_pre_tool_block",
        new_callable=AsyncMock,
        return_value=None,
    )


@pytest.mark.parametrize(
    "executor",
    [execute_tool_calls_concurrent, execute_tool_calls_sequential],
)
@pytest.mark.asyncio
async def test_public_executor_preserves_finalize_false_contract(executor):
    async def flush(_messages):
        return True

    agent = _agent(flush)
    steer_calls = []
    agent._apply_pending_steer_to_tool_results = lambda *args: steer_calls.append(args)
    call = _tool_call("read_file", "read-1")
    messages = []

    with (
        patch(
            "model_tools.handle_function_call",
            new_callable=AsyncMock,
            return_value=json.dumps({"content": "data"}),
        ),
        patch(
            "agent.tool_executor.enforce_turn_budget",
            new_callable=AsyncMock,
        ) as enforce_budget,
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await executor(
            agent,
            SimpleNamespace(tool_calls=[call]),
            messages,
            "task-1",
            finalize=False,
        )

    assert [message["tool_call_id"] for message in messages] == ["read-1"]
    enforce_budget.assert_not_awaited()
    assert steer_calls == []


@pytest.mark.asyncio
async def test_unknown_tool_is_returned_as_observation_instead_of_fail_fast():
    async def flush(_messages):
        return True

    agent = _agent(flush)
    call = _tool_call("unknown_tool", "unknown-1")
    messages = []

    with (
        patch(
            "model_tools.handle_function_call",
            new_callable=AsyncMock,
            return_value=json.dumps({"error": "Unknown tool: unknown_tool"}),
        ) as dispatch,
        _native_policy_path(),
    ):
        await execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[call]),
            messages,
            "task-1",
            finalize=False,
        )

    dispatch.assert_awaited_once()
    assert json.loads(messages[0]["content"])["error"] == (
        "Unknown tool: unknown_tool"
    )


@pytest.mark.asyncio
async def test_session_search_uses_agent_owned_lazy_database():
    from model_tools import _TOOL_HANDLER_CONTEXT

    async def flush(_messages):
        return True

    agent = _agent(flush)
    sentinel_db = object()
    agent._get_session_db_for_recall = AsyncMock(return_value=sentinel_db)
    call = _tool_call("session_search", "search-1", {"query": "Hermes"})
    messages = []
    captured = {}

    async def dispatch(*_args, **_kwargs):
        captured.update(_TOOL_HANDLER_CONTEXT.get() or {})
        return json.dumps({"success": True, "results": []})

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[call]),
            messages,
            "task-1",
            finalize=False,
        )

    agent._get_session_db_for_recall.assert_awaited_once_with()
    assert captured["db"] is sentinel_db
    assert captured["current_session_id"] == "session-1"


@pytest.mark.asyncio
async def test_invoke_tool_preserves_unknown_tool_error_contract():
    agent = _agent(AsyncMock(return_value=True))

    result = await invoke_tool(
        agent,
        "unknown_tool",
        {},
        "task-1",
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert json.loads(result)["error"] == "Unknown tool: unknown_tool"


@pytest.mark.asyncio
async def test_completion_callback_runs_only_after_segment_persistence():
    events = []

    async def flush(messages):
        events.append(("flush", [message["tool_call_id"] for message in messages]))
        return True

    agent = _agent(flush)
    agent.tool_complete_callback = lambda *_args: events.append(("complete", None))
    call = _tool_call("read_file", "read-1")

    with (
        patch(
            "model_tools.handle_function_call",
            new_callable=AsyncMock,
            return_value=json.dumps({"content": "data"}),
        ),
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await execute_tool_calls_segmented(
            agent,
            SimpleNamespace(tool_calls=[call]),
            [],
            "task-1",
            segments=[("sequential", [call])],
        )

    assert events == [("flush", ["read-1"]), ("complete", None)]


@pytest.mark.asyncio
async def test_failed_persistence_stops_later_segments_and_completion_callbacks():
    async def flush(_messages):
        return False

    agent = _agent(flush)
    completed = []
    agent.tool_complete_callback = lambda *args: completed.append(args)
    first = _tool_call("read_file", "read-1")
    second = _tool_call("terminal", "terminal-1")

    with (
        patch(
            "model_tools.handle_function_call",
            new_callable=AsyncMock,
            return_value=json.dumps({"ok": True}),
        ) as dispatch,
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await execute_tool_calls_segmented(
            agent,
            SimpleNamespace(tool_calls=[first, second]),
            [],
            "task-1",
            segments=[("sequential", [first]), ("sequential", [second])],
        )

    assert agent._incremental_persistence_failed is True
    assert dispatch.await_count == 1
    assert completed == []


@pytest.mark.asyncio
async def test_parallel_segment_preserves_model_tool_call_order():
    async def flush(_messages):
        return True

    agent = _agent(flush)
    first = _tool_call("read_file", "read-1")
    second = _tool_call("read_file", "read-2")
    messages = []

    async def dispatch(_name, _args, _task_id, **kwargs):
        if kwargs["tool_call_id"] == "read-1":
            await asyncio.sleep(0.01)
        return json.dumps({"id": kwargs["tool_call_id"]})

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await execute_tool_calls_segmented(
            agent,
            SimpleNamespace(tool_calls=[first, second]),
            messages,
            "task-1",
            segments=[("parallel", [first, second])],
        )

    assert [message["tool_call_id"] for message in messages] == ["read-1", "read-2"]
    assert [json.loads(message["content"])["id"] for message in messages] == [
        "read-1",
        "read-2",
    ]


@pytest.mark.asyncio
async def test_interrupt_before_later_segment_persists_cancelled_observations():
    snapshots = []

    async def flush(messages):
        snapshots.append(copy.deepcopy(messages))
        return True

    agent = _agent(flush)
    first = _tool_call("terminal", "terminal-1")
    second = _tool_call("read_file", "read-1")

    async def dispatch(_name, _args, _task_id, **kwargs):
        if kwargs["tool_call_id"] == "terminal-1":
            agent._interrupt_requested = True
        return json.dumps({"ok": True})

    messages = []
    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await execute_tool_calls_segmented(
            agent,
            SimpleNamespace(tool_calls=[first, second]),
            messages,
            "task-1",
            segments=[("sequential", [first]), ("sequential", [second])],
        )

    assert [message["tool_call_id"] for message in messages] == ["terminal-1", "read-1"]
    assert "cancelled" in messages[-1]["content"]
    assert snapshots[-1][-1]["tool_call_id"] == "read-1"


@pytest.mark.asyncio
async def test_external_memory_tool_uses_native_manager_dispatch():
    async def flush(_messages):
        return True

    agent = _agent(flush)
    memory_dispatch = AsyncMock(return_value=json.dumps({"memories": ["fact"]}))
    agent._memory_manager = SimpleNamespace(
        has_tool=lambda name: name == "external_recall",
        handle_tool_call=memory_dispatch,
    )
    call = _tool_call("external_recall", "memory-1")
    messages = []

    with (
        patch("tools.registry.registry.get_entry", return_value=None),
        _native_policy_path(),
        patch("model_tools.handle_function_call", new_callable=AsyncMock) as registry_dispatch,
        patch(
            "model_tools._emit_post_tool_call_hook",
            new_callable=AsyncMock,
        ) as post_hook,
    ):
        await execute_tool_calls_segmented(
            agent,
            SimpleNamespace(tool_calls=[call]),
            messages,
            "task-1",
            segments=[("sequential", [call])],
        )

    memory_dispatch.assert_awaited_once_with("external_recall", {})
    registry_dispatch.assert_not_awaited()
    post_hook.assert_awaited_once()
    assert post_hook.await_args.kwargs["function_name"] == "external_recall"
    assert post_hook.await_args.kwargs["tool_call_id"] == "memory-1"
    assert json.loads(messages[0]["content"]) == {"memories": ["fact"]}


@pytest.mark.asyncio
async def test_tool_call_bridge_unwraps_before_policy_and_mcp_dispatch():
    from tools.registry import registry

    tool_name = "mcp_async_bridge_read"
    handler = AsyncMock(return_value=json.dumps({"ok": True}))
    registry.register(
        name=tool_name,
        handler=handler,
        schema={
            "name": tool_name,
            "description": "Read through the async bridge.",
            "parameters": {"type": "object", "properties": {}},
        },
        toolset="mcp-async-bridge-test",
    )

    agent = _agent(AsyncMock(return_value=True))
    agent.enabled_toolsets = ["mcp-async-bridge-test"]
    agent.clarify_callback = object()
    call = _tool_call(
        "tool_call",
        "bridge-1",
        {"name": tool_name, "arguments": {}},
    )
    messages = []

    with _native_policy_path() as policy:
        await execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[call]),
            messages,
            "task-1",
            finalize=False,
        )

    assert policy.await_args.args[0] == tool_name
    handler.assert_awaited_once()
    assert handler.await_args.kwargs["elicitation_callback"] is agent.clarify_callback
    assert messages[0]["name"] == tool_name
    assert json.loads(messages[0]["content"]) == {"ok": True}


@pytest.mark.asyncio
async def test_tool_call_bridge_cannot_escape_session_toolset_scope():
    from tools.registry import registry

    tool_name = "mcp_async_bridge_out_of_scope"
    handler = AsyncMock(return_value=json.dumps({"executed": True}))
    registry.register(
        name=tool_name,
        handler=handler,
        schema={
            "name": tool_name,
            "description": "An out-of-scope bridge tool.",
            "parameters": {"type": "object", "properties": {}},
        },
        toolset="mcp-async-bridge-out-of-scope",
    )

    agent = _agent(AsyncMock(return_value=True))
    agent.enabled_toolsets = ["mcp-unrelated-session-scope"]
    call = _tool_call(
        "tool_call",
        "bridge-2",
        {"name": tool_name, "arguments": {}},
    )
    messages = []

    with _native_policy_path():
        await execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[call]),
            messages,
            "task-1",
            finalize=False,
        )

    handler.assert_not_awaited()
    assert "not available in this session" in messages[0]["content"]


@pytest.mark.asyncio
async def test_parallel_tool_batches_preserve_approval_context_isolation():
    """Native TaskGroup workers inherit each caller's approval session."""
    from tools.approval import (
        get_current_session_key,
        reset_current_session_key,
        set_current_session_key,
    )

    observed: dict[str, list[str]] = {}

    async def handle(_name, _args, task_id, **_kwargs):
        await asyncio.sleep(0)
        observed.setdefault(task_id, []).append(
            get_current_session_key(default="FALLBACK")
        )
        return json.dumps({"ok": True})

    async def run_batch(label: str):
        token = set_current_session_key(f"session-{label}")
        try:
            agent = _agent(AsyncMock(return_value=True))
            calls = [
                _tool_call("read_file", f"{label}-1"),
                _tool_call("read_file", f"{label}-2"),
            ]
            await execute_tool_calls_concurrent(
                agent,
                SimpleNamespace(tool_calls=calls),
                [],
                f"task-{label}",
                finalize=False,
            )
        finally:
            reset_current_session_key(token)

    with (
        patch("model_tools.handle_function_call", side_effect=handle),
        _native_tool_entry(),
        _native_policy_path(),
    ):
        await asyncio.gather(run_batch("A"), run_batch("B"))

    assert observed == {
        "task-A": ["session-A", "session-A"],
        "task-B": ["session-B", "session-B"],
    }
