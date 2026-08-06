"""Durability contracts for the native async tool scheduler."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.tool_executor import execute_tool_calls_segmented


def _tool_call(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _agent(flush):
    guardrails = SimpleNamespace(
        before_call=lambda *_args: SimpleNamespace(allows_execution=True)
    )
    return SimpleNamespace(
        _interrupt_requested=False,
        _incremental_persistence_failed=False,
        _flush_messages_to_session_db=flush,
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
    assert json.loads(messages[0]["content"]) == {"memories": ["fact"]}
