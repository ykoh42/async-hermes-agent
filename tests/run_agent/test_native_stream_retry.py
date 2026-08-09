"""Parity tests for the native-async provider stream retry state machine."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent.errors import EmptyStreamError
from agent.chat_completion_helpers import interruptible_streaming_api_call
from hermes_constants import PARTIAL_STREAM_STUB_ID


def _retry_agent(execute):
    return SimpleNamespace(
        _execute_model_request=execute,
        _provider_stale_timeout=0.1,
        _consecutive_stale_streams=0,
        _current_streamed_assistant_text="",
        _current_stream_partial_tool_names=[],
        _interrupt_requested=False,
        _disable_streaming=False,
        _touch_activity=MagicMock(),
        _emit_wait_notice=MagicMock(),
        _stream_diag_init=lambda: {
            "started_at": time.time(),
            "first_chunk_at": None,
            "chunks": 0,
            "bytes": 0,
            "headers": {},
            "http_status": None,
        },
        _is_provider_stream_parse_error=lambda _error: False,
        _emit_stream_drop=MagicMock(),
        _log_stream_retry=MagicMock(),
        _buffer_status=MagicMock(),
        _safe_print=MagicMock(),
        _fire_stream_delta=MagicMock(),
        base_url="https://api.example.test/v1",
        provider="test-provider",
        model="test-model",
    )


@pytest.mark.asyncio
async def test_transient_stream_errors_retry_then_return_success(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    response = object()
    execute = AsyncMock(
        side_effect=[
            httpx.RemoteProtocolError("peer closed"),
            httpx.ConnectError("connect failed"),
            response,
        ]
    )
    agent = _retry_agent(execute)

    assert await interruptible_streaming_api_call(agent, {}) is response

    assert execute.await_count == 3
    assert [
        call.kwargs["attempt"]
        for call in agent._emit_stream_drop.call_args_list
    ] == [2, 3]
    assert all(
        call.kwargs["mid_tool_call"] is False
        for call in agent._emit_stream_drop.call_args_list
    )
    agent._log_stream_retry.assert_not_called()
    assert agent._consecutive_stale_streams == 0
    assert all(
        isinstance(call.kwargs["_stream_diag"], dict)
        for call in execute.await_args_list
    )


@pytest.mark.asyncio
async def test_empty_stream_retries_then_logs_exhaustion(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    execute = AsyncMock(
        side_effect=[
            EmptyStreamError("empty stream"),
            EmptyStreamError("empty stream"),
            EmptyStreamError("empty stream"),
        ]
    )
    agent = _retry_agent(execute)

    with pytest.raises(EmptyStreamError, match="empty stream"):
        await interruptible_streaming_api_call(agent, {})

    assert execute.await_count == 3
    assert agent._emit_stream_drop.call_count == 2
    agent._log_stream_retry.assert_called_once()
    assert agent._log_stream_retry.call_args.kwargs["kind"] == "exhausted"
    assert "empty response stream after 3 attempts" in (
        agent._buffer_status.call_args.args[0]
    )


@pytest.mark.asyncio
async def test_sse_connection_error_retries_as_transient(monkeypatch):
    from openai import APIError

    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    error = APIError(
        message="Network connection lost.",
        request=httpx.Request("POST", "https://api.example.test/v1/chat"),
        body={"message": "Network connection lost."},
    )
    execute = AsyncMock(side_effect=error)
    agent = _retry_agent(execute)

    with pytest.raises(APIError, match="Network connection lost"):
        await interruptible_streaming_api_call(agent, {})

    assert execute.await_count == 3
    assert agent._emit_stream_drop.call_count == 2
    agent._log_stream_retry.assert_called_once()


@pytest.mark.asyncio
async def test_known_provider_parse_error_retries(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    error = ValueError("invalid SSE event: unterminated payload")
    execute = AsyncMock(side_effect=error)
    agent = _retry_agent(execute)
    agent._is_provider_stream_parse_error = lambda candidate: candidate is error

    with pytest.raises(ValueError, match="invalid SSE event"):
        await interruptible_streaming_api_call(agent, {})

    assert execute.await_count == 2
    assert "malformed streaming data after 2 attempts" in (
        agent._buffer_status.call_args.args[0]
    )


@pytest.mark.asyncio
async def test_generic_value_error_is_not_retried(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    execute = AsyncMock(side_effect=ValueError("invalid local request shape"))
    agent = _retry_agent(execute)

    with pytest.raises(ValueError, match="invalid local request shape"):
        await interruptible_streaming_api_call(agent, {})

    execute.assert_awaited_once()
    agent._emit_stream_drop.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_stream_disables_streaming_without_inner_retry(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    execute = AsyncMock(side_effect=RuntimeError("stream not supported"))
    agent = _retry_agent(execute)

    with pytest.raises(RuntimeError, match="stream not supported"):
        await interruptible_streaming_api_call(agent, {})

    execute.assert_awaited_once()
    assert agent._disable_streaming is True
    agent._safe_print.assert_called_once()
    agent._emit_stream_drop.assert_not_called()


@pytest.mark.asyncio
async def test_visible_text_returns_partial_stub_without_retry(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    execute = AsyncMock(side_effect=httpx.RemoteProtocolError("peer closed"))
    agent = _retry_agent(execute)
    agent._current_streamed_assistant_text = "partial answer"

    response = await interruptible_streaming_api_call(agent, {})

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content == "partial answer"
    execute.assert_awaited_once()
    agent._emit_stream_drop.assert_not_called()


@pytest.mark.asyncio
async def test_whitespace_text_returns_upstream_empty_partial_stub(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    execute = AsyncMock(side_effect=httpx.RemoteProtocolError("peer closed"))
    agent = _retry_agent(execute)
    agent._current_streamed_assistant_text = "   "

    response = await interruptible_streaming_api_call(agent, {})

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content is None
    execute.assert_awaited_once()
    agent._emit_stream_drop.assert_not_called()


@pytest.mark.asyncio
async def test_delivered_delta_returns_empty_stub_when_callback_keeps_no_text(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")

    async def execute(*_args, **kwargs):
        kwargs["_on_stream_text"]()
        raise httpx.RemoteProtocolError("peer closed")

    agent = _retry_agent(execute)

    response = await interruptible_streaming_api_call(agent, {})

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content is None
    agent._emit_stream_drop.assert_not_called()


@pytest.mark.asyncio
async def test_partial_tool_stream_retries_and_resets_delivery(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    response = object()
    holder = {}
    attempts = 0

    async def execute(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            agent = holder["agent"]
            agent._current_streamed_assistant_text = "using a tool"
            agent._current_stream_partial_tool_names = ["terminal"]
            raise httpx.RemoteProtocolError("peer closed")
        return response

    agent = _retry_agent(execute)
    holder["agent"] = agent
    resets = []

    def reset_delivery():
        resets.append(True)
        agent._current_streamed_assistant_text = ""
        agent._current_stream_partial_tool_names = []

    agent._reset_stream_delivery_tracking = reset_delivery

    assert await interruptible_streaming_api_call(agent, {}) is response

    assert attempts == 2
    assert resets == [True]
    agent._fire_stream_delta.assert_called_once_with(
        "\n\n⚠ Connection dropped mid tool-call; reconnecting…\n\n"
    )
    agent._emit_stream_drop.assert_called_once()
    assert agent._emit_stream_drop.call_args.kwargs["mid_tool_call"] is True


@pytest.mark.asyncio
async def test_partial_tool_retry_exhaustion_returns_warning_stub(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    holder = {}
    attempts = 0

    async def execute(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        agent = holder["agent"]
        agent._current_streamed_assistant_text = "using a tool"
        agent._current_stream_partial_tool_names = ["write_file"]
        raise httpx.RemoteProtocolError("peer closed")

    agent = _retry_agent(execute)
    holder["agent"] = agent

    def reset_delivery():
        agent._current_streamed_assistant_text = ""
        agent._current_stream_partial_tool_names = []

    agent._reset_stream_delivery_tracking = reset_delivery

    response = await interruptible_streaming_api_call(agent, {})

    assert attempts == 2
    assert "Stream stalled mid tool-call (write_file)" in (
        response.choices[0].message.content
    )
    assert response.choices[0].message.tool_calls is None
    assert agent._emit_stream_drop.call_count == 1


@pytest.mark.asyncio
async def test_interrupt_flag_stops_before_transient_retry(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    holder = {}
    attempts = 0

    async def execute(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        holder["agent"]._interrupt_requested = True
        raise httpx.RemoteProtocolError("peer closed")

    agent = _retry_agent(execute)
    holder["agent"] = agent

    with pytest.raises(InterruptedError, match="before stream retry"):
        await interruptible_streaming_api_call(agent, {})

    assert attempts == 1
    agent._emit_stream_drop.assert_called_once()


@pytest.mark.asyncio
async def test_stream_heartbeat_reports_wait_and_finishes_owned_task(
    monkeypatch,
):
    from agent import chat_completion_helpers

    monkeypatch.setattr(
        chat_completion_helpers,
        "_STREAM_HEARTBEAT_INTERVAL",
        0.01,
    )

    response = object()

    async def execute(*_args, **_kwargs):
        await asyncio.sleep(0.035)
        return response

    agent = _retry_agent(execute)
    agent._provider_stale_timeout = 0.2

    assert await interruptible_streaming_api_call(
        agent,
        {"model": "heartbeat-model"},
    ) is response

    assert agent._emit_wait_notice.call_count >= 1
    assert "waiting on heartbeat-model" in (
        agent._emit_wait_notice.call_args_list[0].args[0]
    )
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("provider-stream-heartbeat-")
    ]


@pytest.mark.asyncio
async def test_external_cancellation_closes_request_and_heartbeat(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def execute(*_args, **_kwargs):
        request_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            request_cancelled.set()

    agent = _retry_agent(execute)
    task = asyncio.create_task(
        interruptible_streaming_api_call(agent, {}),
        name="cancel-native-provider-stream",
    )
    await request_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert request_cancelled.is_set()
    assert not [
        pending
        for pending in asyncio.all_tasks()
        if pending.get_name().startswith("provider-stream-heartbeat-")
    ]


@pytest.mark.asyncio
async def test_stale_watchdog_preserves_upstream_status_diagnostics(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    async def execute(*_args, **_kwargs):
        await asyncio.Event().wait()

    agent = _retry_agent(execute)
    agent._provider_stale_timeout = 0.01

    with pytest.raises(TimeoutError, match="produced no chunks"):
        await interruptible_streaming_api_call(
            agent,
            {"model": "stalled-model", "messages": [{"content": "hello"}]},
        )

    assert "No response from provider" in agent._buffer_status.call_args_list[0].args[0]
    assert "no output from provider" in agent._emit_wait_notice.call_args.args[0]
    assert any(
        "stale stream detected" in call.args[0]
        for call in agent._touch_activity.call_args_list
    )
