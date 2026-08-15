"""Native-async tests for upstream-compatible stream observer hooks."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.chat_completion_helpers import interruptible_streaming_api_call


def _callbacks(callbacks_by_hook):
    return lambda name: tuple(callbacks_by_hook.get(name, ()))


@pytest.fixture(autouse=True)
def _shutdown_dispatcher():
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    yield
    shutdown_plugin_stream_hook_dispatcher()


def _hook_agent():
    from run_agent import AIAgent

    agent = SimpleNamespace(
        _current_turn_id="turn-1",
        _api_call_count=2,
        session_id="session-1",
        model="test-model",
        provider="openrouter",
        platform="library",
    )
    agent._stream_hook_base_payload = lambda: AIAgent._stream_hook_base_payload(agent)
    agent._emit_stream_start = lambda: AIAgent._emit_stream_start(agent)
    agent._emit_stream_end = lambda **kwargs: AIAgent._emit_stream_end(agent, **kwargs)
    return agent


@pytest.mark.asyncio
async def test_stream_hooks_are_valid_and_lifecycle_payload_is_preserved(monkeypatch):
    from hermes_cli.plugins import VALID_HOOKS
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    assert {
        "on_stream_start",
        "on_stream_delta",
        "on_stream_end",
        "on_interim_message",
    }.issubset(VALID_HOOKS)

    calls: list[tuple[str, dict]] = []

    async def on_start(**payload):
        calls.append(("start", payload))

    async def on_end(**payload):
        calls.append(("end", payload))

    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        _callbacks({"on_stream_start": [on_start], "on_stream_end": [on_end]}),
    )
    agent = _hook_agent()
    agent._emit_stream_start()
    agent._emit_stream_end(final_text="done", finished=True, error=None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    shutdown_plugin_stream_hook_dispatcher()

    assert {name for name, _payload in calls} == {"start", "end"}
    start = next(payload for name, payload in calls if name == "start")
    end = next(payload for name, payload in calls if name == "end")
    assert start["model"] == "test-model"
    assert start["provider"] == "openrouter"
    assert end["final_text"] == "done"
    assert end["finished"] is True
    assert end["error"] is None


@pytest.mark.asyncio
async def test_stream_observer_dispatch_is_async_and_callback_errors_are_isolated(
    monkeypatch,
):
    from agent.plugin_stream_hooks import (
        enqueue_plugin_stream_hook,
        shutdown_plugin_stream_hook_dispatcher,
    )

    delivered = asyncio.Event()
    seen: list[str] = []

    async def slow_callback(**payload):
        await asyncio.sleep(0)
        seen.append(payload["delta"])
        delivered.set()

    def broken_callback(**_payload):
        raise RuntimeError("observer failure")

    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        _callbacks({"on_stream_delta": [slow_callback, broken_callback]}),
    )
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="hello") is True
    await asyncio.wait_for(delivered.wait(), timeout=1)
    shutdown_plugin_stream_hook_dispatcher()
    assert seen == ["hello"]


def _stream_agent(execute):
    return SimpleNamespace(
        _execute_model_request=execute,
        api_mode="chat_completions",
        provider="test-provider",
        model="test-model",
        base_url="https://api.example.test/v1",
        _provider_stale_timeout=1.0,
        _provider_request_timeout=1.0,
        _interrupt_requested=False,
        _current_streamed_assistant_text="",
        _current_stream_partial_tool_names=[],
        _consecutive_stale_streams=0,
        _stream_diag_init=lambda: {},
        _is_provider_stream_parse_error=lambda _error: False,
        _touch_activity=MagicMock(),
        _emit_wait_notice=MagicMock(),
        _buffer_status=MagicMock(),
        _safe_print=MagicMock(),
        _emit_stream_drop=MagicMock(),
        _log_stream_retry=MagicMock(),
        _emit_stream_start=MagicMock(),
        _emit_stream_end=MagicMock(),
    )


@pytest.mark.asyncio
async def test_interruptible_stream_emits_start_and_end_once(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    agent = _stream_agent(AsyncMock(return_value=response))

    assert await interruptible_streaming_api_call(agent, {}) is response
    agent._emit_stream_start.assert_called_once_with()
    agent._emit_stream_end.assert_called_once_with(
        final_text="answer", finished=True, error=None
    )


@pytest.mark.asyncio
async def test_interruptible_stream_emits_failed_end_on_cancellation(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    async def execute(*_args, **_kwargs):
        raise asyncio.CancelledError()

    agent = _stream_agent(execute)
    with pytest.raises(asyncio.CancelledError):
        await interruptible_streaming_api_call(agent, {})

    agent._emit_stream_start.assert_called_once_with()
    agent._emit_stream_end.assert_called_once()
    assert agent._emit_stream_end.call_args.kwargs["finished"] is False
