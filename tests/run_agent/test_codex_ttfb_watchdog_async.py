"""Native-async parity tests for the Codex internal-stream watchdogs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.chat_completion_helpers import interruptible_api_call


def _codex_agent(execute, *, stale_timeout: float = 1.0):
    return SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        _base_url_hostname="chatgpt.com",
        _execute_model_request=execute,
        _compute_non_stream_stale_timeout=AsyncMock(return_value=stale_timeout),
        _consecutive_stale_streams=0,
        _codex_silent_hang_hint=lambda **_kwargs: "Try gpt-5.4 instead.",
        _touch_activity=MagicMock(),
        _buffer_status=MagicMock(),
        _emit_wait_notice=MagicMock(),
    )


@pytest.mark.asyncio
async def test_codex_nonstream_entry_uses_internal_stream_and_ttfb(monkeypatch):
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.01")
    cancelled = asyncio.Event()
    seen = {}

    async def execute(_payload, **kwargs):
        seen.update(kwargs)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = _codex_agent(execute)

    with pytest.raises(TimeoutError, match="TTFB threshold") as caught:
        await interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})

    assert seen["use_streaming"] is True
    assert callable(seen["on_stream_activity"])
    assert cancelled.is_set()
    assert "Try gpt-5.4 instead" in str(caught.value)
    assert "Try gpt-5.4 instead" in agent._buffer_status.call_args.args[0]
    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_codex_event_satisfies_ttfb_and_arms_idle_watchdog(monkeypatch):
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.1")
    response = object()

    async def execute(_payload, **kwargs):
        await asyncio.sleep(0.005)
        kwargs["on_stream_activity"]()
        await asyncio.sleep(0.02)
        return response

    agent = _codex_agent(execute)

    assert await interruptible_api_call(agent, {"model": "gpt-5.5"}) is response
    assert agent._codex_stream_last_event_ts is not None
    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_codex_wait_notice_heartbeat_is_owned(monkeypatch):
    from agent import chat_completion_helpers as helpers

    monkeypatch.setattr(helpers, "_STREAM_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")
    response = object()

    async def execute(_payload, **_kwargs):
        await asyncio.sleep(0.025)
        return response

    agent = _codex_agent(execute)

    assert await interruptible_api_call(agent, {"model": "gpt-5.5"}) is response
    assert agent._emit_wait_notice.call_count >= 1
    assert "waiting on gpt-5.5" in agent._emit_wait_notice.call_args_list[0].args[0]
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "codex-nonstream-heartbeat"
    ]


@pytest.mark.asyncio
async def test_codex_event_idle_timeout_cancels_native_stream(monkeypatch):
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.01")
    cancelled = asyncio.Event()

    async def execute(_payload, **kwargs):
        kwargs["on_stream_activity"]()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = _codex_agent(execute)

    with pytest.raises(TimeoutError, match="no SSE events"):
        await interruptible_api_call(agent, {"model": "gpt-5.5"})

    assert cancelled.is_set()
    assert "sent no events" in agent._buffer_status.call_args.args[0]
    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_codex_hard_ceiling_remains_absolute(monkeypatch):
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0.01")

    async def execute(_payload, **_kwargs):
        await asyncio.Event().wait()

    agent = _codex_agent(execute, stale_timeout=60.0)

    with pytest.raises(TimeoutError, match="Non-streaming API call timed out"):
        await interruptible_api_call(
            agent,
            {"model": "gpt-5.5", "input": "x" * 44_000},
        )

    assert agent._consecutive_stale_streams == 1


@pytest.mark.asyncio
async def test_codex_watchdog_preserves_external_cancellation(monkeypatch):
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "10")
    cancelled = asyncio.Event()

    async def execute(_payload, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = _codex_agent(execute)
    task = asyncio.create_task(interruptible_api_call(agent, {"model": "gpt-5.5"}))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert agent._consecutive_stale_streams == 0
