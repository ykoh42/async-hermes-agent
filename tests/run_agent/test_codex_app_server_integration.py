"""Async integration coverage for the Codex app-server AIAgent path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import run_agent
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    TurnResult,
)


def _turn(user_input: str = "hello") -> TurnResult:
    return TurnResult(
        final_text=f"echo: {user_input}",
        projected_messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "exec_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "exec_1", "content": "ok"},
            {"role": "assistant", "content": f"echo: {user_input}"},
        ],
        tool_iterations=1,
        turn_id="turn-stub-1",
        thread_id="thread-stub-1",
    )


@pytest.fixture
def fake_session(monkeypatch):
    async def run_turn(self, user_input: str, **kwargs):
        return _turn(user_input)

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)


def _make_agent(**kwargs):
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **kwargs,
    )


def test_api_mode_is_codex_app_server():
    agent = _make_agent()
    assert agent.api_mode == "codex_app_server"
    assert agent.client is None


def test_interrupt_ignores_codex_session_request_failure():
    agent = _make_agent()

    def request_interrupt():
        raise RuntimeError("interrupt transport failed")

    agent._codex_session = SimpleNamespace(request_interrupt=request_interrupt)

    agent.interrupt("stop")

    assert agent._interrupt_requested is True
    assert agent._interrupt_message == "stop"
    assert agent._interrupt_event.is_set()


@pytest.mark.asyncio
async def test_run_conversation_preserves_result_and_message_shape(fake_session):
    agent = _make_agent()
    try:
        result = await agent.run_conversation("hello there")
        assert result["final_response"] == "echo: hello there"
        assert result["completed"] is True
        assert result["partial"] is False
        assert result["api_calls"] == 1
        assert result["codex_thread_id"] == "thread-stub-1"
        assert result["codex_turn_id"] == "turn-stub-1"
        assert sum(
            message.get("role") == "user"
            and message.get("content") == "hello there"
            for message in result["messages"]
        ) == 1
        assert result["messages"][-1] == {
            "role": "assistant",
            "content": "echo: hello there",
        }
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_token_usage_updates_session_accounting(monkeypatch):
    async def run_turn(self, user_input: str, **kwargs):
        return TurnResult(
            final_text="done",
            projected_messages=[{"role": "assistant", "content": "done"}],
            turn_id="turn-usage-1",
            thread_id="thread-usage-1",
            token_usage_last={
                "totalTokens": 130,
                "inputTokens": 80,
                "cachedInputTokens": 20,
                "outputTokens": 25,
                "reasoningOutputTokens": 5,
            },
            model_context_window=200_000,
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    agent = _make_agent()
    try:
        result = await agent.run_conversation("hello")
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 25
        assert result["total_tokens"] == 130
        assert result["cache_read_tokens"] == 20
        assert result["reasoning_tokens"] == 5
        assert agent.session_api_calls == 1
        assert agent.context_compressor.context_length == 200_000
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_native_compaction_updates_bookkeeping(monkeypatch):
    async def run_turn(self, user_input: str, **kwargs):
        return TurnResult(
            final_text="done",
            projected_messages=[{"role": "assistant", "content": "done"}],
            turn_id="turn-compact-1",
            thread_id="thread-compact-1",
            compacted=True,
            token_usage_last={"totalTokens": 300_000, "inputTokens": 300_000},
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    events = []
    agent = _make_agent(
        event_callback=lambda name, payload: events.append((name, payload))
    )
    try:
        result = await agent.run_conversation("hello")
        assert result["completed"] is True
        assert agent.context_compressor.compression_count == 1
        assert agent.context_compressor.last_prompt_tokens == 300_000
        assert events[0][0] == "session:compress"
        assert events[0][1]["runtime"] == "codex_app_server"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_projected_messages_sync_to_external_memory(fake_session):
    agent = _make_agent()
    manager = SimpleNamespace(sync_all=AsyncMock())
    agent._memory_manager = manager
    try:
        result = await agent.run_conversation("hello")
        manager.sync_all.assert_awaited_once()
        assert manager.sync_all.await_args.kwargs["messages"] == result["messages"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_session_exception_returns_partial_and_retires(monkeypatch):
    closes = 0

    async def run_turn(self, user_input, **kwargs):
        raise RuntimeError("subprocess died")

    async def close(self):
        nonlocal closes
        closes += 1

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    monkeypatch.setattr(CodexAppServerSession, "close", close)
    agent = _make_agent()
    result = await agent.run_conversation("hi")
    assert result["completed"] is False
    assert result["partial"] is True
    assert "subprocess died" in result["error"]
    assert agent._codex_session is None
    assert closes == 1
    await agent.close()


@pytest.mark.asyncio
async def test_cancellation_closes_session_and_propagates(monkeypatch):
    started = asyncio.Event()
    interrupted = False
    closed = False

    async def run_turn(self, user_input, **kwargs):
        started.set()
        await asyncio.Event().wait()

    def request_interrupt(self):
        nonlocal interrupted
        interrupted = True

    async def close(self):
        nonlocal closed
        closed = True

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    monkeypatch.setattr(CodexAppServerSession, "request_interrupt", request_interrupt)
    monkeypatch.setattr(CodexAppServerSession, "close", close)

    agent = _make_agent()
    turn = asyncio.create_task(agent.run_conversation("hi"))
    await started.wait()
    turn.cancel()

    with pytest.raises(asyncio.CancelledError):
        await turn

    assert interrupted is True
    assert closed is True
    assert agent._codex_session is None
    await agent.close()


@pytest.mark.asyncio
async def test_should_retire_closes_session(monkeypatch):
    close = AsyncMock()

    async def run_turn(self, user_input, **kwargs):
        result = _turn(user_input)
        result.error = "turn timed out"
        result.interrupted = True
        result.should_retire = True
        return result

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    monkeypatch.setattr(CodexAppServerSession, "close", close)
    agent = _make_agent()
    result = await agent.run_conversation("hi")
    assert result["partial"] is True
    assert result["error"] == "turn timed out"
    assert agent._codex_session is None
    close.assert_awaited_once()
    await agent.close()


@pytest.mark.asyncio
async def test_session_is_wired_to_live_event_bridge(monkeypatch):
    captured = {}
    events = []

    def init(self, **kwargs):
        captured.update(kwargs)

    async def run_turn(self, user_input, **kwargs):
        captured["on_event"](
            {
                "method": "item/started",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "command": "pytest",
                        "cwd": "/repo",
                    }
                },
            }
        )
        return _turn(user_input)

    monkeypatch.setattr(CodexAppServerSession, "__init__", init)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    agent = _make_agent()
    agent.tool_progress_callback = (
        lambda kind, name, preview, args, **kwargs: events.append(
            (kind, name, preview)
        )
    )
    try:
        await agent.run_conversation("run tests")
        assert events == [("tool.started", "exec_command", "pytest")]
    finally:
        await agent.close()
