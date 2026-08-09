"""Native-async parity tests for the streaming single-writer invariant."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.codex_runtime import run_codex_stream
from run_agent import AIAgent


def _sink_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.reasoning_callback = None
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent._current_streamed_assistant_text = ""
    return agent


@pytest.mark.asyncio
async def test_superseded_task_deltas_are_dropped():
    agent = _sink_agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    first_claimed = asyncio.Event()
    second_claimed = asyncio.Event()

    async def first_writer():
        agent._claim_stream_writer()
        first_claimed.set()
        await second_claimed.wait()
        agent._fire_stream_delta("stale text")
        agent._fire_reasoning_delta("stale reasoning")
        agent._record_streamed_assistant_text("stale record")

    async def second_writer():
        await first_claimed.wait()
        agent._claim_stream_writer()
        second_claimed.set()

    await asyncio.gather(first_writer(), second_writer())

    assert delivered == []
    assert agent._current_streamed_assistant_text == ""
    assert agent._stream_writer_dropped >= 2


@pytest.mark.asyncio
async def test_current_task_writer_is_never_fenced():
    agent = _sink_agent()
    delivered = []
    agent.stream_delta_callback = delivered.append

    agent._claim_stream_writer()
    agent._fire_stream_delta("hello ")
    agent._fire_stream_delta("world")

    assert delivered == ["hello ", "world"]
    assert agent._current_streamed_assistant_text == "hello world"
    assert agent._stream_writer_dropped == 0


@pytest.mark.asyncio
async def test_non_claiming_task_is_not_a_writer():
    agent = _sink_agent()
    delivered = []
    agent.stream_delta_callback = delivered.append

    async def other_writer():
        agent._claim_stream_writer()
        agent._claim_stream_writer()

    await asyncio.create_task(other_writer())

    assert agent._stream_writer_superseded() is False
    agent._fire_stream_delta("plain")
    assert delivered == ["plain"]


def _chat_chunk(content: str, *, finish_reason=None):
    return SimpleNamespace(
        id="stream-id",
        model="test-model",
        usage=None,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    content=content,
                    reasoning=None,
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
    )


class _SupersededChatStream:
    def __init__(self):
        self.agent = None
        self.index = 0
        self.response = SimpleNamespace(headers={})
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.index += 1
        if self.index == 1:
            return _chat_chunk("first")
        if self.index == 2:
            self.agent._claim_stream_writer()
            return _chat_chunk("-stale-tail", finish_reason="stop")
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_chat_consume_loop_stops_when_superseded_mid_stream():
    stream = _SupersededChatStream()

    class _Completions:
        async def create(self, **_kwargs):
            return stream

    agent = _sink_agent()
    stream.agent = agent
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent.api_mode = "chat_completions"
    agent.provider = "test-provider"
    agent.model = "test-model"
    agent.base_url = "https://api.example.test/v1"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    agent._touch_activity = MagicMock()
    agent._capture_rate_limits = MagicMock()
    agent._capture_credits = MagicMock()
    agent._check_openrouter_cache_status = MagicMock()
    agent._fire_tool_gen_started = MagicMock()

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
    )

    assert delivered == ["first"]
    assert response.choices[0].message.content == "first"
    assert stream.closed is True


class _SupersededCodexStream:
    def __init__(self, agent):
        self.agent = agent
        self.index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.index += 1
        if self.index == 1:
            return SimpleNamespace(
                type="response.output_text.delta",
                delta="first",
                item_id="i1",
            )
        if self.index == 2:
            self.agent._claim_stream_writer()
            return SimpleNamespace(
                type="response.output_text.delta",
                delta="-stale-tail",
                item_id="i1",
            )
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_codex_consume_loop_stops_when_superseded_mid_stream():
    agent = _sink_agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent._codex_stream_event_callback = None
    agent._codex_stream_text_callback = None
    agent._fire_streamed_codex_commentary = MagicMock()
    agent.show_commentary = True
    agent.client = SimpleNamespace()
    stream = _SupersededCodexStream(agent)
    agent.client.responses = SimpleNamespace(create=AsyncMock(return_value=stream))

    response = await run_codex_stream(
        agent,
        {"model": "gpt-test"},
        client=agent.client,
    )

    assert delivered == ["first"]
    assert "-stale-tail" not in response.output_text
    assert stream.closed is True
