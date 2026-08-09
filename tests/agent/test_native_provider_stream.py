"""Behavioral parity tests for native Anthropic and chat stream paths."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.anthropic_adapter import create_anthropic_message
from agent.errors import EmptyStreamError
from hermes_constants import PARTIAL_STREAM_STUB_ID
from run_agent import AIAgent


class _AsyncMessageStream:
    def __init__(self, events, final_message=None, final_error=None):
        self._events = list(events)
        self._final_message = final_message
        self._final_error = final_error
        self.response = SimpleNamespace(headers={})
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.exited = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_message(self):
        if self._final_error is not None:
            raise self._final_error
        return self._final_message


class _Messages:
    def __init__(self, stream):
        self._stream = stream

    def stream(self, **_kwargs):
        return self._stream

    async def create(self, **_kwargs):
        raise AssertionError("non-streaming fallback was not expected")


class _AsyncChatStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.response = SimpleNamespace(headers={})
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        self.closed = True


def _chat_chunk(*, content=None, reasoning=None, tool_calls=None, finish=None):
    return SimpleNamespace(
        id="stream-id",
        model="test-model",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                delta=SimpleNamespace(
                    content=content,
                    reasoning=reasoning,
                    reasoning_content=None,
                    tool_calls=tool_calls,
                ),
            )
        ],
    )


def _chat_tool(index, call_id, name, arguments, *, extra_content=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        extra_content=extra_content,
    )


def _native_chat_agent(stream):
    class _Completions:
        async def create(self, **_kwargs):
            return stream

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider = "test-provider"
    agent.model = "test-model"
    agent.base_url = "https://api.example.test/v1"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    agent._touch_activity = MagicMock()
    agent._capture_rate_limits = MagicMock()
    agent._capture_credits = MagicMock()
    agent._check_openrouter_cache_status = MagicMock()
    agent.stream_delta_callback = None
    agent._record_streamed_assistant_text = MagicMock()
    return agent


@pytest.mark.asyncio
async def test_native_anthropic_stream_preserves_delta_callback_order():
    events = [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="reasoning"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="visible before tool"),
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="tool_use", name="terminal"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hidden after tool"),
        ),
    ]
    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="terminal")],
        stop_reason="tool_use",
    )
    stream = _AsyncMessageStream(events, final_message=final_message)
    client = SimpleNamespace(messages=_Messages(stream))

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent._anthropic_api_key = "test-key"
    agent._anthropic_base_url = None
    agent._oauth_1m_beta_disabled = False
    agent._anthropic_client = client
    agent._anthropic_client_source = ("test-key", None, False)
    agent.log_prefix = ""
    agent._capture_anthropic_response_headers = MagicMock()
    agent._touch_activity = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()
    first_delta = MagicMock()
    stream_activity = MagicMock()

    response = await agent._execute_model_request(
        {"model": "claude-test", "messages": []},
        use_streaming=True,
        on_first_delta=first_delta,
        on_stream_activity=stream_activity,
    )

    assert response is final_message
    assert stream.exited is True
    assert stream_activity.call_count == 5
    assert agent._touch_activity.call_count == 5
    first_delta.assert_called_once_with()
    agent._fire_reasoning_delta.assert_called_once_with("reasoning")
    agent._fire_stream_delta.assert_called_once_with("visible before tool")
    agent._fire_tool_gen_started.assert_called_once_with("terminal")


@pytest.mark.asyncio
async def test_eventless_anthropic_stream_raises_empty_stream_error():
    stream = _AsyncMessageStream([], final_error=AssertionError("no message_start"))
    client = SimpleNamespace(messages=_Messages(stream))

    with pytest.raises(EmptyStreamError, match="empty stream with no events"):
        await create_anthropic_message(
            client,
            {"model": "claude-test", "messages": []},
        )

    assert stream.exited is True


@pytest.mark.asyncio
async def test_contentless_anthropic_stream_raises_empty_stream_error():
    stream = _AsyncMessageStream(
        [SimpleNamespace(type="message_start")],
        final_message=SimpleNamespace(content=[], stop_reason=None),
    )
    client = SimpleNamespace(messages=_Messages(stream))

    with pytest.raises(EmptyStreamError, match="empty stream with no stop_reason"):
        await create_anthropic_message(
            client,
            {"model": "claude-test", "messages": []},
        )

    assert stream.exited is True


@pytest.mark.asyncio
async def test_native_chat_stream_preserves_delta_callback_order():
    chunks = [
        SimpleNamespace(
            id="stream-id", model="test-model", usage=None, choices=[]
        ),
        _chat_chunk(reasoning="reasoning"),
        _chat_chunk(content="visible before tool"),
        _chat_chunk(
            tool_calls=[_chat_tool(0, "call-1", "terminal", '{"cmd":')]
        ),
        _chat_chunk(
            content="hidden after tool",
            tool_calls=[_chat_tool(0, None, None, '"pwd"}')],
            finish="tool_calls",
        ),
    ]
    stream = _AsyncChatStream(chunks)
    agent = _native_chat_agent(stream)
    events = []
    first_delta = MagicMock(side_effect=lambda: events.append("first"))
    stream_activity = MagicMock()
    agent._fire_reasoning_delta = lambda text: events.append(
        ("reasoning", text)
    )
    agent._fire_stream_delta = lambda text: events.append(("text", text))
    agent._fire_tool_gen_started = lambda name: events.append(("tool", name))
    agent.stream_delta_callback = lambda text: events.append(("raw", text))

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
        on_first_delta=first_delta,
        on_stream_activity=stream_activity,
    )

    assert events == [
        "first",
        ("reasoning", "reasoning"),
        ("text", "visible before tool"),
        ("tool", "terminal"),
        ("raw", "hidden after tool"),
    ]
    first_delta.assert_called_once_with()
    assert stream_activity.call_count == len(chunks)
    assert agent._touch_activity.call_count == len(chunks)
    agent._capture_rate_limits.assert_called_once_with(stream.response)
    agent._capture_credits.assert_called_once_with(stream.response)
    agent._check_openrouter_cache_status.assert_called_once_with(
        stream.response
    )
    assert stream.closed is True
    message = response.choices[0].message
    assert message.content == "visible before toolhidden after tool"
    assert message.reasoning_content == "reasoning"
    assert message.tool_calls[0].function.arguments == '{"cmd":"pwd"}'


@pytest.mark.asyncio
async def test_native_chat_stream_preserves_reused_tool_index_and_extra_content():
    chunks = [
        _chat_chunk(
            tool_calls=[
                _chat_tool(
                    0,
                    101,
                    "first_tool",
                    "{}",
                    extra_content={"thought_signature": "sig"},
                )
            ]
        ),
        _chat_chunk(
            tool_calls=[_chat_tool(0, "call-2", "second_tool", "{}")],
            finish="tool_calls",
        ),
    ]
    stream = _AsyncChatStream(chunks)
    agent = _native_chat_agent(stream)
    tool_names = []
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = tool_names.append

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
    )

    tool_calls = response.choices[0].message.tool_calls
    assert [call.id for call in tool_calls] == ["101", "call-2"]
    assert [call.function.name for call in tool_calls] == [
        "first_tool",
        "second_tool",
    ]
    assert tool_calls[0].extra_content == {"thought_signature": "sig"}
    assert tool_names == ["first_tool", "second_tool"]
    assert response.id.startswith("stream-")
    assert response.choices[0].message.content is None
    assert stream.closed is True


@pytest.mark.asyncio
async def test_native_chat_stream_close_failure_preserves_response_and_rebuilds_client():
    class _CloseFailingStream(_AsyncChatStream):
        async def aclose(self):
            self.closed = True
            raise RuntimeError("stream close failed")

    stream = _CloseFailingStream([
        _chat_chunk(content="complete", finish="stop")
    ])
    agent = _native_chat_agent(stream)
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()
    agent._replace_primary_openai_client = AsyncMock(return_value=True)

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
    )

    assert response.choices[0].message.content == "complete"
    assert stream.closed is True
    agent._replace_primary_openai_client.assert_awaited_once_with(
        reason="chat_stream_close_failed"
    )


@pytest.mark.asyncio
async def test_native_chat_stream_close_failure_poisons_unrebuildable_client():
    class _CloseFailingStream(_AsyncChatStream):
        async def aclose(self):
            raise RuntimeError("stream close failed")

    stream = _CloseFailingStream([
        _chat_chunk(content="complete", finish="stop")
    ])
    agent = _native_chat_agent(stream)
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()
    agent._replace_primary_openai_client = AsyncMock(return_value=False)

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
    )

    assert response.choices[0].message.content == "complete"
    assert agent.client is None


@pytest.mark.asyncio
async def test_native_chat_stream_close_finishes_before_cancellation_propagates():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class _SlowCloseStream(_AsyncChatStream):
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            self.closed = True
            close_finished.set()

    stream = _SlowCloseStream([
        _chat_chunk(content="complete", finish="stop")
    ])
    agent = _native_chat_agent(stream)
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()
    agent._replace_primary_openai_client = AsyncMock(return_value=True)

    request = asyncio.create_task(
        agent._execute_model_request(
            {"model": "test-model", "messages": []},
            use_streaming=True,
        )
    )
    await close_started.wait()
    request.cancel()
    await asyncio.sleep(0)
    request.cancel()
    assert request.done() is False

    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert close_finished.is_set()
    assert stream.closed is True
    agent._replace_primary_openai_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_eventless_native_chat_stream_raises_empty_stream_error():
    stream = _AsyncChatStream([])
    agent = _native_chat_agent(stream)
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()

    with pytest.raises(EmptyStreamError, match="empty stream with no finish_reason"):
        await agent._execute_model_request(
            {"model": "test-model", "messages": []},
            use_streaming=True,
        )

    assert stream.closed is True


@pytest.mark.asyncio
async def test_text_stream_without_finish_reason_returns_partial_stub():
    stream = _AsyncChatStream([_chat_chunk(content="partial")])
    agent = _native_chat_agent(stream)
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
    )

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content == "partial"
    assert response.choices[0].finish_reason == "length"
    assert response._dropped_tool_names is None


@pytest.mark.asyncio
async def test_incomplete_tool_stream_without_finish_reason_drops_tool_call():
    stream = _AsyncChatStream(
        [
            _chat_chunk(
                tool_calls=[
                    _chat_tool(0, "call-1", "terminal", "not valid json")
                ]
            )
        ]
    )
    agent = _native_chat_agent(stream)
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()

    response = await agent._execute_model_request(
        {"model": "test-model", "messages": []},
        use_streaming=True,
    )

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].finish_reason == "length"
    assert response._dropped_tool_names == ["terminal"]


@pytest.mark.asyncio
async def test_native_gemini_stream_omits_openai_stream_options():
    stream = _AsyncChatStream([_chat_chunk(content="done", finish="stop")])
    requests = []

    class _Completions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return stream

    agent = _native_chat_agent(stream)
    agent.base_url = "https://generativelanguage.googleapis.com/v1beta"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_stream_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()

    await agent._execute_model_request(
        {"model": "gemini-test", "messages": []},
        use_streaming=True,
    )

    assert requests[0]["stream"] is True
    assert "stream_options" not in requests[0]
