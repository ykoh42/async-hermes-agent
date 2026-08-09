"""Responses API stream contract tests."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.codex_runtime import run_codex_stream


def test_run_codex_stream_preserves_upstream_signature():
    parameters = list(inspect.signature(run_codex_stream).parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "agent",
        "api_kwargs",
        "client",
        "on_first_delta",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    )
    assert [parameter.default for parameter in parameters] == [
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        None,
        None,
    ]


class _Stream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def iterate():
            for event in self._events:
                yield event

        return iterate()

    async def aclose(self):
        return None


class _Responses:
    def __init__(self, events):
        self.events = events

    async def create(self, **kwargs):
        assert kwargs["stream"] is True
        return _Stream(self.events)


class _Client:
    def __init__(self, events):
        self.responses = _Responses(events)


@pytest.mark.asyncio
async def test_codex_responses_stream_reads_async_iterator():
    events = [
        {"type": "response.output_item.added", "item": {"type": "message", "phase": "final"}},
        {"type": "response.output_text.delta", "delta": "hello"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        },
        {
            "type": "response.completed",
            "response": {"id": "r1", "status": "completed", "usage": {}},
        },
    ]
    deltas = []
    agent = SimpleNamespace(
        _fire_stream_delta=deltas.append,
        _fire_reasoning_delta=lambda _text: None,
    )

    response = await run_codex_stream(
        agent,
        {"model": "gpt-test"},
        client=_Client(events),
    )

    assert response.status == "completed"
    assert response.output_text == "hello"
    assert deltas == ["hello"]


@pytest.mark.asyncio
async def test_codex_stream_close_failure_preserves_response_and_rebuilds_client():
    events = [
        {"type": "response.output_text.delta", "delta": "complete"},
        {
            "type": "response.completed",
            "response": {"id": "r1", "status": "completed", "usage": {}},
        },
    ]

    class _CloseFailingStream(_Stream):
        async def aclose(self):
            raise RuntimeError("stream close failed")

    class _CloseFailingResponses:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return _CloseFailingStream(events)

    client = SimpleNamespace(responses=_CloseFailingResponses())
    agent = SimpleNamespace(
        client=client,
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _replace_primary_openai_client=AsyncMock(return_value=True),
    )

    response = await run_codex_stream(
        agent,
        {"model": "gpt-test"},
        client=client,
    )

    assert response.output_text == "complete"
    agent._replace_primary_openai_client.assert_awaited_once_with(
        reason="codex_stream_close_failed"
    )


@pytest.mark.asyncio
async def test_codex_stream_close_failure_poisons_unrebuildable_client():
    events = [{"type": "response.output_text.delta", "delta": "complete"}]

    class _CloseFailingStream(_Stream):
        async def aclose(self):
            raise RuntimeError("stream close failed")

    class _CloseFailingResponses:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return _CloseFailingStream(events)

    client = SimpleNamespace(responses=_CloseFailingResponses())
    agent = SimpleNamespace(
        client=client,
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _replace_primary_openai_client=AsyncMock(return_value=False),
    )

    response = await run_codex_stream(
        agent,
        {"model": "gpt-test"},
        client=client,
    )

    assert response.output_text == "complete"
    assert agent.client is None


@pytest.mark.asyncio
async def test_codex_stream_close_finishes_before_cancellation_propagates():
    events = [{"type": "response.output_text.delta", "delta": "complete"}]
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class _SlowCloseStream(_Stream):
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    class _SlowCloseResponses:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return _SlowCloseStream(events)

    client = SimpleNamespace(responses=_SlowCloseResponses())
    agent = SimpleNamespace(
        client=client,
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _replace_primary_openai_client=AsyncMock(return_value=True),
    )

    request = asyncio.create_task(
        run_codex_stream(
            agent,
            {"model": "gpt-test"},
            client=client,
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
    agent._replace_primary_openai_client.assert_not_awaited()
