"""Responses API stream contract tests."""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import run_codex_stream


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
