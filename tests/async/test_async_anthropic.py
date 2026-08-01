"""Native async Anthropic transport tests."""

import pytest

from agent.anthropic_adapter import create_anthropic_message_async
from run_agent import AIAgent


class _Stream:
    response = object()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def __aiter__(self):
        async def events():
            yield {"type": "message_start"}
            yield {"type": "content_block_delta", "delta": {"text": "ok"}}

        return events()

    async def get_final_message(self):
        return {"content": "ok"}


class _Messages:
    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _Stream()


class _Client:
    def __init__(self):
        self.messages = _Messages()


@pytest.mark.asyncio
async def test_anthropic_stream_is_consumed_without_a_thread():
    events = []
    response = await create_anthropic_message_async(
        _Client(),
        {"model": "claude-test", "messages": [], "stream": True},
        on_stream_event=events.append,
    )

    assert response == {"content": "ok"}
    assert len(events) == 2


@pytest.mark.asyncio
async def test_agent_uses_native_async_messages_dispatch(monkeypatch):
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.model = "claude-test"
    agent._anthropic_api_key = "test-key"
    agent._anthropic_base_url = "https://example.invalid"
    agent.log_prefix = ""
    agent._capture_anthropic_response_headers = lambda _response: None

    built = []

    def fake_build(*args, **kwargs):
        built.append(kwargs)
        return object()

    async def fake_create(*args, **kwargs):
        return {"native_async": True}

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", fake_build)
    monkeypatch.setattr("agent.anthropic_adapter.create_anthropic_message_async", fake_create)

    result = await agent._async_execute_model_request(
        {"model": "claude-test", "messages": []}, use_streaming=False
    )

    assert result == {"native_async": True}
    assert built[0]["async_mode"] is True
