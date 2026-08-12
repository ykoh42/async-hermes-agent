from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from run_agent import AIAgent


def _make_agent(*, api_mode: str):
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = api_mode
    agent.provider = "copilot"
    agent.base_url = "https://api.githubcopilot.com"
    return agent


@pytest.mark.asyncio
async def test_request_client_adds_copilot_vision_header_for_native_image_payload():
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    agent = _make_agent(api_mode="chat_completions")
    agent._ensure_primary_openai_client = AsyncMock(
        return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions()),
        )
    )
    api_kwargs = {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ],
        "extra_headers": {"x-initiator": "user"},
    }

    await agent._execute_model_request(api_kwargs)

    assert captured["extra_headers"] == {
        "x-initiator": "user",
        "Copilot-Vision-Request": "true",
    }
    assert api_kwargs["extra_headers"] == {"x-initiator": "user"}


@pytest.mark.asyncio
async def test_copilot_responses_input_adds_vision_header_without_replacing_headers():
    captured = {}

    class _Responses:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output=[])

    agent = _make_agent(api_mode="codex_responses")
    agent._ensure_primary_openai_client = AsyncMock(
        return_value=SimpleNamespace(responses=_Responses())
    )
    api_kwargs = {
        "model": "gpt-5.4",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this"},
                    {"type": "input_image", "image_url": "https://example.test/a.png"},
                ],
            }
        ],
        "extra_headers": {"x-client-request-id": "request-id"},
    }

    await agent._execute_model_request(api_kwargs)

    assert captured["extra_headers"] == {
        "x-client-request-id": "request-id",
        "Copilot-Vision-Request": "true",
    }


@pytest.mark.asyncio
async def test_non_copilot_image_request_does_not_add_copilot_header():
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    agent = _make_agent(api_mode="chat_completions")
    agent.provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    agent._ensure_primary_openai_client = AsyncMock(
        return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions()),
        )
    )

    await agent._execute_model_request(
        {
            "model": "gpt-5.4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "image"}},
                    ],
                }
            ],
        }
    )

    assert "extra_headers" not in captured


@pytest.mark.asyncio
async def test_completed_response_disables_streaming_and_fires_native_callbacks():
    completed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    reasoning_content="reasoning",
                    reasoning=None,
                    content="completed response",
                ),
                finish_reason="stop",
            )
        ]
    )

    class _Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return completed

    agent = _make_agent(api_mode="chat_completions")
    agent.model = "gpt-5.4"
    agent._disable_streaming = False
    agent._ensure_primary_openai_client = AsyncMock(
        return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions()),
        )
    )
    reasoning = []
    text = []
    first_delta = []
    delivered = []
    agent._fire_reasoning_delta = reasoning.append
    agent._fire_stream_delta = text.append

    result = await agent._execute_model_request(
        {"model": "gpt-5.4", "messages": []},
        use_streaming=True,
        on_first_delta=lambda: first_delta.append(True),
        _on_stream_text=lambda: delivered.append(True),
    )

    assert result is completed
    assert agent._disable_streaming is True
    assert first_delta == [True]
    assert reasoning == ["reasoning"]
    assert text == ["completed response"]
    assert delivered == [True]
