from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agent import background_review as review


class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model
        self._credential_pool: Any = None
        self.request_overrides = {}
        self.max_tokens: int | None = None
        self.acp_command = None
        self.acp_args = []

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


@pytest.mark.asyncio
async def test_auto_route_inherits_parent_and_downgrades_codex_app_server():
    config = {
        "auxiliary": {
            "background_review": {"provider": "auto", "model": ""}
        }
    }
    with patch(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value=config),
    ):
        runtime = await review._resolve_review_runtime(_FakeAgent())

    assert runtime["routed"] is False
    assert runtime["provider"] == "openai-codex"
    assert runtime["model"] == "gpt-5.5"
    assert runtime["api_mode"] == "codex_responses"


@pytest.mark.asyncio
async def test_different_model_route_awaits_native_provider_resolution():
    config = {
        "auxiliary": {
            "background_review": {
                "provider": "openrouter",
                "model": "google/gemini-3-flash-preview",
            }
        }
    }
    resolved = {
        "provider": "openrouter",
        "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "credential_pool": "routed-pool",
        "request_overrides": {"extra_body": {"store": False}},
        "max_output_tokens": 2048,
    }
    resolver = AsyncMock(return_value=resolved)
    with (
        patch(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(return_value=config),
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            resolver,
        ),
    ):
        runtime = await review._resolve_review_runtime(_FakeAgent())

    resolver.assert_awaited_once()
    assert runtime["routed"] is True
    assert runtime["model"] == "google/gemini-3-flash-preview"
    assert runtime["api_key"] == "or-key"
    assert runtime["credential_pool"] == "routed-pool"
    assert runtime["request_overrides"] == {"extra_body": {"store": False}}
    assert runtime["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_route_resolution_failure_falls_back_to_parent():
    config = {
        "auxiliary": {
            "background_review": {
                "provider": "openrouter",
                "model": "other-model",
            }
        }
    }
    with (
        patch(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(return_value=config),
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            AsyncMock(side_effect=RuntimeError("unavailable")),
        ),
    ):
        runtime = await review._resolve_review_runtime(_FakeAgent())

    assert runtime["routed"] is False
    assert runtime["provider"] == "openai-codex"


def _message(role, content, tool_calls=None):
    message = {"role": role, "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def test_routed_digest_keeps_tail_and_never_opens_on_tool_result():
    messages = []
    for index in range(40):
        messages.append(_message("user", f"user-{index}"))
        messages.append(
            _message(
                "assistant",
                "",
                tool_calls=[{"function": {"name": "terminal"}}],
            )
        )
        messages.append({"role": "tool", "content": f"result-{index}"})

    digest = review._digest_history(messages, tail=2)

    assert digest[0]["role"] == "user"
    assert digest[0]["content"].startswith("[Earlier conversation digest")
    assert digest[1]["role"] != "tool"
    assert digest[-1] == messages[-1]


def test_routed_digest_records_old_tool_names():
    messages = [
        _message("user", "do the thing"),
        _message(
            "assistant",
            "",
            tool_calls=[
                {"function": {"name": "skill_view"}},
                {"function": {"name": "patch"}},
            ],
        ),
        *[_message("user", f"tail-{index}") for index in range(30)],
    ]

    digest = review._digest_history(messages, tail=10)[0]["content"]

    assert "USER: do the thing" in digest
    assert "tools: skill_view, patch" in digest
