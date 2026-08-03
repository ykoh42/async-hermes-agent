"""Tests for agent.auxiliary_client._try_custom_endpoint's anthropic_messages branch.

When a user configures a custom endpoint with ``api_mode: anthropic_messages``
(e.g. MiniMax, Zhipu GLM, LiteLLM in Anthropic-proxy mode), auxiliary tasks
(compression, web_extract, session_search, title generation) must use the
native Anthropic transport rather than being silently downgraded to an
OpenAI-wire client that speaks the wrong protocol.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _install_anthropic_adapter_mocks():
    """Patch build_anthropic_client so the test doesn't need the SDK."""
    fake_client = MagicMock(name="anthropic_client")
    return patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_client,
    ), fake_client


@pytest.mark.asyncio
async def test_custom_endpoint_anthropic_messages_builds_anthropic_wrapper():
    """api_mode=anthropic_messages → returns AnthropicAuxiliaryClient, not OpenAI."""
    from agent.auxiliary_client import _try_custom_endpoint, AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._normalize_main_runtime",
        return_value={
            "base_url": "https://api.minimax.io/anthropic",
            "api_key": "minimax-key",
            "api_mode": "anthropic_messages",
        },
    ), patch(
        "agent.auxiliary_client._read_main_model_for_aux_async",
        new=AsyncMock(return_value="claude-sonnet-4-6"),
    ):
        adapter_patch, fake_client = _install_anthropic_adapter_mocks()
        with adapter_patch:
            client, model = await _try_custom_endpoint()

    assert isinstance(client, AnthropicAuxiliaryClient), (
        "Custom endpoint with api_mode=anthropic_messages must return the "
        f"native Anthropic wrapper, got {type(client).__name__}"
    )
    assert model == "claude-sonnet-4-6"
    # Wrapper should NOT be marked as OAuth — third-party endpoints are
    # always API-key authenticated.
    assert client.api_key == "minimax-key"
    assert client.base_url == "https://api.minimax.io/anthropic"


@pytest.mark.asyncio
async def test_custom_endpoint_anthropic_messages_falls_back_when_sdk_missing():
    """Graceful degradation when anthropic SDK is unavailable."""
    from agent.auxiliary_client import _try_custom_endpoint

    import_error = ImportError("anthropic package not installed")

    with patch(
        "agent.auxiliary_client._normalize_main_runtime",
        return_value={
            "base_url": "https://api.minimax.io/anthropic",
            "api_key": "k",
            "api_mode": "anthropic_messages",
        },
    ), patch(
        "agent.auxiliary_client._read_main_model_for_aux_async",
        new=AsyncMock(return_value="claude-sonnet-4-6"),
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=import_error,
    ):
        client, model = await _try_custom_endpoint()

    # Should fall back to an OpenAI-wire client rather than returning
    # (None, None) — the tool still needs to do *something*.
    assert client is not None
    assert model == "claude-sonnet-4-6"
    # OpenAI client, not AnthropicAuxiliaryClient.
    from agent.auxiliary_client import AnthropicAuxiliaryClient
    assert not isinstance(client, AnthropicAuxiliaryClient)


@pytest.mark.asyncio
async def test_custom_endpoint_chat_completions_still_uses_openai_wire():
    """Regression: default path (no api_mode) must remain OpenAI client."""
    from agent.auxiliary_client import _try_custom_endpoint, AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._normalize_main_runtime",
        return_value={
            "base_url": "https://api.example.com/v1",
            "api_key": "key",
        },
    ), patch(
        "agent.auxiliary_client._read_main_model_for_aux_async",
        new=AsyncMock(return_value="my-model"),
    ):
        client, model = await _try_custom_endpoint()

    assert client is not None
    assert model == "my-model"
    assert not isinstance(client, AnthropicAuxiliaryClient)
