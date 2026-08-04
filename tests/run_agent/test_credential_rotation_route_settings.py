"""Native async credential rotation contracts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


class _NativeClient:
    def __init__(self):
        self._platform = None
        self.close = AsyncMock()


@pytest.mark.asyncio
async def test_credential_rotation_rebuilds_natively_without_sync_settings_io():
    """A pool rotation replaces the client without re-reading config.yaml.

    Route-scoped TLS and user-header policy is resolved when the async runtime
    is initialized; the retry path must not perform synchronous settings I/O
    merely to rotate an API credential.
    """
    native_client = _NativeClient()
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=native_client) as async_openai,
        patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=AssertionError("credential rotation must not read settings"),
        ),
    ):
        agent = AIAgent(
            provider="custom",
            model="shared-model",
            api_key="old-key",
            base_url="https://a.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._runtime_config_loaded = True
        entry = SimpleNamespace(
            id="pool-entry-b",
            runtime_api_key="new-key",
            access_token="",
            runtime_base_url="https://b.example/v1",
            base_url="https://b.example/v1",
        )

        await agent._swap_credential(entry)

    assert agent.api_key == "new-key"
    assert agent.base_url == "https://b.example/v1"
    assert agent._credential_pool_entry_id == "pool-entry-b"
    assert agent.client is native_client
    assert agent.client._platform == "Unknown"
    async_openai.assert_called_once()
    await agent.close()


@pytest.mark.asyncio
async def test_credential_rotation_drops_previous_route_headers():
    native_client = _NativeClient()
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=native_client),
    ):
        agent = AIAgent(
            provider="custom",
            model="shared-model",
            api_key="old-key",
            base_url="https://a.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._runtime_config_loaded = True
        agent._client_kwargs["default_headers"] = {"Authorization": "old-secret"}
        entry = SimpleNamespace(
            id="pool-entry-b",
            runtime_api_key="new-key",
            access_token="",
            runtime_base_url="https://b.example/v1",
            base_url="https://b.example/v1",
        )

        await agent._swap_credential(entry)

    assert "default_headers" not in agent._client_kwargs
    await agent.close()
