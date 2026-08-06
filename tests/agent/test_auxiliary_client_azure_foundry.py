"""Native-async Azure Foundry provider coverage."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def patch_load_config(monkeypatch):
    def apply(model_config):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(return_value={"model": model_config}),
        )

    return apply


@pytest.mark.asyncio
async def test_static_key_builds_async_openai_client(monkeypatch, patch_load_config):
    from agent.auxiliary_client import _try_azure_foundry
    from openai import AsyncOpenAI

    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
    patch_load_config(
        {
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4o",
        }
    )

    client, resolved_model = await _try_azure_foundry(model="gpt-4o")

    assert isinstance(client, AsyncOpenAI)
    assert client.api_key == "sk-azure-static-key"
    assert resolved_model == "gpt-4o"


@pytest.mark.asyncio
async def test_missing_static_key_returns_none(monkeypatch, patch_load_config):
    from agent.auxiliary_client import _try_azure_foundry

    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    patch_load_config(
        {
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4o",
        }
    )

    assert await _try_azure_foundry(model="gpt-4o") == (None, None)


@pytest.mark.asyncio
async def test_entra_id_builds_async_openai_token_provider(monkeypatch, patch_load_config):
    from agent.auxiliary_client import _try_azure_foundry

    async def token_provider():
        return "entra-token"

    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    patch_load_config(
        {
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "auth_mode": "entra_id",
            "default": "gpt-4o",
        }
    )

    monkeypatch.setattr(
        "agent.azure_identity_adapter.build_token_provider",
        AsyncMock(return_value=token_provider),
    )

    client, resolved_model = await _try_azure_foundry(model="gpt-4o")

    assert client._api_key_provider is token_provider
    assert await client._api_key_provider() == "entra-token"
    assert resolved_model == "gpt-4o"
    await client.close()


@pytest.mark.asyncio
async def test_public_resolver_uses_azure_static_key_branch(
    monkeypatch,
    patch_load_config,
):
    from agent.auxiliary_client import resolve_provider_client
    from openai import AsyncOpenAI

    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-azure-static-key")
    patch_load_config(
        {
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "default": "gpt-4o",
        }
    )

    client, resolved_model = await resolve_provider_client("azure-foundry", "gpt-4o")

    assert isinstance(client, AsyncOpenAI)
    assert resolved_model == "gpt-4o"


@pytest.mark.asyncio
async def test_direct_agent_initializes_entra_provider_at_first_await(monkeypatch):
    from run_agent import AIAgent

    async def token_provider():
        return "entra-token"

    monkeypatch.setattr(
        "agent.azure_identity_adapter.build_token_provider",
        AsyncMock(return_value=token_provider),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(
            return_value={
                "model": {
                    "provider": "azure-foundry",
                    "base_url": "https://r.openai.azure.com/openai/v1",
                    "auth_mode": "entra_id",
                    "default": "gpt-4o",
                }
            }
        ),
    )
    native_client = SimpleNamespace(aclose=AsyncMock(), _platform=None)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=native_client) as client_factory,
    ):
        agent = AIAgent(
            provider="azure-foundry",
            model="gpt-4o",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        client_factory.assert_not_called()

        assert await agent._ensure_provider_runtime() is True

    assert agent.api_key is token_provider
    assert agent.client is native_client
    assert client_factory.call_args.kwargs["api_key"] is token_provider
    await agent.close()
