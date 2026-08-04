"""Native-async Azure Foundry auxiliary-provider coverage."""

from unittest.mock import AsyncMock

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
async def test_entra_id_fails_fast_without_sync_fallback(monkeypatch, patch_load_config):
    from agent.auxiliary_client import _try_azure_foundry

    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    patch_load_config(
        {
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "auth_mode": "entra_id",
            "default": "gpt-4o",
        }
    )

    assert await _try_azure_foundry(model="gpt-4o") == (None, None)


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
