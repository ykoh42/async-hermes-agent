"""Credential pools must never cross provider or custom-endpoint boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.credential_pool import credential_pool_matches_provider
from hermes_cli import runtime_provider as rp


def test_provider_match_requires_exact_non_custom_identity():
    assert credential_pool_matches_provider("deepseek", "deepseek")
    assert not credential_pool_matches_provider("openai-codex", "deepseek")
    assert not credential_pool_matches_provider("", "deepseek")


def test_custom_pool_match_is_scoped_by_endpoint():
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:lab",
    ):
        assert credential_pool_matches_provider(
            "custom:lab", "custom", base_url="https://lab.example/v1"
        )
        assert not credential_pool_matches_provider(
            "custom:other", "custom", base_url="https://lab.example/v1"
        )


@pytest.mark.asyncio
async def test_runtime_ignores_pool_loaded_for_different_provider(monkeypatch):
    entry = SimpleNamespace(
        provider="openai-codex",
        access_token="wrong-token",
        runtime_api_key="wrong-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    async def select():
        return entry

    async def load_pool(_provider):
        return SimpleNamespace(
            provider="openai-codex",
            has_credentials=lambda: True,
            select=select,
        )

    monkeypatch.setattr(rp, "load_pool", load_pool)
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: "deepseek")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {"provider": "deepseek", "default": "deepseek-chat"},
    )
    monkeypatch.setattr(
        rp.auth_mod,
        "resolve_api_key_provider_credentials_async",
        AsyncMock(return_value={
            "provider": "deepseek",
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "source": "env",
        }),
    )

    resolved = await rp.resolve_runtime_provider(requested="deepseek")

    assert resolved["provider"] == "deepseek"
    assert resolved["api_key"] == "deepseek-key"
    assert resolved["base_url"] == "https://api.deepseek.com/v1"
