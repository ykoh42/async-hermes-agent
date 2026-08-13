"""Profile isolation for fallback-model credential environment reads."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.chat_completion_helpers import try_activate_fallback
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


def _fallback_agent(*, base_url: str, key_env: str | None = None):
    fallback = {
        "provider": "custom",
        "model": "fallback-model",
        "base_url": base_url,
    }
    if key_env is not None:
        fallback["key_env"] = key_env

    agent = SimpleNamespace(
        provider="custom",
        requested_provider="custom",
        model="primary-model",
        api_key="primary-key",
        base_url="https://primary.example/v1",
        api_mode="chat_completions",
        client=object(),
        _primary_runtime={"provider": "custom"},
        _fallback_chain=[fallback],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(),
        _deferred_provider_runtime=None,
        _transport_cache={},
        _credential_pool=None,
        _credential_pool_entry_id=None,
        _config_context_length=None,
        context_compressor=None,
        _cached_system_prompt=None,
        _client_kwargs={},
        _is_azure_openai_url=lambda _url: False,
        _is_direct_openai_url=lambda _url: False,
        _provider_model_requires_responses_api=lambda *_args, **_kwargs: False,
        _anthropic_prompt_cache_policy=lambda **_kwargs: (False, False),
        _buffer_status=lambda _message: None,
    )

    async def ensure_provider_runtime():
        await asyncio.sleep(0)
        pending = agent._deferred_provider_runtime
        agent.provider = pending["provider"]
        agent.requested_provider = pending["provider"]
        agent.model = pending["model"]
        agent.api_key = pending["api_key"]
        agent.base_url = pending["base_url"]
        agent.api_mode = "chat_completions"
        agent.client = object()
        agent._deferred_provider_runtime = None

    agent._ensure_provider_runtime = ensure_provider_runtime
    agent._try_activate_fallback = AsyncMock(return_value=False)
    return agent


@pytest.mark.asyncio
async def test_concurrent_fallback_key_env_reads_active_profile(monkeypatch):
    monkeypatch.setenv("CUSTOM_FALLBACK_KEY", "process-key")
    set_multiplex_active(True)

    async def activate(profile_key: str):
        token = set_secret_scope({"CUSTOM_FALLBACK_KEY": profile_key})
        agent = _fallback_agent(
            base_url="https://fallback.example/v1",
            key_env="CUSTOM_FALLBACK_KEY",
        )
        try:
            assert await try_activate_fallback(agent) is True
            return agent.api_key
        finally:
            reset_secret_scope(token)

    with (
        patch(
            "agent.credential_pool.load_pool",
            AsyncMock(return_value=None),
        ),
        patch(
            "agent.chat_completion_helpers._reset_stale_streak",
            lambda _agent: None,
        ),
    ):
        key_a, key_b = await asyncio.gather(
            activate("profile-a-key"),
            activate("profile-b-key"),
        )

    assert (key_a, key_b) == ("profile-a-key", "profile-b-key")


@pytest.mark.asyncio
async def test_ollama_cloud_fallback_uses_scoped_key_and_preserves_empty(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "process-key")
    set_multiplex_active(True)

    async def activate(secrets: dict[str, str]):
        token = set_secret_scope(secrets)
        agent = _fallback_agent(base_url="https://ollama.com/v1")
        try:
            assert await try_activate_fallback(agent) is True
            return agent.api_key
        finally:
            reset_secret_scope(token)

    with (
        patch(
            "agent.credential_pool.load_pool",
            AsyncMock(return_value=None),
        ),
        patch(
            "agent.chat_completion_helpers._reset_stale_streak",
            lambda _agent: None,
        ),
    ):
        scoped, empty = await asyncio.gather(
            activate({"OLLAMA_API_KEY": "profile-key"}),
            activate({"OLLAMA_API_KEY": ""}),
        )

    assert scoped == "profile-key"
    assert empty is None


@pytest.mark.asyncio
async def test_unscoped_fallback_key_env_fails_closed(monkeypatch):
    monkeypatch.setenv("CUSTOM_FALLBACK_KEY", "process-key")
    set_multiplex_active(True)
    agent = _fallback_agent(
        base_url="https://fallback.example/v1",
        key_env="CUSTOM_FALLBACK_KEY",
    )

    with pytest.raises(UnscopedSecretError, match="CUSTOM_FALLBACK_KEY"):
        await try_activate_fallback(agent)

    agent._try_activate_fallback.assert_not_awaited()
