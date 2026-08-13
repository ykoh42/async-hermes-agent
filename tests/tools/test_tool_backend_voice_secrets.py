"""Native-async parity tests for shared STT/TTS secret resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import secret_scope
from tools import tool_backend_helpers


@pytest.fixture(autouse=True)
def reset_scope():
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)
    secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_config_value_has_highest_precedence(monkeypatch):
    monkeypatch.setenv("VOICE_TEST_KEY", "process-key")
    assert await tool_backend_helpers.resolve_provider_secret(
        "VOICE_TEST_KEY",
        "voice-test",
        config_value=" config-key ",
    ) == "config-key"


@pytest.mark.asyncio
async def test_async_env_getter_precedes_credential_pool(monkeypatch):
    async def env_getter(name):
        assert name == "VOICE_TEST_KEY"
        return "dotenv-key"

    async def pool_should_not_load(provider):
        raise AssertionError(provider)

    monkeypatch.delenv("VOICE_TEST_KEY", raising=False)
    monkeypatch.setattr("agent.credential_pool.load_pool", pool_should_not_load)
    assert await tool_backend_helpers.resolve_provider_secret(
        "VOICE_TEST_KEY",
        "voice-test",
        env_getter=env_getter,
    ) == "dotenv-key"


@pytest.mark.asyncio
async def test_pool_fallback_awaits_native_async_pool(monkeypatch):
    entry = SimpleNamespace(runtime_api_key="pool-key", access_token="")

    class Pool:
        def has_credentials(self):
            return True

        async def peek(self):
            return entry

    seen = []

    async def load_pool(provider):
        seen.append(provider)
        return Pool()

    async def empty_env(name):
        del name
        return ""

    monkeypatch.delenv("VOICE_TEST_KEY", raising=False)
    monkeypatch.setattr("agent.credential_pool.load_pool", load_pool)
    assert await tool_backend_helpers.resolve_provider_secret(
        "VOICE_TEST_KEY",
        "voice-test",
        env_getter=empty_env,
    ) == "pool-key"
    assert seen == ["voice-test"]


@pytest.mark.asyncio
async def test_multiplex_scope_is_authoritative_and_isolated(monkeypatch):
    monkeypatch.setenv("VOICE_TEST_KEY", "other-profile-key")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"VOICE_TEST_KEY": "profile-key"})
    try:
        assert await tool_backend_helpers.resolve_provider_secret(
            "VOICE_TEST_KEY", "voice-test"
        ) == "profile-key"
    finally:
        secret_scope.reset_secret_scope(token)

    assert await tool_backend_helpers.resolve_provider_secret(
        "VOICE_TEST_KEY", "voice-test"
    ) == ""

    token = secret_scope.set_secret_scope({})
    try:
        assert await tool_backend_helpers.resolve_provider_secret(
            "VOICE_TEST_KEY", "voice-test"
        ) == ""
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_openai_audio_key_preference(monkeypatch):
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "voice-key")
    monkeypatch.setenv("OPENAI_API_KEY", "general-key")
    assert await tool_backend_helpers.resolve_openai_audio_api_key() == "voice-key"
