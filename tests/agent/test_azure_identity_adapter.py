"""Tests for the native-async Microsoft Entra ID adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def adapter(monkeypatch):
    from agent import azure_identity_adapter as module

    module._credential_caches.clear()
    module._credential_leases.clear()
    module._issued_providers.clear()

    class Credential:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.calls = 0
            self.instances.append(self)

        async def get_token(self, scope):
            await asyncio.sleep(0)
            self.calls += 1
            return SimpleNamespace(token="entra-token", expires_on=1234)

        async def close(self):
            self.closed = True

    def get_bearer_token_provider(credential, scope):
        async def provide():
            return (await credential.get_token(scope)).token

        return provide

    identity = SimpleNamespace(
        DefaultAzureCredential=Credential,
        get_bearer_token_provider=get_bearer_token_provider,
    )
    monkeypatch.setattr(module, "_require_azure_identity", lambda: identity)
    monkeypatch.setattr(module, "has_azure_identity_installed", lambda: True)
    return module, Credential


@pytest.mark.asyncio
async def test_token_provider_is_coroutine_and_credential_is_loop_cached(adapter):
    module, credential_type = adapter
    config = module.EntraIdentityConfig(scope="https://ai.azure.com/.default")

    first = await module.build_token_provider(config=config)
    second = await module.build_token_provider(config=config)

    assert await first() == "entra-token"
    assert await second() == "entra-token"
    assert len(credential_type.instances) == 1
    await module._release_token_provider(first)
    await module._release_token_provider(second)


@pytest.mark.asyncio
async def test_reset_credential_cache_awaits_close(adapter):
    module, credential_type = adapter
    await module.build_token_provider()

    await module.reset_credential_cache()

    assert credential_type.instances[0].closed is True
    assert not module._credential_caches


@pytest.mark.asyncio
async def test_shared_credential_closes_after_last_provider_lease(adapter):
    module, credential_type = adapter
    first = await module.build_token_provider()
    second = await module.build_token_provider()

    await module._release_token_provider(first)
    assert credential_type.instances[0].closed is False

    await module._release_token_provider(second)
    assert credential_type.instances[0].closed is True
    await module._release_token_provider(second)


@pytest.mark.asyncio
async def test_auxiliary_client_close_releases_credential_lease(adapter):
    module, credential_type = adapter
    from agent.auxiliary_client import _close_cached_client, _create_openai_client

    provider = await module.build_token_provider()
    client = await _create_openai_client(
        api_key=provider,
        base_url="https://foundry.invalid/v1",
        config={},
    )

    await _close_cached_client(client)

    assert credential_type.instances[0].closed is True


@pytest.mark.asyncio
async def test_async_bearer_hook_replaces_conflicting_headers(adapter):
    module, _ = adapter
    provider = await module.build_token_provider()

    async def endpoint(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer entra-token"
        assert "api-key" not in request.headers
        assert "x-api-key" not in request.headers
        return httpx.Response(200, json={"ok": True})

    client = await module.build_bearer_http_client(
        provider,
        transport=httpx.MockTransport(endpoint),
        headers={"Authorization": "Bearer stale", "api-key": "stale"},
    )
    try:
        response = await client.get("https://foundry.invalid/test")
    finally:
        await client.aclose()
        await module._release_token_provider(provider)

    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_credential_probe_uses_async_timeout(adapter, monkeypatch):
    module, _ = adapter
    started = asyncio.Event()

    class SlowCredential:
        async def get_token(self, scope):
            started.set()
            await asyncio.Event().wait()

    async def build(_config):
        return SlowCredential()

    monkeypatch.setattr(module, "build_credential", build)

    task = asyncio.create_task(
        module.has_azure_identity_credentials(timeout_seconds=0.01)
    )
    await started.wait()
    assert await task is False


@pytest.mark.asyncio
async def test_external_cancellation_is_not_swallowed(adapter, monkeypatch):
    module, _ = adapter
    started = asyncio.Event()

    class SlowCredential:
        async def get_token(self, scope):
            started.set()
            await asyncio.Event().wait()

    async def build(_config):
        return SlowCredential()

    monkeypatch.setattr(module, "build_credential", build)
    task = asyncio.create_task(module.has_azure_identity_credentials())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_describe_active_credential_preserves_shape(adapter):
    module, _ = adapter

    info = await module.describe_active_credential(timeout_seconds=1)

    assert info["ok"] is True
    assert info["scope"] == module.SCOPE_AI_AZURE_DEFAULT
    assert info["expires_on"] == 1234
    assert isinstance(info["env_sources"], list)


@pytest.mark.asyncio
async def test_anthropic_foundry_uses_async_bearer_client(adapter, monkeypatch):
    module, _ = adapter
    from agent import anthropic_adapter

    provider = await module.build_token_provider()

    class AsyncAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.api_key = "environment-key"

    monkeypatch.setattr(
        anthropic_adapter,
        "_get_anthropic_sdk",
        lambda: SimpleNamespace(AsyncAnthropic=AsyncAnthropic),
    )

    client = await anthropic_adapter.build_anthropic_client(
        provider,
        "https://resource.services.ai.azure.com/anthropic",
    )
    try:
        assert isinstance(client.kwargs["http_client"], httpx.AsyncClient)
        assert client.kwargs["auth_token"] == "entra-id-bearer-via-http-hook"
        assert client.kwargs["default_query"] == {"api-version": "2025-04-15"}
        assert client.api_key is None
    finally:
        await client.kwargs["http_client"].aclose()
        await module._release_token_provider(provider)
