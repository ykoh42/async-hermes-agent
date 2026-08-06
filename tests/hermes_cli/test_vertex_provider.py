"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, alias canonicalization,
resolve_runtime_provider(vertex) minting an OAuth token, and the friendly
AuthError when credentials can't be resolved. No network calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest








@pytest.mark.asyncio
async def test_resolve_runtime_provider_raises_autherror_when_unresolved(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    async def unresolved():
        return None, None

    monkeypatch.setattr(va, "get_vertex_config", unresolved)
    with pytest.raises(AuthError) as exc:
        await rp.resolve_runtime_provider(requested="vertex")
    msg = str(exc.value)
    assert "OAuth2" in msg
    assert "not a static API key" in msg


@pytest.mark.asyncio
async def test_resolve_runtime_provider_returns_native_vertex_runtime(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    async def resolved():
        return "vertex-token", "https://aiplatform.googleapis.com/v1beta1"

    monkeypatch.setattr(va, "get_vertex_config", resolved)

    runtime = await rp.resolve_runtime_provider(requested="vertex")

    assert runtime == {
        "provider": "vertex",
        "api_mode": "chat_completions",
        "base_url": "https://aiplatform.googleapis.com/v1beta1",
        "api_key": "vertex-token",
        "source": "vertex-oauth",
        "requested_provider": "vertex",
    }


@pytest.mark.asyncio
async def test_direct_agent_initializes_vertex_at_first_await(monkeypatch):
    import agent.vertex_adapter as va
    from run_agent import AIAgent

    async def resolved():
        return "vertex-token", "https://aiplatform.googleapis.com/v1beta1"

    monkeypatch.setattr(va, "get_vertex_config", resolved)
    native_client = SimpleNamespace(aclose=AsyncMock(), _platform=None)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=native_client) as client_factory,
    ):
        agent = AIAgent(
            provider="vertex",
            model="google/gemini-3-flash-preview",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.client is None
        client_factory.assert_not_called()

        assert await agent._ensure_provider_runtime() is True

    assert agent.provider == "vertex"
    assert agent.api_key == "vertex-token"
    assert agent.base_url == "https://aiplatform.googleapis.com/v1beta1"
    assert agent.client is native_client
    await agent.close()


@pytest.mark.asyncio
async def test_vertex_refresh_rebuilds_runtime_with_same_public_method(monkeypatch):
    import agent.vertex_adapter as va
    from run_agent import AIAgent

    async def resolved():
        return "fresh-token", "https://aiplatform.googleapis.com/v1beta1/"

    monkeypatch.setattr(va, "get_vertex_config", resolved)
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="vertex",
        model="google/gemini-3-flash-preview",
        _provider_request_timeout=10,
        _provider_stale_timeout=20,
        _ensure_provider_runtime=AsyncMock(return_value=True),
    )

    assert await AIAgent._try_refresh_vertex_client_credentials(agent) is True
    assert agent._deferred_provider_runtime == {
        "provider": "vertex",
        "model": "google/gemini-3-flash-preview",
        "api_key": "fresh-token",
        "base_url": "https://aiplatform.googleapis.com/v1beta1",
        "api_mode": "chat_completions",
        "request_timeout": 10,
        "stale_timeout": 20,
        "update_primary": False,
    }
    agent._ensure_provider_runtime.assert_awaited_once()


def test_vertex_registered_in_provider_registry():
    """PROVIDER_REGISTRY (hermes_cli.auth) is what agent/auxiliary_client.py's
    resolve_provider_client() looks up before dispatching on auth_type. Without
    an entry here, the ``elif pconfig.auth_type == "vertex":`` branch there is
    unreachable dead code — every auxiliary Vertex call (vision, title
    generation, MoA reference/aggregator slots, ...) fails at the
    ``pconfig is None`` guard before ever reaching it."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get("vertex")
    assert cfg is not None
    assert cfg.auth_type == "vertex"


def test_vertex_registered_in_hermes_overlays():
    """hermes_cli.providers.get_provider("vertex") backs
    _preserve_provider_with_base_url() in agent/auxiliary_client.py, which
    decides whether a MoA slot's resolved Vertex (base_url, api_key) pair
    keeps its "vertex" provider identity or silently collapses to "custom" —
    losing the identity _refresh_provider_credentials() needs to re-mint an
    expired OAuth2 token on a 401."""
    from hermes_cli.providers import get_provider

    resolved = get_provider("vertex")
    assert resolved is not None
    assert resolved.auth_type == "vertex"
