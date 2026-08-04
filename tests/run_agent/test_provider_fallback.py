"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None



    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"


    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    @pytest.mark.asyncio
    async def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert await agent._try_activate_fallback() is False

    @pytest.mark.asyncio
    async def test_advances_index(self):
        fbs = [
            {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "fallback-key",
                "base_url": "https://api.openai.com/v1",
            },
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert await agent._try_activate_fallback() is True
        assert agent._fallback_index == 1
        assert agent.model == "gpt-4o"
        assert agent._fallback_activated is True



    @pytest.mark.asyncio
    async def test_skips_unconfigured_provider_to_next(self):
        """A provider without native async credentials is skipped."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "fallback-key",
                "base_url": "https://api.openai.com/v1",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        assert await agent._try_activate_fallback() is True
        assert agent.model == "gpt-4o"
        assert agent._fallback_index == 2

    @pytest.mark.asyncio
    async def test_skips_provider_that_raises_to_next(self):
        """An unsupported native credential source is skipped."""
        fbs = [
            {"provider": "anthropic", "model": "claude-test"},
            {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "fallback-key",
                "base_url": "https://api.openai.com/v1",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        assert await agent._try_activate_fallback() is True
        assert agent.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False):
            assert await agent._try_activate_fallback() is True
        assert agent.api_key == "env-secret"
        assert agent.base_url == "https://fallback.example/v1"


    @pytest.mark.asyncio
    async def test_nous_anthropic_fallback_uses_the_messages_wire(self):
        """Portal Claude fallbacks must not stay on chat_completions.

        The native resolver must derive ``api_mode`` from the model and use
        an Anthropic client — otherwise the turn POSTs /chat/completions.
        """
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [
            {
                "provider": "nous",
                "model": "anthropic/claude-opus-4.8",
                "api_key": "portal-jwt",
                "base_url": portal,
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        rebuilt = {"count": 0}

        def _fake_build(api_key, base_url, timeout=None, **kwargs):
            rebuilt["count"] += 1
            rebuilt["api_key"] = api_key
            rebuilt["base_url"] = base_url
            return MagicMock(name="anthropic-client")

        with (
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=_fake_build,
            ),
        ):
            assert await agent._try_activate_fallback() is True

        assert agent.api_mode == "anthropic_messages"
        assert agent.provider == "nous"
        assert agent.model == "anthropic/claude-opus-4.8"
        assert agent.client is None
        assert rebuilt["count"] == 1
        assert rebuilt["api_key"] == "portal-jwt"
        assert rebuilt["base_url"] == portal
        assert agent._anthropic_client is not None

    @pytest.mark.asyncio
    async def test_nous_non_anthropic_fallback_stays_on_chat_completions(self):
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [{
            "provider": "nous",
            "model": "hermes-4-405b",
            "api_key": "portal-jwt",
            "base_url": portal,
        }]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=AssertionError("must not build Anthropic client"),
            ),
        ):
            assert await agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"
        assert agent.client is not None


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False







# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    @pytest.mark.asyncio
    async def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {
                "provider": "zai",
                "model": "glm-4.7",
                "api_key": "zai-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda m, p: m,
        ):
            ok = await agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped and the second native client activated.
        assert agent._fallback_index == 2
        assert agent.provider == "zai"


    @pytest.mark.asyncio
    async def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        ok = await agent._try_activate_fallback()

        assert ok is False

    @pytest.mark.asyncio
    async def test_allows_xai_api_fallback_from_xai_oauth_same_host_model(self):
        """xai-oauth and xai share api.x.ai but use different credentials.

        A spending-limit 403 on OAuth must still be able to fall over to the
        API-key provider even when both entries use the same model slug and
        base URL.  Blind base_url+model dedup incorrectly skipped that path.
        """
        fbs = [
            {
                "provider": "xai",
                "model": "grok-4.5",
                "base_url": "https://api.x.ai/v1",
                "api_key": "xai-key",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "xai-oauth"
        agent.model = "grok-4.5"
        agent.base_url = "https://api.x.ai/v1"

        with patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda m, p: m,
        ):
            ok = await agent._try_activate_fallback()

        assert ok is True
        assert agent.provider == "xai"
        assert agent.model == "grok-4.5"
