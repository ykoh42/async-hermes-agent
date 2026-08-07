"""Regression test for #17929: first async initialization tries fallback_model
when primary provider credentials are exhausted."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from hermes_cli.auth import AuthError
from run_agent import AIAgent


def _make_tool_defs():
    return [{"type": "function", "function": {"name": "web_search",
             "description": "search", "parameters": {"type": "object", "properties": {}}}}]


def _mock_client(api_key="fb-key-1234567890", base_url="https://fb.example.com/v1"):
    c = MagicMock()
    c.api_key = api_key
    c.base_url = base_url
    c._default_headers = None
    return c


class _EmptyPool:
    def has_credentials(self):
        return False

    async def select(self):
        return None


@pytest.mark.asyncio
async def test_init_tries_fallback_when_primary_returns_none():
    """The synchronous constructor stays lazy; first await activates fallback."""
    fb = _mock_client()

    with patch("hermes_cli.runtime_provider.load_pool", new_callable=AsyncMock, return_value=_EmptyPool()), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=fb):

        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{
                "provider": "custom",
                "model": "kimi2.5",
                "api_key": "fb-key-1234567890",
                "base_url": "https://fb.example.com/v1",
            }],
        )
        await agent._ensure_provider_runtime()
        assert agent.provider == "custom"
        assert agent.model == "kimi2.5"
        assert agent._fallback_activated is True


@pytest.mark.asyncio
async def test_init_raises_when_no_fallback_configured():
    """Missing primary credentials fail on first async initialization."""
    with patch("hermes_cli.runtime_provider.load_pool", new_callable=AsyncMock, return_value=_EmptyPool()), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=None,
        )
        with pytest.raises(AuthError, match="No native async credentials"):
            await agent._ensure_provider_runtime()
