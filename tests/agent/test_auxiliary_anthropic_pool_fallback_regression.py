"""Regression coverage for the native async Anthropic auxiliary adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio


class TestAnthropicAuxiliaryCapability:
    @staticmethod
    def _exhausted_pool():
        pool = MagicMock()
        pool.entries.return_value = [object()]
        pool.select = AsyncMock(return_value=None)
        return pool

    async def test_pool_present_no_entry_falls_back_to_resolve_token(self):
        from agent.auxiliary_client import _try_anthropic

        with (
            patch(
                "agent.credential_pool.load_pool",
                new=AsyncMock(return_value=self._exhausted_pool()),
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                new=AsyncMock(return_value="sk-ant-oat01-token"),
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=MagicMock(),
            ) as build_client,
        ):
            client, model = await _try_anthropic()

        assert client is not None
        assert model == "claude-haiku-4-5-20251001"
        assert client.chat.completions._is_oauth is True
        assert build_client.await_args.args[0] == "sk-ant-oat01-token"

    async def test_pool_present_no_entry_and_no_token_still_returns_none(self):
        from agent.auxiliary_client import _try_anthropic

        with (
            patch(
                "agent.credential_pool.load_pool",
                new=AsyncMock(return_value=self._exhausted_pool()),
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            assert await _try_anthropic() == (None, None)

    async def test_base_url_defaults_when_pool_present_but_no_entry(self):
        from agent.auxiliary_client import _try_anthropic

        with (
            patch(
                "agent.credential_pool.load_pool",
                new=AsyncMock(return_value=self._exhausted_pool()),
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                new=AsyncMock(return_value="sk-ant-oat01-token"),
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                new=AsyncMock(return_value=MagicMock()),
            ) as build_client,
        ):
            client, _model = await _try_anthropic()

        assert client is not None
        assert build_client.await_args.args[1] == "https://api.anthropic.com"
