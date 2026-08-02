"""Regression coverage for the native async Anthropic auxiliary adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio


class TestAnthropicAuxiliaryCapability:
    async def test_pool_without_entry_falls_back_to_native_token_resolution(self):
        from agent.auxiliary_client import _try_anthropic

        with (
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
