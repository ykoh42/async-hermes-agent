"""Tests for the 1M-context beta header on AWS Bedrock Claude models.

Claude Opus 4.6/4.7 and Sonnet 4.6 support a 1M context window, but on AWS
Bedrock (and Microsoft Foundry) that window is still gated behind the
``context-1m-2025-08-07`` beta header as of 2026-04. Without it, Bedrock
caps these models at 200K even though ``model_metadata.py`` advertises 1M.

These tests guard the invariant that the header is always emitted on the
Bedrock client path, and that it survives the MiniMax bearer-auth strip.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBedrockContext1MBeta:
    """``context-1m-2025-08-07`` must reach Bedrock Claude requests."""



    def test_common_betas_strips_1m_for_minimax(self):
        """MiniMax bearer-auth endpoints host their own models — strip 1M beta."""
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _CONTEXT_1M_BETA,
        )

        for url in (
            "https://api.minimax.io/anthropic",
            "https://api.minimaxi.com/anthropic",
        ):
            betas = _common_betas_for_base_url(url)
            assert _CONTEXT_1M_BETA not in betas, (
                f"1M beta must be stripped for MiniMax bearer endpoint {url}"
            )
            # Other betas still present
            assert "interleaved-thinking-2025-05-14" in betas

    @pytest.mark.asyncio
    async def test_build_anthropic_bedrock_client_sends_1m_beta(self):
        """AnthropicBedrock client must carry the 1M beta in default_headers.

        This is the load-bearing assertion for the reported bug:
        without this header Bedrock serves Opus 4.6/4.7 with a 200K cap.
        """
        import agent.anthropic_adapter as adapter

        fake_sdk = MagicMock()
        fake_sdk.AsyncAnthropicBedrock = MagicMock()

        credentials = MagicMock(
            get_frozen_credentials=AsyncMock(
                return_value=SimpleNamespace(
                    access_key="access",
                    secret_key="secret",
                    token="token",
                )
            )
        )
        session = MagicMock(
            get_credentials=AsyncMock(return_value=credentials)
        )
        with (
            patch.object(adapter, "_anthropic_sdk", fake_sdk),
            patch("aiobotocore.session.get_session", return_value=session),
            patch.object(
                adapter,
                "_build_anthropic_default_http_client",
                new=AsyncMock(return_value=AsyncMock()),
            ),
        ):
            await adapter.build_anthropic_bedrock_client(region="us-west-2")

        call_kwargs = fake_sdk.AsyncAnthropicBedrock.call_args.kwargs
        assert call_kwargs["aws_region"] == "us-west-2"

        default_headers = call_kwargs.get("default_headers") or {}
        beta_header = default_headers.get("anthropic-beta", "")
        assert "context-1m-2025-08-07" in beta_header, (
            "Bedrock client must send context-1m-2025-08-07 or Opus 4.6/4.7 "
            "silently caps at 200K context"
        )
        # Other common betas still present — no regression.
        assert "interleaved-thinking-2025-05-14" in beta_header
        assert "fine-grained-tool-streaming-2025-05-14" in beta_header

    @pytest.mark.asyncio
    async def test_bedrock_uses_awaited_anthropic_http_transport(
        self,
        monkeypatch,
    ):
        import anthropic
        import anthropic._base_client as base_client
        import agent.anthropic_adapter as adapter

        credentials = MagicMock(
            get_frozen_credentials=AsyncMock(
                return_value=SimpleNamespace(
                    access_key="access",
                    secret_key="secret",
                    token="token",
                )
            )
        )
        session = MagicMock(get_credentials=AsyncMock(return_value=credentials))
        monkeypatch.setattr(adapter, "_anthropic_sdk", anthropic)
        monkeypatch.setattr("aiobotocore.session.get_session", lambda: session)

        client = await adapter.build_anthropic_bedrock_client("us-west-2")
        try:
            assert isinstance(
                client._client,
                base_client.AsyncHttpxClientWrapper,
            )
            assert str(client.base_url) == (
                "https://bedrock-runtime.us-west-2.amazonaws.com"
            )
            assert client._client._transport._pool._socket_options == (
                adapter._anthropic_socket_options()
            )
        finally:
            await client.close()
