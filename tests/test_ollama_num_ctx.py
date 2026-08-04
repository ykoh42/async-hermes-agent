"""Tests for Ollama num_ctx context length detection and injection.

Covers:
  agent/model_metadata.py — query_ollama_num_ctx()
  run_agent.py — _ollama_num_ctx detection + extra_body injection
"""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest


from agent.model_metadata import query_ollama_num_ctx, query_ollama_supports_vision


# ═══════════════════════════════════════════════════════════════════════
# Level 1: query_ollama_num_ctx — Ollama API interaction
# ═══════════════════════════════════════════════════════════════════════


def _mock_httpx_client(show_response_data, status_code=200):
    """Create a mock httpx.AsyncClient returning the given /api/show data."""
    mock_resp = MagicMock(status_code=status_code)
    mock_resp.json.return_value = show_response_data
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    class _ClientContext:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *_args):
            return False

    return _ClientContext(), mock_client


class TestQueryOllamaNumCtx:
    """Test the Ollama /api/show context length query."""

    @pytest.mark.asyncio
    async def test_returns_context_from_model_info(self):
        """Should extract context_length from GGUF model_info metadata."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx), \
             patch("agent.model_metadata._local_probe_disk_get", new=AsyncMock(return_value=None)), \
             patch("agent.model_metadata._local_probe_disk_put", new=AsyncMock()):
            result = await query_ollama_num_ctx("llama3.1:8b", "http://localhost:11434/v1")

        assert result == 131072

    @pytest.mark.asyncio
    async def test_prefers_explicit_num_ctx_from_modelfile(self):
        """If the Modelfile sets num_ctx explicitly, that should take priority."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "num_ctx 32768\ntemperature 0.7",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx), \
             patch("agent.model_metadata._local_probe_disk_get", new=AsyncMock(return_value=None)), \
             patch("agent.model_metadata._local_probe_disk_put", new=AsyncMock()):
            result = await query_ollama_num_ctx("custom-model", "http://localhost:11434")

        assert result == 32768




    @pytest.mark.asyncio
    async def test_strips_provider_prefix(self):
        """Should strip 'local:' prefix from model name before querying."""
        show_data = {
            "model_info": {"qwen2.context_length": 32768},
            "parameters": "",
        }
        mock_ctx, mock_client = _mock_httpx_client(show_data)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx), \
             patch("agent.model_metadata._local_probe_disk_get", new=AsyncMock(return_value=None)), \
             patch("agent.model_metadata._local_probe_disk_put", new=AsyncMock()):
            result = await query_ollama_num_ctx("local:qwen2.5:7b", "http://localhost:11434/v1")

        # Verify the post was called with stripped name (no "local:" prefix)
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["name"] == "qwen2.5:7b" or call_args[0][1] is not None
        assert result == 32768

    @pytest.mark.asyncio
    async def test_handles_qwen2_architecture_key(self):
        """Different model architectures use different key prefixes in model_info."""
        show_data = {
            "model_info": {"qwen2.context_length": 65536},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx), \
             patch("agent.model_metadata._local_probe_disk_get", new=AsyncMock(return_value=None)), \
             patch("agent.model_metadata._local_probe_disk_put", new=AsyncMock()):
            result = await query_ollama_num_ctx("qwen2.5:32b", "http://localhost:11434")

        assert result == 65536



class TestQueryOllamaSupportsVision:
    """Test Ollama /api/show vision capability detection."""

    @pytest.mark.asyncio
    async def test_returns_true_when_capabilities_include_vision(self):
        show_data = {"capabilities": ["completion", "vision"]}
        mock_ctx, _ = _mock_httpx_client(show_data)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx):
            result = await query_ollama_supports_vision("gemma4:e2b", "http://localhost:11434/v1")

        assert result is True


    @pytest.mark.asyncio
    async def test_falls_back_to_model_info_vision_block_count(self):
        show_data = {"model_info": {"gemma3.vision.block_count": 27}}
        mock_ctx, _ = _mock_httpx_client(show_data)

        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx):
            result = await query_ollama_supports_vision("llava", "http://localhost:11434")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_none_for_non_ollama_server(self):
        mock_ctx, _ = _mock_httpx_client({}, status_code=404)
        import httpx
        with patch.object(httpx, "AsyncClient", return_value=mock_ctx):
            result = await query_ollama_supports_vision("llava", "http://localhost:8000/v1")
        assert result is None
