"""Tests for trajectory_compressor AsyncOpenAI event loop binding.

The compressor constructor remains synchronous and side-effect-free. Its
AsyncOpenAI transport is created lazily by _get_client() so it binds to the
event loop that performs the first request.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from blockbuster import BlockBuster


class TestAsyncClientLazyCreation:
    """trajectory_compressor.py — lazy native-async client ownership."""

    def test_client_none_after_init(self):
        """client should be None after __init__ (not eagerly created)."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp.config.api_key_env = "TEST_API_KEY"
        comp._use_call_llm = False
        comp.client = None
        comp._client_api_key = "test-key"

        assert comp.client is None

    def test_get_client_creates_new_client(self):
        """_get_client() should create a fresh AsyncOpenAI instance."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._client_api_key = "test-key"
        comp.client = None

        mock_async_openai = MagicMock()
        with patch("openai.AsyncOpenAI", mock_async_openai):
            client = comp._get_client()

        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )
        assert comp.client is not None

    def test_get_client_reuses_owned_client(self):
        """One compressor lifecycle reuses one event-loop-bound client."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._client_api_key = "test-key"
        comp.client = None

        call_count = 0
        instances = []

        def mock_constructor(**kwargs):
            nonlocal call_count
            call_count += 1
            instance = MagicMock()
            instances.append(instance)
            return instance

        with patch("openai.AsyncOpenAI", side_effect=mock_constructor):
            client1 = comp._get_client()
            client2 = comp._get_client()

        assert call_count == 1
        assert client1 is client2 is instances[0]

    @pytest.mark.asyncio
    async def test_close_releases_owned_client(self):
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        client = MagicMock(close=AsyncMock())
        comp.client = client

        await comp.close()

        client.close.assert_awaited_once_with()
        assert comp.client is None


@pytest.mark.asyncio
async def test_single_file_temp_workspace_is_non_blocking(tmp_path):
    from trajectory_compressor import TrajectoryCompressor, main

    source = tmp_path / "input.jsonl"
    source.write_text('{"messages": []}\n', encoding="utf-8")
    output = tmp_path / "output.jsonl"
    blocker = BlockBuster()
    blocker.activate()
    try:
        with (
            patch.object(TrajectoryCompressor, "process_directory", AsyncMock()),
            patch.object(TrajectoryCompressor, "close", AsyncMock()),
        ):
            await main(
                input=str(source),
                output=str(output),
                config=str(tmp_path / "missing.yaml"),
            )
    finally:
        blocker.deactivate()

    assert output.read_text(encoding="utf-8") == ""


class TestSourceLineVerification:
    """Verify the actual source has the lazy pattern applied."""

    @staticmethod
    def _read_file() -> str:
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "trajectory_compressor.py")) as f:
            return f.read()

    def test_no_eager_async_openai_in_init(self):
        """__init__ should NOT create AsyncOpenAI eagerly."""
        src = self._read_file()
        # AsyncOpenAI construction must remain inside the lazy accessor.
        lines = src.split("\n")
        for i, line in enumerate(lines, 1):
            if "self.client = AsyncOpenAI(" in line and "_get_client" not in lines[max(0,i-3):i+1]:
                # Check if we're inside _get_client by looking at context
                context = "\n".join(lines[max(0,i-20):i+1])
                if "_get_client" not in context:
                    pytest.fail(
                        f"Line {i}: AsyncOpenAI created eagerly outside _get_client()"
                    )

    def test_get_client_method_exists(self):
        """The lazy client accessor should exist."""
        src = self._read_file()
        assert "def _get_client(self)" in src


@pytest.mark.asyncio
async def test_generate_summary_async_kimi_omits_temperature():
    """Kimi models should have temperature omitted — server manages it."""
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

    config = CompressionConfig(
        summarization_model="kimi-for-coding",
        temperature=0.3,
        summary_target_tokens=100,
        max_retries=1,
    )
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))]
    ))
    compressor._get_client = MagicMock(return_value=async_client)

    metrics = TrajectoryMetrics()
    result = await compressor._generate_summary("tool output", metrics)

    assert result.startswith("[CONTEXT SUMMARY]:")
    assert "temperature" not in async_client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_generate_summary_async_public_moonshot_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on the public Moonshot API should not get a forced temperature."""
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

    config = CompressionConfig(
        summarization_model="kimi-k2.5",
        base_url="https://api.moonshot.ai/v1",
        temperature=0.3,
        summary_target_tokens=100,
        max_retries=1,
    )
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))]
    ))
    compressor._get_client = MagicMock(return_value=async_client)

    metrics = TrajectoryMetrics()
    result = await compressor._generate_summary("tool output", metrics)

    assert result.startswith("[CONTEXT SUMMARY]:")
    assert "temperature" not in async_client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_generate_summary_async_public_moonshot_cn_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on api.moonshot.cn should not get a forced temperature."""
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

    config = CompressionConfig(
        summarization_model="kimi-k2.5",
        base_url="https://api.moonshot.cn/v1",
        temperature=0.3,
        summary_target_tokens=100,
        max_retries=1,
    )
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))]
    ))
    compressor._get_client = MagicMock(return_value=async_client)

    metrics = TrajectoryMetrics()
    result = await compressor._generate_summary("tool output", metrics)

    assert result.startswith("[CONTEXT SUMMARY]:")
    assert "temperature" not in async_client.chat.completions.create.call_args.kwargs
