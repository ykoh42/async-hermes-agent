"""Tests for trajectory_compressor AsyncOpenAI event loop binding.

The compressor constructor remains synchronous and side-effect-free. Its
AsyncOpenAI transport is created lazily by _get_client() so it binds to the
event loop that performs the first request.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from blockbuster import BlockBuster

from agent import secret_scope


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
    @pytest.mark.asyncio
    async def test_get_client_creates_new_client(self):
        """_get_client() should create a fresh AsyncOpenAI instance."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._client_api_key = "test-key"
        comp.client = None
        comp._client_init_lock = asyncio.Lock()

        mock_async_openai = MagicMock()
        create_client = AsyncMock(return_value=MagicMock())
        with (
            patch("openai.AsyncOpenAI", mock_async_openai),
            patch(
                "agent.ssl_verify._create_openai_sdk_client",
                create_client,
            ),
        ):
            client = await comp._get_client()

        create_client.assert_awaited_once_with(
            mock_async_openai,
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )
        assert client is comp.client

    @pytest.mark.asyncio
    async def test_get_client_reuses_owned_client(self):
        """One compressor lifecycle reuses one event-loop-bound client."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._client_api_key = "test-key"
        comp.client = None
        comp._client_init_lock = asyncio.Lock()

        instance = MagicMock()
        create_client = AsyncMock(return_value=instance)
        with patch(
            "agent.ssl_verify._create_openai_sdk_client",
            create_client,
        ):
            client1 = await comp._get_client()
            client2 = await comp._get_client()

        create_client.assert_awaited_once()
        assert client1 is client2 is instance

    @pytest.mark.asyncio
    async def test_get_client_serializes_concurrent_initialization(self):
        """Concurrent files must share one owned summarizer client."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._client_api_key = "test-key"
        comp.client = None
        comp._client_init_lock = asyncio.Lock()

        construction_started = asyncio.Event()
        allow_construction = asyncio.Event()
        instance = MagicMock()

        async def create_client(*args, **kwargs):
            construction_started.set()
            await allow_construction.wait()
            return instance

        with patch(
            "agent.ssl_verify._create_openai_sdk_client",
            AsyncMock(side_effect=create_client),
        ) as client_factory:
            first = asyncio.create_task(comp._get_client())
            await construction_started.wait()
            second = asyncio.create_task(comp._get_client())
            await asyncio.sleep(0)

            client_factory.assert_awaited_once()
            allow_construction.set()
            first_client, second_client = await asyncio.gather(first, second)

        client_factory.assert_awaited_once()
        assert first_client is second_client is instance

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
    async def test_close_finishes_through_repeated_caller_cancellation(self):
        from trajectory_compressor import TrajectoryCompressor

        close_started = asyncio.Event()
        allow_close = asyncio.Event()

        class BlockingClient:
            async def close(self):
                close_started.set()
                await allow_close.wait()

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.client = BlockingClient()

        close_task = asyncio.create_task(comp.close())
        await close_started.wait()
        close_task.cancel()
        await asyncio.sleep(0)
        close_task.cancel()

        await asyncio.sleep(0)
        assert close_task.done() is False

        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

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
    compressor._get_client = AsyncMock(return_value=async_client)

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
    compressor._get_client = AsyncMock(return_value=async_client)

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
    compressor._get_client = AsyncMock(return_value=async_client)

    metrics = TrajectoryMetrics()
    result = await compressor._generate_summary("tool output", metrics)

    assert result.startswith("[CONTEXT SUMMARY]:")
    assert "temperature" not in async_client.chat.completions.create.call_args.kwargs


@pytest.fixture
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    scope_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
async def test_custom_summarizer_api_key_is_profile_scoped(
    monkeypatch,
    _restore_secret_scope,
):
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor

    monkeypatch.setenv("CUSTOM_SUMMARY_KEY", "process-key")
    secret_scope.set_multiplex_active(True)

    async def initialize(api_key: str) -> str:
        token = secret_scope.set_secret_scope({"CUSTOM_SUMMARY_KEY": api_key})
        try:
            compressor = TrajectoryCompressor(
                CompressionConfig(
                    base_url="https://custom.summary.test/v1",
                    api_key_env="CUSTOM_SUMMARY_KEY",
                )
            )
            await asyncio.sleep(0)
            compressor._init_summarizer()
            return compressor._client_api_key
        finally:
            secret_scope.reset_secret_scope(token)

    key_a, key_b = await asyncio.gather(
        initialize("profile-a-key"),
        initialize("profile-b-key"),
    )

    assert key_a == "profile-a-key"
    assert key_b == "profile-b-key"


@pytest.mark.asyncio
async def test_huggingface_tokenizer_auth_is_profile_scoped(
    monkeypatch,
    _restore_secret_scope,
):
    import trajectory_compressor as compressor_module

    captured: list[str | None] = []

    class Response:
        status_code = 200
        content = b"{}"
        headers = {"etag": "test-etag"}

        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, authorization: str | None) -> None:
            self.authorization = authorization

        async def __aenter__(self):
            captured.append(self.authorization)
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url: str, *, headers: dict[str, str]):
            return Response()

    async def create_client(**kwargs):
        headers = kwargs.get("headers") or {}
        return Client(headers.get("Authorization"))

    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_client,
    )
    monkeypatch.setattr(
        compressor_module,
        "_read_optional_bytes",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        compressor_module,
        "_atomic_write_bytes",
        AsyncMock(return_value=None),
    )
    monkeypatch.setenv("HF_TOKEN", "process-token")
    secret_scope.set_multiplex_active(True)

    async def load(token_value: str) -> None:
        token = secret_scope.set_secret_scope({"HF_TOKEN": token_value})
        try:
            await compressor_module._load_hf_tokenizer_assets(
                "owner/tokenizer",
                ("tokenizer.json",),
            )
        finally:
            secret_scope.reset_secret_scope(token)

    await asyncio.gather(load("profile-a-token"), load("profile-b-token"))

    assert sorted(captured) == [
        "Bearer profile-a-token",
        "Bearer profile-b-token",
    ]


@pytest.mark.asyncio
async def test_compressor_credentials_fail_closed_without_profile_scope(
    monkeypatch,
    _restore_secret_scope,
):
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor

    monkeypatch.setenv("CUSTOM_SUMMARY_KEY", "process-key")
    secret_scope.set_multiplex_active(True)
    compressor = TrajectoryCompressor(
        CompressionConfig(
            base_url="https://custom.summary.test/v1",
            api_key_env="CUSTOM_SUMMARY_KEY",
        )
    )

    with pytest.raises(secret_scope.UnscopedSecretError, match="CUSTOM_SUMMARY_KEY"):
        compressor._init_summarizer()
