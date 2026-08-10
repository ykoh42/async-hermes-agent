from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import mini_swe_runner as runner_module
from mini_swe_runner import MiniSWERunner


pytestmark = pytest.mark.asyncio


async def test_run_task_kimi_omits_temperature():
    """Kimi models should NOT have client-side temperature overrides.

    The Kimi gateway selects the correct temperature server-side.
    """
    with patch.object(runner_module, "AsyncOpenAI") as mock_openai:
        client = SimpleNamespace()
        client.close = AsyncMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        runner = MiniSWERunner(
            model="kimi-for-coding",
            base_url="https://api.kimi.com/coding/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = AsyncMock()
        runner._cleanup_env = AsyncMock()

        result = await runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs
    client.close.assert_awaited_once()


async def test_run_task_public_moonshot_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on the public Moonshot API should not get a forced temperature."""
    with patch.object(runner_module, "AsyncOpenAI") as mock_openai:
        client = SimpleNamespace()
        client.close = AsyncMock()
        client.base_url = "https://api.moonshot.ai/v1"
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        runner = MiniSWERunner(
            model="kimi-k2.5",
            base_url="https://api.moonshot.ai/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = AsyncMock()
        runner._cleanup_env = AsyncMock()

        result = await runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs
    client.close.assert_awaited_once()
