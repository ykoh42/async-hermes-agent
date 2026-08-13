import pytest

from tools import tool_backend_helpers, tts_tool


@pytest.mark.asyncio
async def test_tts_provider_key_reads_async_dotenv(monkeypatch):
    async def dotenv(name):
        return "dotenv-key" if name == "MINIMAX_API_KEY" else ""

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        dotenv,
    )
    assert await tool_backend_helpers.resolve_provider_secret(
        "MINIMAX_API_KEY", "minimax"
    ) == "dotenv-key"
    assert await tts_tool._resolve_provider_key(
        "MINIMAX_API_KEY", "minimax"
    ) == "dotenv-key"


@pytest.mark.asyncio
async def test_check_requirements_sees_dotenv_minimax(monkeypatch):
    async def config():
        return {"provider": "minimax"}

    async def key(env_var, provider_id):
        del provider_id
        return "dotenv-key" if env_var == "MINIMAX_API_KEY" else ""

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", key)
    assert await tts_tool.check_tts_requirements() is True
