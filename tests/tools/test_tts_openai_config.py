import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_config_credentials_and_base_url_win(monkeypatch):
    async def config():
        return {
            "openai": {
                "api_key": "config-key",
                "base_url": "https://voice.example/v1",
            }
        }

    async def direct(*args):
        del args
        return "env-key"

    async def no_gateway(section):
        del section
        return False

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", direct)
    monkeypatch.setattr(tts_tool, "prefers_gateway", no_gateway)
    assert await tts_tool._resolve_openai_audio_client_config() == (
        "config-key",
        "https://voice.example/v1",
        False,
    )
    assert await tts_tool._has_openai_audio_backend() is True
