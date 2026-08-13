import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_raises_when_no_model_resolvable(monkeypatch, tmp_path):
    async def key(*args):
        del args
        return "deep-key"

    async def no_models(tag):
        assert tag == "tts"
        return []

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", key)
    monkeypatch.setattr("hermes_cli.models._fetch_deepinfra_models_by_tag", no_models)
    with pytest.raises(ValueError, match="No DeepInfra TTS model"):
        await tts_tool._generate_deepinfra_tts(
            "hello", str(tmp_path / "out.mp3"), {}
        )


@pytest.mark.asyncio
async def test_requirements_follow_explicit_provider(monkeypatch):
    async def config():
        return {"provider": "deepinfra"}

    async def key(*args):
        del args
        return "deep-key"

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", key)
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
    assert await tts_tool.check_tts_requirements() is True
