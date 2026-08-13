"""Native-async parity tests for the retained KittenTTS provider."""

import json

import aiofiles
import pytest

from tools import tts_tool


OFFICIAL_WHEEL_URL = (
    "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/"
    "kittentts-0.8.1-py3-none-any.whl"
)
OFFICIAL_WHEEL_SHA256 = (
    "482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851"
)


@pytest.fixture(autouse=True)
def clear_kittentts_cache():
    tts_tool._kittentts_model_cache.clear()
    yield
    tts_tool._kittentts_model_cache.clear()


@pytest.mark.asyncio
async def test_check_kittentts_available_uses_async_probe(monkeypatch):
    async def probe(name):
        assert name == "kittentts"
        return True

    monkeypatch.setattr(tts_tool, "_check_python_module_available", probe)
    assert await tts_tool._check_kittentts_available() is True


@pytest.mark.asyncio
async def test_generate_forwards_voice_speed_and_clean_text(tmp_path, monkeypatch):
    captured = {}

    async def synth(provider, payload):
        captured.update(provider=provider, payload=payload)
        async with aiofiles.open(payload["output_path"], "wb") as output:
            await output.write(b"RIFF\x00\x00\x00\x00WAVE")

    monkeypatch.setattr(tts_tool, "_run_local_tts_synth", synth)
    output = tmp_path / "out.wav"
    config = {
        "kittentts": {
            "model": "KittenML/kitten-tts-mini-0.8",
            "voice": "Luna",
            "speed": 1.25,
            "clean_text": False,
        }
    }
    assert await tts_tool._generate_kittentts("Hi", str(output), config) == str(output)
    assert captured["provider"] == "kittentts"
    assert captured["payload"]["voice"] == "Luna"
    assert captured["payload"]["speed"] == 1.25
    assert captured["payload"]["clean_text"] is False


@pytest.mark.asyncio
async def test_dispatches_to_kittentts(tmp_path, monkeypatch):
    async def config():
        return {"provider": "kittentts"}

    async def available():
        return True

    async def generate(_text, output_path, _config):
        async with aiofiles.open(output_path, "wb") as output:
            await output.write(b"RIFF\x00\x00\x00\x00WAVE")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_kittentts_available", available)
    monkeypatch.setattr(tts_tool, "_generate_kittentts", generate)
    result = json.loads(
        await tts_tool.text_to_speech_tool(
            "Hello",
            output_path=str(tmp_path / "clip.wav"),
        )
    )
    assert result["success"] is True
    assert result["provider"] == "kittentts"


@pytest.mark.asyncio
async def test_missing_package_returns_upstream_help(tmp_path, monkeypatch):
    async def config():
        return {"provider": "kittentts"}

    async def unavailable():
        return False

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_kittentts_available", unavailable)
    result = json.loads(
        await tts_tool.text_to_speech_tool(
            "Hello",
            output_path=str(tmp_path / "clip.wav"),
        )
    )
    assert result["success"] is False
    error = result["error"]
    assert f"{OFFICIAL_WHEEL_URL}#sha256={OFFICIAL_WHEEL_SHA256}" in error
    assert "public-index 'kittentts'" in error
    assert "not the compatible KittenML 0.8.1 artifact" in error
    assert "soundfile" not in error.lower()
    assert "pip install kittentts" not in error.lower()
    assert "hermes setup tts" not in error.lower()


@pytest.mark.asyncio
async def test_kittentts_availability_controls_requirements(monkeypatch):
    async def config():
        return {"provider": "kittentts"}

    async def available():
        return True

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_kittentts_available", available)
    assert await tts_tool.check_tts_requirements() is True
