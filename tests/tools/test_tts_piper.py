"""Native-async parity tests for the retained Piper provider."""

from __future__ import annotations

import json
from pathlib import Path

import aiofiles
import pytest

from tools import tts_tool


def test_piper_registration_and_cache_contract():
    assert "piper" in tts_tool.BUILTIN_TTS_PROVIDERS
    assert tts_tool.PROVIDER_MAX_TEXT_LENGTH["piper"] > 0
    assert isinstance(tts_tool._piper_voice_cache, dict)


@pytest.mark.asyncio
async def test_check_piper_available_uses_async_probe(monkeypatch):
    async def probe(name):
        assert name == "piper"
        return True

    monkeypatch.setattr(tts_tool, "_check_python_module_available", probe)
    assert await tts_tool._check_piper_available() is True


@pytest.mark.asyncio
async def test_resolve_direct_onnx_path(tmp_path):
    model = tmp_path / "custom.onnx"
    async with aiofiles.open(model, "wb") as model_file:
        await model_file.write(b"model")
    assert await tts_tool._resolve_piper_voice_path(str(model), tmp_path) == str(model)


@pytest.mark.asyncio
async def test_empty_voice_uses_cached_default(tmp_path):
    model = tmp_path / f"{tts_tool.DEFAULT_PIPER_VOICE}.onnx"
    model_config = tmp_path / f"{tts_tool.DEFAULT_PIPER_VOICE}.onnx.json"
    async with aiofiles.open(model, "wb") as model_file:
        await model_file.write(b"model")
    async with aiofiles.open(model_config, "w") as config_file:
        await config_file.write("{}")
    assert await tts_tool._resolve_piper_voice_path("", tmp_path) == str(model)


@pytest.mark.asyncio
async def test_generate_forwards_upstream_piper_config(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    async with aiofiles.open(model, "wb") as model_file:
        await model_file.write(b"model")
    captured = {}

    async def synth(provider, payload):
        captured.update(provider=provider, payload=payload)
        async with aiofiles.open(payload["output_path"], "wb") as output:
            await output.write(b"RIFF\x00\x00\x00\x00WAVE")

    monkeypatch.setattr(tts_tool, "_run_local_tts_synth", synth)
    output = tmp_path / "out.wav"
    config = {
        "piper": {
            "voice": str(model),
            "use_cuda": True,
            "speaker_id": 2,
            "length_scale": 1.2,
        }
    }
    assert await tts_tool._generate_piper_tts("hello", str(output), config) == str(output)
    assert captured["provider"] == "piper"
    assert captured["payload"]["model_path"] == str(model)
    assert captured["payload"]["speaker_id"] == 2
    assert captured["payload"]["use_cuda"] is True
    assert captured["payload"]["has_advanced"] is True


@pytest.mark.asyncio
async def test_dispatches_to_piper(tmp_path, monkeypatch):
    async def config():
        return {"provider": "piper"}

    async def available():
        return True

    async def generate(text, output_path, _config):
        assert text == "hi"
        async with aiofiles.open(output_path, "wb") as output:
            await output.write(b"RIFF\x00\x00\x00\x00WAVE")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_piper_available", available)
    monkeypatch.setattr(tts_tool, "_generate_piper_tts", generate)
    output = tmp_path / "clip.wav"
    result = json.loads(
        await tts_tool.text_to_speech_tool("hi", output_path=str(output))
    )
    assert result["success"] is True
    assert result["provider"] == "piper"
    assert Path(result["file_path"]) == output


@pytest.mark.asyncio
async def test_missing_package_surfaces_upstream_error(tmp_path, monkeypatch):
    async def config():
        return {"provider": "piper"}

    async def unavailable():
        return False

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_piper_available", unavailable)
    result = json.loads(
        await tts_tool.text_to_speech_tool(
            "hi",
            output_path=str(tmp_path / "clip.wav"),
        )
    )
    assert result["success"] is False
    assert "piper-tts" in result["error"]
    assert (
        "python -m pip install 'async-hermes-agent[piper-tts]'" in result["error"]
    )
    assert "hermes tools" not in result["error"].lower()


@pytest.mark.asyncio
async def test_piper_availability_controls_requirements(monkeypatch):
    async def config():
        return {"provider": "piper"}

    async def available():
        return True

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_piper_available", available)
    assert await tts_tool.check_tts_requirements() is True
