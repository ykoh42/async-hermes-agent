import json

import aiofiles
import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_edge_cli_preserves_native_mp3(tmp_path, monkeypatch):
    async def config():
        return {"provider": "edge"}

    async def generate(text, output_path, config):
        del text, config
        async with aiofiles.open(output_path, "wb") as output_file:
            await output_file.write(b"ID3" + b"\0" * 32)
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", generate)
    output = tmp_path / "voice.mp3"
    result = json.loads(
        await tts_tool.text_to_speech_tool("hello", str(output))
    )
    assert result["success"] is True
    assert result["file_path"] == str(output)
    assert result["voice_compatible"] is False
