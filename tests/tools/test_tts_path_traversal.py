import json

import pytest

from tools import tts_tool


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../escape.mp3", "audio/../../escape.mp3"])
async def test_output_path_rejects_traversal(path, monkeypatch):
    async def config():
        return {}

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    result = json.loads(await tts_tool.text_to_speech_tool("hello", path))
    assert result["success"] is False
    assert "traversal" in result["error"].lower()
