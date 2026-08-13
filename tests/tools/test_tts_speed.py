import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_edge_speed_maps_to_rate(monkeypatch, tmp_path):
    calls = {}

    class Communicate:
        def __init__(self, text, **kwargs):
            calls.update(text=text, **kwargs)

        async def save(self, output_path):
            calls["output_path"] = output_path

    edge = type("Edge", (), {"Communicate": Communicate})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: edge)
    output = str(tmp_path / "out.mp3")
    assert await tts_tool._generate_edge_tts(
        "hello", output, {"speed": 1.5}
    ) == output
    assert calls["rate"] == "+50%"


@pytest.mark.asyncio
async def test_tool_clamps_speed_before_dispatch(monkeypatch, tmp_path):
    seen = {}

    async def config():
        return {"provider": "edge"}

    async def generate(text, output_path, config):
        del text
        seen.update(config)
        import aiofiles

        async with aiofiles.open(output_path, "wb") as output_file:
            await output_file.write(b"ID3" + b"\0" * 16)
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", generate)
    await tts_tool.text_to_speech_tool(
        "hello", str(tmp_path / "out.mp3"), speed=99
    )
    assert seen["speed"] == 4.0
