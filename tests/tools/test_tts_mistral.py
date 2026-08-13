import base64

import aiofiles
import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch, tmp_path):
    async def no_key(*args):
        del args
        return ""

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", no_key)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        await tts_tool._generate_mistral_tts(
            "hello", str(tmp_path / "out.mp3"), {}
        )


@pytest.mark.asyncio
async def test_complete_async_generation_contract(monkeypatch, tmp_path):
    calls = {}
    payload = b"mistral-audio"

    class Speech:
        async def complete_async(self, **kwargs):
            calls.update(kwargs)
            return type(
                "Response",
                (),
                {"audio_data": base64.b64encode(payload).decode()},
            )()

    class Client:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.audio = type("Audio", (), {"speech": Speech()})()

    async def key(*args):
        del args
        return "mistral-key"

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", key)
    monkeypatch.setattr(tts_tool, "_import_mistral_client", lambda: Client)
    output = tmp_path / "out.ogg"
    assert await tts_tool._generate_mistral_tts(
        "hello", str(output), {"mistral": {"base_url": "https://m.example"}}
    ) == str(output)
    assert calls["response_format"] == "opus"
    assert calls["client"]["server_url"] == "https://m.example"
    async with aiofiles.open(output, "rb") as output_file:
        assert await output_file.read() == payload
