import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_openai_instructions_are_forwarded(tmp_path, monkeypatch):
    calls = {}

    class Response:
        async def stream_to_file(self, path):
            import aiofiles

            async with aiofiles.open(path, "wb") as output_file:
                await output_file.write(b"audio")

    class Context:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *args):
            del args

    class Streaming:
        def create(self, **kwargs):
            calls.update(kwargs)
            return Context()

    class Client:
        def __init__(self, **kwargs):
            del kwargs
            self.audio = type("Audio", (), {})()
            self.audio.speech = type("Speech", (), {})()
            self.audio.speech.with_streaming_response = Streaming()

        async def close(self):
            return None

    async def config():
        return "key", "https://api.openai.com/v1", False

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: Client)
    monkeypatch.setattr(tts_tool, "_resolve_openai_audio_client_config", config)
    await tts_tool._generate_openai_tts(
        "hello",
        str(tmp_path / "out.mp3"),
        {},
        instructions="Speak cheerfully",
    )
    assert calls["instructions"] == "Speak cheerfully"
    assert "instructions" in tts_tool.TTS_SCHEMA["parameters"]["properties"]
