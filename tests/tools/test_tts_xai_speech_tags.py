import pytest

from tools import tts_tool


@pytest.mark.asyncio
async def test_local_pause_added_when_auxiliary_rewrite_fails(monkeypatch):
    async def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("offline")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fail)
    result = await tts_tool._apply_xai_auto_speech_tags(
        "This is the first sentence. This is the second sentence."
    )
    assert "[pause]" in result


@pytest.mark.asyncio
async def test_explicit_tags_are_preserved():
    text = "[whisper]This is already tagged.[/whisper]"
    assert await tts_tool._apply_xai_auto_speech_tags(text) == text
