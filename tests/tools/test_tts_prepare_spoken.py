import json

import pytest

from tools import tts_tool
from tools.tts_text_normalize import prepare_spoken_text, strip_nonspoken_blocks


def test_reasoning_footer_markdown_emoji_and_newlines_are_cleaned():
    text = (
        "<think>secret</think>## Answer\n**Hello** 🎉\n"
        "⚠️ File-mutation verifier: 1 file(s) were NOT modified this turn"
    )
    spoken = prepare_spoken_text(text)
    assert "secret" not in spoken
    assert "File-mutation verifier" not in spoken
    assert "**" not in spoken
    assert "🎉" not in spoken
    assert "\n" not in spoken


def test_normal_text_is_preserved():
    text = "Just a normal reply."
    assert strip_nonspoken_blocks(text) == text


@pytest.mark.asyncio
async def test_tool_rejects_empty_after_cleanup():
    result = json.loads(
        await tts_tool.text_to_speech_tool("<think>only reasoning</think>")
    )
    assert result["success"] is False
