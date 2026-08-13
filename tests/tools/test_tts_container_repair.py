import struct

import aiofiles
import pytest

from tools.tts_tool import (
    OPUS_VOICE_PLATFORMS,
    _repair_ogg_container,
    _sniff_audio_container,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"ID3\x04" + b"\0" * 16, "mp3"),
        (b"\xff\xfb\x90\0" + b"\0" * 16, "mp3"),
        (b"OggS" + b"\0" * 16, "ogg"),
        (b"fLaC" + b"\0" * 16, "flac"),
        (b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"\0" * 16, "wav"),
    ],
)
async def test_magic_bytes(tmp_path, data, expected):
    path = tmp_path / "audio.bin"
    async with aiofiles.open(path, "wb") as audio_file:
        await audio_file.write(data)
    assert await _sniff_audio_container(str(path)) == expected


@pytest.mark.asyncio
async def test_real_ogg_is_untouched(tmp_path):
    path = tmp_path / "voice.ogg"
    payload = b"OggS" + b"\0" * 32
    async with aiofiles.open(path, "wb") as audio_file:
        await audio_file.write(payload)
    assert await _repair_ogg_container(str(path)) == str(path)
    async with aiofiles.open(path, "rb") as audio_file:
        assert await audio_file.read() == payload


def test_voice_bubble_platform_set_matches_upstream():
    assert {"telegram", "matrix", "feishu", "whatsapp", "signal"} <= (
        OPUS_VOICE_PLATFORMS
    )
    assert "cli" not in OPUS_VOICE_PLATFORMS
