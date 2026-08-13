import struct

import pytest

from tools import tts_tool


def test_pcm_wav_header_contract():
    pcm = b"\x01\x00\x02\x00"
    wav = tts_tool._wrap_pcm_as_wav(pcm)
    assert len(wav) == 44 + len(pcm)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[40:44])[0] == len(pcm)


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch, tmp_path):
    async def no_key(*args):
        del args
        return ""

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", no_key)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        await tts_tool._generate_gemini_tts(
            "hello", str(tmp_path / "out.wav"), {}
        )
