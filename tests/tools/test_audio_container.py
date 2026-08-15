"""Magic-byte audio container parity tests from upstream."""

from __future__ import annotations

import json

import pytest

from tools.audio_container import CONTAINER_TO_EXT, sniff_audio_ext, sniff_container

OGG = b"OggS\x00\x02" + b"\x00" * 64
FLAC = b"fLaC" + b"\x00" * 64
WAV = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 64
MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64
MP3_FRAME = b"\xff\xfb\x90\x00" + b"\x00" * 64
AAC_ADTS = b"\xff\xf1\x50\x80" + b"\x00" * 64
M4A = b"\x00\x00\x00\x1cftypM4A " + b"\x00" * 64
M4B = b"\x00\x00\x00\x1cftypM4B " + b"\x00" * 64
MP4_ISOM = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
UNKNOWN = b"not-audio-at-all" + b"\x00" * 64


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (OGG, "ogg"),
        (FLAC, "flac"),
        (WAV, "wav"),
        (MP3_ID3, "mp3"),
        (MP3_FRAME, "mp3"),
        (AAC_ADTS, "aac"),
        (M4A, "m4a"),
        (M4B, "m4a"),
        (MP4_ISOM, "mp4"),
        (WEBM, "webm"),
    ],
)
def test_magic_bytes(data, expected):
    assert sniff_container(data) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (OGG, ".ogg"),
        (FLAC, ".flac"),
        (WAV, ".wav"),
        (MP3_ID3, ".mp3"),
        (MP3_FRAME, ".mp3"),
        (AAC_ADTS, ".aac"),
        (M4A, ".m4a"),
        (WEBM, ".webm"),
    ],
)
def test_sniff_audio_ext_overrides_claimed_extension(data, expected):
    assert sniff_audio_ext(data, ".ogg" if expected != ".ogg" else ".mp3") == expected


def test_unknown_audio_falls_back_and_normalizes_dot():
    assert sniff_audio_ext(UNKNOWN, "mp3") == ".mp3"
    for data in (OGG, FLAC, WAV, MP3_ID3, AAC_ADTS, M4A, MP4_ISOM, WEBM):
        assert sniff_container(data) in CONTAINER_TO_EXT


@pytest.mark.asyncio
async def test_cache_audio_from_bytes_repairs_claimed_extension(tmp_path, monkeypatch):
    from gateway.platforms import base

    monkeypatch.setattr(base, "get_hermes_home", lambda: tmp_path)
    path = await base.cache_audio_from_bytes(MP3_ID3, ext=".ogg")
    assert path.endswith(".mp3")
    assert (tmp_path / "cache" / "audio" / path.rsplit("/", 1)[-1]).read_bytes() == MP3_ID3


@pytest.mark.asyncio
async def test_cache_audio_from_bytes_returns_json_safe_path(tmp_path, monkeypatch):
    from gateway.platforms import base

    monkeypatch.setattr(base, "get_hermes_home", lambda: tmp_path)
    path = await base.cache_audio_from_bytes(WAV, ext="ogg")
    assert json.loads(json.dumps(path)) == path
    assert path.endswith(".wav")
