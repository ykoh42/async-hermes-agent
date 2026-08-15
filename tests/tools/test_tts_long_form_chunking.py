"""Long-form TTS chunking/delivery tests at the native async boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.tts_tool import (
    AudioDeliveryProfile,
    _build_audio_delivery_files,
    _pack_audio_files_for_delivery,
    _resolve_audio_delivery_profile,
    _split_oversized_sentence,
    _split_text_for_tts,
)


def test_short_and_empty_text():
    assert _split_text_for_tts("Hello world.", 4096) == ["Hello world."]
    assert _split_text_for_tts("", 4096) == []
    assert _split_text_for_tts("   ", 4096) == []


def test_long_text_is_split_without_loss():
    text = "A" * 5000
    chunks = _split_text_for_tts(text, 4096)
    assert chunks == ["A" * 4096, "A" * 904]
    assert "".join(chunks) == text


def test_sentence_and_very_long_word_splitting():
    text = "First sentence. Second sentence. Third sentence."
    chunks = _split_text_for_tts(text, 30)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
    word = "A" * 100
    chunks = _split_oversized_sentence(word, 30)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "".join(chunks) == word


def test_sentence_boundary_split_produces_multiple_chunks():
    chunks = _split_text_for_tts(
        "First sentence. Second sentence. Third sentence.",
        30,
    )
    assert len(chunks) >= 2
    assert "".join(chunks).replace(" ", "") == (
        "First sentence. Second sentence. Third sentence.".replace(" ", "")
    )


def test_short_oversized_sentence_is_unchanged():
    assert _split_oversized_sentence("Hello world.", 100) == ["Hello world."]


def test_oversized_sentence_word_boundary_split():
    chunks = _split_oversized_sentence(" ".join(["word"] * 50), 30)
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_audio_delivery_profile_validation():
    profile = AudioDeliveryProfile(platform="default", max_file_bytes=10_000)
    assert 0 < profile.target_file_bytes < profile.max_file_bytes
    assert _resolve_audio_delivery_profile(
        "custom", {"delivery_profiles": {"custom": {"max_file_bytes": 1000, "safety_ratio": 0.5}}}
    ).target_file_bytes == 500


def test_default_audio_delivery_profile():
    profile = _resolve_audio_delivery_profile("default")
    assert 0 < profile.target_file_bytes < profile.max_file_bytes


@pytest.mark.asyncio
async def test_pack_audio_files_respects_size_and_suffix(tmp_path):
    files = []
    for index in range(5):
        path = tmp_path / f"chunk{index:02d}.mp3"
        path.write_bytes(b"x" * 300)
        files.append(str(path))
    profile = AudioDeliveryProfile(platform="default", max_file_bytes=1000, safety_ratio=0.5)
    groups = await _pack_audio_files_for_delivery(files, profile)
    assert groups == [[path] for path in files]
    ogg = tmp_path / "other.ogg"
    ogg.write_bytes(b"x")
    assert await _pack_audio_files_for_delivery([files[0], str(ogg)], profile) == [[files[0]], [str(ogg)]]


@pytest.mark.asyncio
async def test_pack_audio_files_single_file_returns_one_group(tmp_path):
    path = tmp_path / "single.mp3"
    path.write_bytes(b"x" * 100)
    groups = await _pack_audio_files_for_delivery(
        [str(path)], AudioDeliveryProfile(platform="default", max_file_bytes=10_000)
    )
    assert groups == [[str(path)]]


@pytest.mark.asyncio
async def test_build_audio_delivery_files_single_and_oversized(tmp_path):
    source = tmp_path / "chunk.mp3"
    source.write_bytes(b"x" * 100)
    output = tmp_path / "output.mp3"
    profile = AudioDeliveryProfile(platform="default", max_file_bytes=10_000)
    paths, combined = await _build_audio_delivery_files([str(source)], str(output), profile)
    assert paths == [str(output)]
    assert combined is False
    assert output.read_bytes() == b"x" * 100

    oversized = tmp_path / "oversized.mp3"
    oversized.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="exceeds"):
        await _build_audio_delivery_files(
            [str(oversized)], str(tmp_path / "too-large.mp3"),
            AudioDeliveryProfile(platform="default", max_file_bytes=50),
        )


@pytest.mark.asyncio
async def test_build_audio_delivery_files_combines_multiple_chunks(tmp_path, monkeypatch):
    files = []
    for index in range(3):
        path = tmp_path / f"chunk{index:02d}.mp3"
        path.write_bytes(bytes([index]) * 100)
        files.append(str(path))
    output = tmp_path / "output.mp3"

    # Avoid relying on ffmpeg in this unit test while still exercising async
    # grouping, final placement, and cleanup.
    async def concat(paths, destination, *, voice_compatible=False):
        del voice_compatible
        data = b"".join(Path(path).read_bytes() for path in paths)
        Path(destination).write_bytes(data)
        return destination

    monkeypatch.setattr("tools.tts_tool._concat_audio_files", concat)
    paths, combined = await _build_audio_delivery_files(
        files,
        str(output),
        AudioDeliveryProfile(platform="default", max_file_bytes=10_000),
    )
    assert paths == [str(output)]
    assert combined is True
    assert output.read_bytes() == b"\x00" * 100 + b"\x01" * 100 + b"\x02" * 100


@pytest.mark.asyncio
async def test_public_wrapper_splits_and_preserves_chunk_order(tmp_path, monkeypatch):
    calls: list[str] = []

    async def load_config():
        return {}

    async def synthesize(*, text, output_path, **kwargs):
        del kwargs
        calls.append(text)
        Path(output_path).write_bytes(b"x")
        return json.dumps(
            {
                "success": True,
                "file_path": output_path,
                "media_tag": f"MEDIA:{output_path}",
                "provider": "fake",
                "voice_compatible": False,
            }
        )

    monkeypatch.setattr("tools.tts_tool._load_tts_config", load_config)
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda _config: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda *_args: 5)
    monkeypatch.setattr("tools.tts_tool._text_to_speech_single", synthesize)
    monkeypatch.setattr(
        "tools.tts_tool._resolve_audio_delivery_profile",
        lambda *_args: AudioDeliveryProfile(
            platform="default", max_file_bytes=1, safety_ratio=1.0
        ),
    )
    from tools import tts_tool

    result = json.loads(
        await tts_tool.text_to_speech_tool(
            "alpha bravo charlie",
            output_path=str(tmp_path / "speech.mp3"),
            provider="fake",
        )
    )
    assert result["success"] is True
    assert result["chunk_count"] == len(calls)
    assert calls == ["alpha", "bravo", "charl", "ie"]
    assert result["delivery_file_count"] == len(calls)
