"""Tests for magic-byte type disclosure in binary-file refusals."""

import json
import random

import pytest

from tools.file_operations import describe_binary_file, identify_binary_bytes
from tools.file_tools import read_file_tool


_RNG = random.Random(20260810)


def _noise(n: int) -> bytes:
    return bytes(_RNG.getrandbits(8) for _ in range(n))


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (b"\x89PNG\r\n\x1a\n", "PNG image data"),
        (b"\xff\xd8\xff", "JPEG image data"),
        (b"%PDF-", "PDF document"),
        (b"PK\x03\x04", "ZIP archive"),
        (b"\x1f\x8b", "gzip compressed data"),
        (b"\x7fELF", "ELF executable"),
        (b"MZ", "Windows PE executable"),
        (b"SQLite format 3\x00", "SQLite database"),
    ],
)
def test_known_binary_signatures(prefix, expected):
    assert expected in identify_binary_bytes(prefix + _noise(64))


def test_unknown_binary_and_empty_sample():
    assert identify_binary_bytes(b"\x00\x01\x02" * 100) == "unknown binary"
    assert identify_binary_bytes(b"") == "unknown binary"


def test_iso_media_requires_ftyp():
    assert identify_binary_bytes(b"\x00\x00\x00\x18ftypmp42" + _noise(32)).startswith("ISO media")
    assert identify_binary_bytes(b"\x00\x00\x00\x18AAAA" + _noise(32)) == "unknown binary"


def test_binary_size_units():
    assert "512 bytes" in describe_binary_file(b"\x7fELF", 512)
    assert "4.0 KB" in describe_binary_file(b"\x7fELF", 4096)
    assert "2.0 MB" in describe_binary_file(b"\x7fELF", 2 * 1024 * 1024)
    assert "unknown binary" in describe_binary_file(None, 100)


@pytest.mark.asyncio
async def test_lying_extension_names_real_type(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    p = tmp_path / "data.txt"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + _noise(4096))
    result = json.loads(await read_file_tool(str(p)))
    assert "PNG image data" in result.get("error", "")
    assert "4.1 KB" in result["error"] or "4.0 KB" in result["error"]


@pytest.mark.asyncio
async def test_extensionless_binary_names_real_type(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    p = tmp_path / "mystery"
    p.write_bytes(b"\x7fELF" + _noise(1024))
    result = json.loads(await read_file_tool(str(p)))
    assert "ELF executable" in result.get("error", "")


@pytest.mark.asyncio
async def test_unknown_binary_is_still_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    p = tmp_path / "junk.out"
    p.write_bytes(b"\x00\x01\x02" * 500)
    result = json.loads(await read_file_tool(str(p)))
    assert "unknown binary" in result.get("error", "")


@pytest.mark.asyncio
async def test_text_file_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    p = tmp_path / "ok.txt"
    p.write_text("hello world\n")
    result = json.loads(await read_file_tool(str(p)))
    assert "hello world" in result.get("content", "")
