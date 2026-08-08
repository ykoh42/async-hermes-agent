"""Tests for write_file post-write content verification (verified flag)."""

import json
from unittest.mock import AsyncMock, patch as mock_patch

import pytest

from tools.file_tools import write_file_tool


def test_write_schema_explains_verified_result_contract():
    from tools.file_tools import WRITE_FILE_SCHEMA

    description = WRITE_FILE_SCHEMA["description"]
    assert "verified:true" in description
    assert "do NOT re-read" in description


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


@pytest.mark.asyncio
class TestWriteVerification:
    async def test_successful_write_reports_verified(self, workdir):
        f = workdir / "out.txt"
        r = json.loads(await write_file_tool(str(f), "hello verified world\n", task_id="t-wv"))
        assert r.get("bytes_written") == len("hello verified world\n")
        assert r.get("verified") is True

    async def test_unicode_content_verified(self, workdir):
        f = workdir / "uni.txt"
        content = "línea → uno · ✓\n"
        r = json.loads(await write_file_tool(str(f), content, task_id="t-wv"))
        assert r.get("verified") is True

    async def test_crlf_preservation_still_verifies(self, workdir):
        # Existing CRLF file: write_file converts LF content to CRLF before
        # writing; verification hashes the shim-adjusted content, so it must
        # still report verified.
        f = workdir / "win.txt"
        f.write_bytes(b"old line\r\n")
        r = json.loads(await write_file_tool(str(f), "new line\nsecond\n", task_id="t-wv"))
        assert "error" not in r
        assert r.get("verified") is True
        assert b"\r\n" in f.read_bytes()

    async def test_hash_mismatch_is_hard_error(self, workdir):
        f = workdir / "bad.txt"
        with mock_patch(
            "tools.file_tools._verify_native_file",
            new_callable=AsyncMock,
            return_value=False,
        ):
            r = json.loads(await write_file_tool(
                str(f), "actual content\n", task_id="t-wv"
            ))
        assert "error" in r
        assert "did not persist" in r["error"]

    async def test_verification_failure_never_breaks_write(self, workdir):
        # sha256sum unavailable/failing -> verified omitted, write still ok.
        f = workdir / "ok.txt"
        with mock_patch(
            "tools.file_tools._verify_native_file",
            new_callable=AsyncMock,
            side_effect=OSError("read-back unavailable"),
        ):
            r = json.loads(await write_file_tool(
                str(f), "content lands anyway\n", task_id="t-wv2"
            ))
        assert "error" not in r
        assert f.read_text() == "content lands anyway\n"
        assert "verified" not in r or r.get("verified") is None
