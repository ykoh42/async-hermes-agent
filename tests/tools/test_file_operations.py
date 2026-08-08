"""Contracts retained from the v2026.8.3 file-operations implementation."""

import json
import os

import pytest

from agent.file_safety import is_write_denied

from tools.file_operations import (
    LintResult,
    PatchResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    WriteResult,
    normalize_read_pagination,
)


def test_result_dataclasses_preserve_upstream_shapes():
    assert "error" not in ReadResult().to_dict()
    assert "similar_files" not in ReadResult().to_dict()
    assert WriteResult(bytes_written=100).to_dict()["bytes_written"] == 100
    assert PatchResult(success=True, files_modified=["a.py"]).to_dict() == {
        "success": True,
        "files_modified": ["a.py"],
    }
    assert LintResult(skipped=True, message="none").to_dict() == {
        "status": "skipped",
        "message": "none",
    }


def test_search_result_densification_preserves_paths_with_spaces():
    result = SearchResult(
        matches=[
            SearchMatch("dir with spaces/a.py", line, f"value {line}")
            for line in range(1, 6)
        ],
        total_count=5,
    ).to_dict(densify=True)

    assert result["matches_text"].startswith("dir with spaces/a.py\n")
    assert "matches" not in result


def test_read_pagination_clamps_invalid_values():
    offset, limit = normalize_read_pagination("bad", -5)
    assert offset == 1
    assert limit == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_path",
    [
        ".ssh/authorized_keys",
        ".netrc",
        ".pgpass",
        ".npmrc",
        ".pypirc",
        ".aws/credentials",
    ],
)
async def test_upstream_sensitive_home_paths_are_write_denied(
    tmp_path, monkeypatch, relative_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert await is_write_denied(str(tmp_path / relative_path))


@pytest.mark.asyncio
async def test_upstream_oauth_path_is_write_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert await is_write_denied(str(tmp_path / ".anthropic_oauth.json"))


@pytest.mark.asyncio
async def test_upstream_profile_and_root_token_pairing_paths_are_denied(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    for path in (
        profile / "mcp-tokens" / "tok.json",
        root / "mcp-tokens" / "tok.json",
        root / "mcp-tokens",
        profile / "pairing" / "telegram-approved.json",
        profile / "pairing",
        root / "pairing" / "telegram-approved.json",
        root / "pairing",
    ):
        assert await is_write_denied(str(path)), path


@pytest.mark.asyncio
@pytest.mark.parametrize("test_umask", [0o022, 0o002, 0o077])
async def test_new_file_uses_umask_default_permissions(tmp_path, test_umask):
    from tools.file_tools import write_file_tool

    target = tmp_path / "new_file.txt"
    old_umask = os.umask(test_umask)
    try:
        result = json.loads(await write_file_tool(str(target), "test content\n"))
    finally:
        os.umask(old_umask)

    assert "error" not in result
    assert target.stat().st_mode & 0o777 == 0o666 & ~test_umask


@pytest.mark.asyncio
async def test_atomic_write_follows_symlink_and_preserves_link(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "real.txt"
    link = tmp_path / "link.txt"
    target.write_text("original\n")
    link.symlink_to(target)

    result = json.loads(await write_file_tool(str(link), "updated\n"))

    assert "error" not in result
    assert link.is_symlink()
    assert target.read_text() == "updated\n"
    assert os.path.realpath(link) == str(target)


@pytest.mark.asyncio
async def test_atomic_write_through_broken_symlink_creates_target(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "target.txt"
    link = tmp_path / "broken.lnk"
    link.symlink_to(target)

    result = json.loads(await write_file_tool(str(link), "data\n"))

    assert "error" not in result
    assert link.is_symlink()
    assert target.read_text() == "data\n"


@pytest.mark.asyncio
async def test_overwrite_preserves_existing_mode(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "executable.sh"
    target.write_text("old\n")
    target.chmod(0o751)

    result = json.loads(await write_file_tool(str(target), "new\n"))

    assert "error" not in result
    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.asyncio
async def test_non_utf8_content_is_reported_as_binary(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "latin1.txt"
    target.write_bytes(b"caf\xe9 r\xe9sum\xe9\n")

    result = json.loads(await read_file_tool(str(target)))

    assert result["is_binary"] is True
    assert "Binary file" in result["error"]


@pytest.mark.asyncio
async def test_plain_utf8_content_is_not_reported_as_binary(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "utf8.txt"
    target.write_text("café résumé\n")

    result = json.loads(await read_file_tool(str(target)))

    assert "error" not in result
    assert result.get("is_binary") is not True
    assert result["content"] == "1|café résumé"
