"""Contracts retained from the v2026.8.3 file-operations implementation."""

import json
import os

import pytest

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
async def test_non_utf8_content_is_reported_as_binary(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "latin1.txt"
    target.write_bytes(b"caf\xe9 r\xe9sum\xe9\n")

    result = json.loads(await read_file_tool(str(target)))

    assert result["is_binary"] is True
    assert "Binary file" in result["error"]
