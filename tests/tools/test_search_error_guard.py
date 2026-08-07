"""Behavior tests for native-async rg diagnostics and payload handling."""

import json
import shutil

import pytest

from tools.file_tools import _pattern_has_regex_newline


pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="rg unavailable")


@pytest.mark.asyncio
async def test_happy_path_returns_all_matches(tmp_path):
    from tools.file_tools import search_tool

    for index in range(5):
        (tmp_path / f"f{index}.txt").write_text(f"needle line {index}\n")

    result = json.loads(
        await search_tool("needle", path=str(tmp_path), task_id="rg-happy")
    )

    assert "error" not in result
    assert result["total_count"] == 5


@pytest.mark.asyncio
async def test_invalid_regex_is_surfaced(tmp_path):
    from tools.file_tools import search_tool

    result = json.loads(await search_tool("[", path=str(tmp_path), task_id="rg-bad"))

    assert "Search failed" in result["error"]
    assert "matches" not in result


def test_regex_newline_detection_preserves_upstream_contract():
    assert _pattern_has_regex_newline(r"needle\n")
    assert _pattern_has_regex_newline("needle\\\\\\n")


@pytest.mark.asyncio
async def test_literal_backslash_n_has_no_warning(tmp_path):
    from tools.file_tools import search_tool

    result = json.loads(
        await search_tool(
            r"absent\\npattern",
            path=str(tmp_path),
            task_id="rg-literal-newline",
        )
    )

    assert result["total_count"] == 0
    assert "warning" not in result
