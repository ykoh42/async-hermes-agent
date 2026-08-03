"""Loop damping for repeated async read and search tool calls."""

import json

import pytest

import tools.file_tools as file_tools


@pytest.fixture(autouse=True)
def clean_read_state(monkeypatch):
    monkeypatch.setattr(file_tools, "_read_tracker", {})


@pytest.mark.asyncio
async def test_repeated_read_warns_then_blocks_when_dedup_is_reset(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\nbeta\n")
    task_id = "loop"

    results = []
    for _ in range(4):
        file_tools.reset_file_dedup(task_id)
        results.append(
            json.loads(await file_tools.read_file_tool(str(target), task_id=task_id))
        )

    assert "_warning" not in results[0]
    assert "_warning" not in results[1]
    assert "_warning" in results[2]
    assert results[3]["error"].startswith("BLOCKED:")


@pytest.mark.asyncio
async def test_other_tool_call_breaks_consecutive_read_loop(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\n")
    task_id = "loop"
    for _ in range(3):
        file_tools.reset_file_dedup(task_id)
        await file_tools.read_file_tool(str(target), task_id=task_id)

    file_tools.notify_other_tool_call(task_id)
    file_tools.reset_file_dedup(task_id)
    result = json.loads(await file_tools.read_file_tool(str(target), task_id=task_id))

    assert "error" not in result
    assert "_warning" not in result


@pytest.mark.asyncio
async def test_pagination_ranges_are_tracked_independently(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("one\ntwo\nthree\n")
    task_id = "pages"

    first = json.loads(
        await file_tools.read_file_tool(str(target), offset=1, limit=1, task_id=task_id)
    )
    second = json.loads(
        await file_tools.read_file_tool(str(target), offset=2, limit=1, task_id=task_id)
    )

    assert first["content"] == "1|one"
    assert second["content"] == "2|two"


@pytest.mark.asyncio
async def test_repeated_search_is_damped(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("needle\n")
    task_id = "search-loop"

    results = [
        json.loads(
            await file_tools.search_tool(
                "needle", path=str(tmp_path), task_id=task_id
            )
        )
        for _ in range(4)
    ]

    assert results[0]["matches"]
    assert "_warning" in results[2]
    assert results[3]["error"].startswith("BLOCKED:")


@pytest.mark.asyncio
async def test_search_offset_changes_loop_key(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("needle\nneedle\n")
    task_id = "search-pages"

    first = json.loads(
        await file_tools.search_tool("needle", path=str(tmp_path), offset=0, task_id=task_id)
    )
    second = json.loads(
        await file_tools.search_tool("needle", path=str(tmp_path), offset=1, task_id=task_id)
    )

    assert first["matches"]
    assert second["matches"]
    assert "_warning" not in second
