"""Safety, size, deduplication, and invalidation guards for async file tools."""

import json

import pytest

import tools.file_tools as file_tools


@pytest.fixture(autouse=True)
def clean_read_state(monkeypatch):
    monkeypatch.setattr(file_tools, "_read_tracker", {})
    monkeypatch.setattr(file_tools, "_max_read_chars_cached", None)


@pytest.mark.asyncio
async def test_device_file_is_rejected():
    result = json.loads(await file_tools.read_file_tool("/dev/zero"))
    assert "device" in result["error"].lower()


@pytest.mark.asyncio
async def test_device_symlink_is_rejected(tmp_path):
    alias = tmp_path / "device"
    alias.symlink_to("/dev/zero")
    result = json.loads(await file_tools.read_file_tool(str(alias)))
    assert "device" in result["error"].lower()


@pytest.mark.asyncio
async def test_character_budget_truncates_on_line_boundary(tmp_path, monkeypatch):
    target = tmp_path / "large.txt"
    target.write_text("\n".join(f"line-{index:03d}" for index in range(100)))

    async def small_budget():
        return 80

    monkeypatch.setattr(file_tools, "_get_max_read_chars", small_budget)
    result = json.loads(await file_tools.read_file_tool(str(target), limit=100))

    assert result["truncated"] is True
    assert result["truncated_by"] == "bytes"
    assert result["next_offset"] > 1
    assert len(result["content"]) <= 80
    assert "80-char read budget" in result["hint"]


@pytest.mark.asyncio
async def test_long_line_uses_upstream_truncation_marker(tmp_path, monkeypatch):
    target = tmp_path / "wide.txt"
    target.write_text("x" * 50)
    monkeypatch.setattr("tools.tool_output_limits.get_max_line_length", lambda: 10)

    result = json.loads(await file_tools.read_file_tool(str(target)))

    assert result["content"] == "1|xxxxxxxxxx... [truncated]"


@pytest.mark.asyncio
async def test_large_truncated_file_gets_targeted_read_hint(tmp_path):
    target = tmp_path / "large.txt"
    target.write_text("\n".join("x" * 1000 for _ in range(600)))

    result = json.loads(await file_tools.read_file_tool(str(target), limit=500))

    assert result["truncated"] is True
    assert "This file is large" in result["_hint"]


@pytest.mark.asyncio
async def test_configured_character_budget_is_loaded_asynchronously(monkeypatch):
    async def load_config():
        return {"file_read_max_chars": 12345}

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config,
    )
    assert await file_tools._get_max_read_chars() == 12345


@pytest.mark.asyncio
async def test_unchanged_read_returns_stub_then_blocks(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\nbeta\n")

    first = json.loads(await file_tools.read_file_tool(str(target), task_id="task"))
    second = json.loads(await file_tools.read_file_tool(str(target), task_id="task"))
    third = json.loads(await file_tools.read_file_tool(str(target), task_id="task"))

    assert "alpha" in first["content"]
    assert second["status"] == "unchanged"
    assert third["error"].startswith("BLOCKED:")


@pytest.mark.asyncio
async def test_dedup_state_is_scoped_per_task(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\n")
    await file_tools.read_file_tool(str(target), task_id="one")

    other = json.loads(await file_tools.read_file_tool(str(target), task_id="two"))

    assert "alpha" in other["content"]


@pytest.mark.asyncio
async def test_write_invalidates_all_cached_ranges(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("one\ntwo\nthree\n")
    await file_tools.read_file_tool(str(target), offset=1, limit=1, task_id="task")
    await file_tools.read_file_tool(str(target), offset=2, limit=1, task_id="task")

    await file_tools.write_file_tool(str(target), "changed\n", task_id="task")
    result = json.loads(
        await file_tools.read_file_tool(str(target), offset=1, limit=1, task_id="task")
    )

    assert result["content"] == "1|changed"


@pytest.mark.asyncio
async def test_internal_read_status_cannot_be_written(tmp_path):
    target = tmp_path / "data.txt"
    result = json.loads(
        await file_tools.write_file_tool(
            str(target),
            file_tools._READ_DEDUP_STATUS_MESSAGE,
        )
    )
    assert "internal" in result["error"].lower()


def test_line_numbered_read_output_cannot_be_rewritten_as_content():
    assert file_tools._is_internal_file_tool_content("1|alpha\n2|beta\n3|gamma")
