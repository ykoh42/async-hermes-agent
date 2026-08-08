"""Safety, size, deduplication, and invalidation guards for async file tools."""

import json

import pytest

import tools.file_tools as file_tools


@pytest.fixture(autouse=True)
def clean_read_state(monkeypatch):
    monkeypatch.setattr(file_tools, "_read_tracker", {})
    monkeypatch.setattr(file_tools, "_max_read_chars_cached", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/stdin",
        "/dev/tty",
        "/dev/console",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/fd/0",
        "/dev/fd/1",
        "/dev/fd/2",
        "/proc/self/fd/0",
        "/proc/12345/fd/2",
        "/proc/self/environ",
        "/proc/12345/cmdline",
        "/proc/self/maps",
        "/proc/1/smaps",
        "/proc/self/smaps_rollup",
        "/proc/99/numa_maps",
        "/proc/self/mem",
        "/proc/1/auxv",
        "/proc/99/pagemap",
        "/proc/self/task/1234/maps",
        "/proc/self/task/1234/smaps",
        "/proc/self/task/1234/auxv",
        "/proc/self/task/1234/pagemap",
        "/proc/self/task/1234/environ",
        "/dev/../dev/zero",
        "/dev/./urandom",
    ],
)
async def test_upstream_blocked_device_matrix(path):
    assert await file_tools._is_blocked_device(path), path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/dev/null",
        "/dev/sda1",
        "/proc/cpuinfo",
        "/proc/meminfo",
        "/proc/uptime",
        "/proc/version",
        "/tmp/test.py",
        "/home/user/.bashrc",
    ],
)
async def test_upstream_safe_device_matrix(path):
    assert not await file_tools._is_blocked_device(path), path


def test_upstream_proc_fd_other_is_not_pattern_blocked():
    assert not file_tools._is_blocked_device_path("/proc/self/fd/3")


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
async def test_regular_file_symlink_is_not_blocked(tmp_path):
    target = tmp_path / "regular.txt"
    target.write_text("safe\n")
    alias = tmp_path / "regular-link"
    alias.symlink_to(target)

    assert not await file_tools._is_blocked_device(str(alias))


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
async def test_block_persists_until_file_changes(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\n")

    await file_tools.read_file_tool(str(target), task_id="task")
    await file_tools.read_file_tool(str(target), task_id="task")
    assert "error" in json.loads(
        await file_tools.read_file_tool(str(target), task_id="task")
    )
    for _ in range(3):
        blocked = json.loads(
            await file_tools.read_file_tool(str(target), task_id="task")
        )
        assert blocked["error"].startswith("BLOCKED:")

    target.write_text("changed and longer\n")
    fresh = json.loads(await file_tools.read_file_tool(str(target), task_id="task"))
    assert fresh["content"] == "1|changed and longer"


@pytest.mark.asyncio
async def test_other_tool_call_resets_stub_hit_counter(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\n")
    await file_tools.read_file_tool(str(target), task_id="task")
    first_stub = json.loads(
        await file_tools.read_file_tool(str(target), task_id="task")
    )
    assert first_stub["status"] == "unchanged"

    file_tools.notify_other_tool_call("task")
    next_stub = json.loads(
        await file_tools.read_file_tool(str(target), task_id="task")
    )
    assert next_stub["status"] == "unchanged"
    assert "error" not in next_stub


@pytest.mark.asyncio
async def test_different_read_ranges_are_independent(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("one\ntwo\nthree\n")
    await file_tools.read_file_tool(
        str(target), offset=1, limit=1, task_id="task"
    )
    await file_tools.read_file_tool(
        str(target), offset=1, limit=1, task_id="task"
    )
    assert "error" in json.loads(
        await file_tools.read_file_tool(
            str(target), offset=1, limit=1, task_id="task"
        )
    )

    other = json.loads(
        await file_tools.read_file_tool(
            str(target), offset=2, limit=1, task_id="task"
        )
    )
    assert other["content"] == "2|two"


@pytest.mark.asyncio
async def test_reset_file_dedup_restores_full_read(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("alpha\n")
    await file_tools.read_file_tool(str(target), task_id="task")
    assert json.loads(
        await file_tools.read_file_tool(str(target), task_id="task")
    )["status"] == "unchanged"

    file_tools.reset_file_dedup("task")
    fresh = json.loads(await file_tools.read_file_tool(str(target), task_id="task"))
    assert fresh["content"] == "1|alpha"


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
