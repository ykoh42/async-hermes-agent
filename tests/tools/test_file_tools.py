"""Native-async behavior contracts for the local file tools."""

from __future__ import annotations

import json

import pytest

from tools.file_tools import PATCH_SCHEMA


@pytest.mark.asyncio
async def test_read_write_and_empty_content_round_trip(tmp_path):
    from tools.file_tools import read_file_tool, write_file_tool

    target = tmp_path / "sample.txt"
    written = json.loads(await write_file_tool(str(target), "line1\nline2\n"))
    assert written["resolved_path"] == str(target)

    read = json.loads(await read_file_tool(str(target)))
    assert read["content"] == "1|line1\n2|line2"
    assert read["total_lines"] == 2

    truncated = json.loads(await write_file_tool(str(target), ""))
    assert truncated["bytes_written"] == 0
    assert target.read_text() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,field",
    [({"path": "/tmp/x"}, "content"), ({"content": "x"}, "path")],
)
async def test_write_handler_rejects_missing_fields(args, field):
    from tools.file_tools import _handle_write_file

    result = json.loads(await _handle_write_file(args))
    assert field in result["error"]


@pytest.mark.asyncio
async def test_write_rejects_non_string_and_read_display(tmp_path):
    from tools.file_tools import _handle_write_file, write_file_tool

    bad_type = json.loads(
        await _handle_write_file({"path": str(tmp_path / "x"), "content": {"x": 1}})
    )
    assert "string" in bad_type["error"]

    display = json.loads(
        await write_file_tool(str(tmp_path / "config.yaml"), " 1|a: b\n 2|c: d\n")
    )
    assert "internal read_file display" in display["error"]


@pytest.mark.asyncio
async def test_patch_replace_and_no_match_hint(tmp_path):
    from tools.file_tools import patch_tool

    target = tmp_path / "app.py"
    target.write_text("foo\n")
    changed = json.loads(
        await patch_tool(
            mode="replace", path=str(target), old_string="foo", new_string="bar"
        )
    )
    assert changed["success"] is True
    assert target.read_text() == "bar\n"

    missing = json.loads(
        await patch_tool(
            mode="replace", path=str(target), old_string="absent", new_string="x"
        )
    )
    assert "read_file" in missing["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["Update", "Add", "Move"])
async def test_v4a_rejects_traversal_before_mutation(operation):
    from tools.file_tools import patch_tool

    if operation == "Move":
        header = "*** Move File: safe.txt -> ../../../tmp/dropped.py"
    else:
        header = f"*** {operation} File: ../../../tmp/dropped.py"
    result = json.loads(
        await patch_tool(
            mode="patch",
            patch=f"*** Begin Patch\n{header}\n+x\n*** End Patch\n",
        )
    )
    assert "traversal" in result["error"].lower()


@pytest.mark.asyncio
async def test_sensitive_paths_and_config_are_blocked(tmp_path, monkeypatch):
    from tools.file_tools import write_file_tool

    fake_config = tmp_path / "config.yaml"
    monkeypatch.setattr("tools.file_tools._hermes_config_resolved", str(fake_config))
    monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)

    config_result = json.loads(await write_file_tool(str(fake_config), "approvals: off"))
    assert "Hermes config" in config_result["error"]
    system_result = json.loads(await write_file_tool("/etc/passwd", "no"))
    assert "sensitive system path" in system_result["error"]


@pytest.mark.asyncio
async def test_search_uses_native_subprocess_and_paginates(tmp_path):
    from tools.file_tools import search_tool

    (tmp_path / "a.py").write_text("TODO one\n")
    (tmp_path / "b.py").write_text("TODO two\n")
    result = json.loads(
        await search_tool(pattern="TODO", path=str(tmp_path), limit=1, task_id="search")
    )
    assert len(result["matches"]) == 1
    assert result["truncated"] is True
    assert result["next_offset"] == 1


@pytest.mark.asyncio
async def test_relative_write_uses_recorded_session_cwd(tmp_path):
    import tools.terminal_tool as terminal
    from tools.file_tools import write_file_tool

    task_id = "cwd-round-trip"
    terminal.record_session_cwd(task_id, str(tmp_path))
    try:
        result = json.loads(await write_file_tool("report.txt", "hello\n", task_id))
        assert result["resolved_path"] == str(tmp_path / "report.txt")
        assert (tmp_path / "report.txt").read_text() == "hello\n"
    finally:
        await terminal.cleanup_vm(task_id)
        terminal.clear_session_cwd(task_id)


def test_patch_schema_documents_mode_specific_contracts():
    description = PATCH_SCHEMA["description"]
    assert "REQUIRED PARAMETERS: mode, path, old_string, new_string" in description
    assert "REQUIRED PARAMETERS: mode, patch" in description
    assert PATCH_SCHEMA["parameters"]["required"] == ["mode"]
