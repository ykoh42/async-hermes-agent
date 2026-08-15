"""Native-async behavior contracts for the local file tools."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.file_tools import PATCH_SCHEMA, READ_FILE_SCHEMA


@pytest.mark.asyncio
async def test_write_file_surfaces_native_lsp_diagnostics_after_write(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    service = MagicMock()
    service.enabled_for = AsyncMock(return_value=True)

    async def snapshot(path):
        assert path == str(target)
        assert target.read_text() == "value = 1\n"

    async def diagnostics(path, **kwargs):
        assert path == str(target)
        assert target.read_text() == "value = 'wrong'\n"
        assert kwargs["delta"] is True
        assert callable(kwargs["line_shift"])
        return [
            {
                "severity": 1,
                "message": "Type mismatch",
                "range": {
                    "start": {"line": 0, "character": 8},
                    "end": {"line": 0, "character": 15},
                },
                "source": "pyright",
            }
        ]

    service.snapshot_baseline = AsyncMock(side_effect=snapshot)
    service.get_diagnostics_sync = AsyncMock(side_effect=diagnostics)
    with patch(
        "agent.lsp.get_service", new=AsyncMock(return_value=service)
    ):
        result = json.loads(
            await write_file_tool(str(target), "value = 'wrong'\n")
        )

    assert "Type mismatch" in result["lsp_diagnostics"]
    service.snapshot_baseline.assert_awaited_once_with(str(target))
    service.get_diagnostics_sync.assert_awaited_once()


@pytest.mark.asyncio
async def test_write_file_lsp_cancellation_propagates_before_write(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    service = MagicMock()
    service.snapshot_baseline = AsyncMock(side_effect=asyncio.CancelledError)
    with patch(
        "agent.lsp.get_service", new=AsyncMock(return_value=service)
    ):
        with pytest.raises(asyncio.CancelledError):
            await write_file_tool(str(target), "value = 2\n")

    assert target.read_text() == "value = 1\n"


@pytest.mark.asyncio
async def test_patch_surfaces_native_lsp_diagnostics(tmp_path):
    from tools.file_tools import patch_tool

    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    service = MagicMock()
    service.snapshot_baseline = AsyncMock()
    service.enabled_for = AsyncMock(return_value=True)
    service.get_diagnostics_sync = AsyncMock(
        return_value=[
            {
                "severity": 1,
                "message": "Semantic warning",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 5},
                },
            }
        ]
    )
    with patch(
        "agent.lsp.get_service", new=AsyncMock(return_value=service)
    ):
        result = json.loads(
            await patch_tool(
                mode="replace",
                path=str(target),
                old_string="value = 1",
                new_string="value = 2",
            )
        )

    assert "Semantic warning" in result["lsp_diagnostics"]


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
async def test_read_file_public_default_matches_upstream_contract(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "long.txt"
    target.write_text("\n".join(f"line-{index}" for index in range(600)))

    result = json.loads(await read_file_tool(str(target)))

    assert result["total_lines"] == 599
    assert result["content"].splitlines()[-1] == "600|line-599"
    assert READ_FILE_SCHEMA["parameters"]["properties"]["limit"]["default"] == 2000


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
async def test_patch_replace_and_no_match_error(tmp_path):
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
    assert "Could not find" in missing["error"]


@pytest.mark.asyncio
async def test_read_file_suggests_similar_filenames(tmp_path):
    from tools.file_tools import read_file_tool

    (tmp_path / "config.yaml").write_text("enabled: true\n")
    result = json.loads(await read_file_tool(str(tmp_path / "config.yml")))

    assert result["error"].startswith("File not found:")
    assert result["similar_files"] == [str(tmp_path / "config.yaml")]


@pytest.mark.asyncio
async def test_structured_write_fails_closed_without_touching_disk(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "config.json"
    target.write_text('{"valid": true}\n')

    result = json.loads(await write_file_tool(str(target), '{"broken":'))

    assert "Refusing to write" in result["error"]
    assert target.read_text() == '{"valid": true}\n'


@pytest.mark.asyncio
async def test_python_write_reports_new_syntax_error(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "module.py"
    target.write_text("value = 1\n")

    result = json.loads(await write_file_tool(str(target), "def broken(:\n"))

    assert result["lint"]["status"] == "error"
    assert "SyntaxError" in result["lint"]["output"]
    assert target.read_text() == "def broken(:\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["write", "patch"])
async def test_syntax_failure_skips_post_write_lsp_query(tmp_path, operation):
    from tools.file_tools import patch_tool, write_file_tool

    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    service = MagicMock()
    service.snapshot_baseline = AsyncMock()
    service.enabled_for = AsyncMock(return_value=True)
    service.get_diagnostics_sync = AsyncMock(return_value=[])

    with patch("agent.lsp.get_service", new=AsyncMock(return_value=service)):
        if operation == "write":
            result = json.loads(await write_file_tool(str(target), "def broken(:\n"))
        else:
            result = json.loads(
                await patch_tool(
                    mode="replace",
                    path=str(target),
                    old_string="value = 1",
                    new_string="def broken(:",
                )
            )

    assert result["lint"]["status"] == "error"
    service.snapshot_baseline.assert_awaited_once_with(str(target))
    service.get_diagnostics_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_and_patch_preserve_bom_and_mode(tmp_path):
    from tools.file_tools import patch_tool, write_file_tool

    target = tmp_path / "script.py"
    target.write_bytes(b"\xef\xbb\xbfvalue = 1\n")
    target.chmod(0o750)

    await write_file_tool(str(target), "value = 2\n")
    assert target.read_bytes() == b"\xef\xbb\xbfvalue = 2\n"
    assert target.stat().st_mode & 0o777 == 0o750

    result = json.loads(
        await patch_tool(
            mode="replace",
            path=str(target),
            old_string="value = 2",
            new_string="value = 3",
        )
    )
    assert result["success"] is True
    assert "replacements" not in result
    assert "-value = 2" in result["diff"]
    assert "+value = 3" in result["diff"]
    assert target.read_bytes() == b"\xef\xbb\xbfvalue = 3\n"
    assert target.stat().st_mode & 0o777 == 0o750


@pytest.mark.asyncio
async def test_patch_replace_uses_upstream_fuzzy_matching(tmp_path):
    from tools.file_tools import patch_tool

    target = tmp_path / "fuzzy.py"
    target.write_text("def greet():\n    return 'hello'\n")

    result = json.loads(
        await patch_tool(
            mode="replace",
            path=str(target),
            old_string="def greet():\n  return 'hello'",
            new_string="def greet():\n  return 'hi'",
        )
    )

    assert result["success"] is True
    assert "return 'hi'" in target.read_text()


@pytest.mark.asyncio
async def test_v4a_update_preserves_bom_and_reports_lint(tmp_path):
    from tools.file_tools import patch_tool

    target = tmp_path / "module.py"
    target.write_bytes(b"\xef\xbb\xbfvalue = 1\n")

    result = json.loads(
        await patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                f"*** Update File: {target}\n"
                "@@\n"
                "-value = 1\n"
                "+value = 2\n"
                "*** End Patch\n"
            ),
        )
    )

    assert result["success"] is True
    # v2026.8.3 lints the raw BOM-marked text after V4A writes, so ast.parse
    # reports the retained marker even though the write itself is valid.
    assert result["lint"][str(target)]["status"] == "error"
    assert target.read_bytes() == b"\xef\xbb\xbfvalue = 2\n"


@pytest.mark.asyncio
async def test_v4a_add_rejects_invalid_structured_content(tmp_path):
    from tools.file_tools import patch_tool

    target = tmp_path / "broken.json"
    result = json.loads(
        await patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                f"*** Add File: {target}\n"
                '+{"broken":\n'
                "*** End Patch\n"
            ),
        )
    )

    assert result["success"] is False
    assert "Refusing to write" in result["error"]
    assert not target.exists()


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
async def test_ssh_config_write_uses_native_approval_gate(tmp_path, monkeypatch):
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".ssh" / "config"
    approval = AsyncMock(return_value={"approved": True, "message": None})
    monkeypatch.setattr("tools.approval.request_tool_approval", approval)

    result = json.loads(await write_file_tool(str(target), "Host example\n"))

    assert result["bytes_written"] > 0
    assert target.read_text() == "Host example\n"
    approval.assert_awaited_once()
    assert approval.await_args.kwargs["rule_key"] == "ssh_config_write"


@pytest.mark.asyncio
async def test_ssh_config_write_fails_closed_without_approval(tmp_path, monkeypatch):
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".ssh" / "config"
    monkeypatch.setattr(
        "tools.approval.request_tool_approval",
        AsyncMock(return_value={"approved": False, "message": "BLOCKED"}),
    )

    result = json.loads(await write_file_tool(str(target), "Host denied\n"))

    assert result["error"] == "BLOCKED"
    assert not target.exists()


@pytest.mark.asyncio
async def test_search_uses_native_subprocess_and_paginates(tmp_path):
    from tools.file_tools import search_tool

    (tmp_path / "a.py").write_text("TODO one\n")
    (tmp_path / "b.py").write_text("TODO two\n")
    output = await search_tool(
        pattern=".py",
        target="files",
        path=str(tmp_path),
        limit=1,
        task_id="search",
    )
    payload, hint = output.split("\n\n", 1)
    result = json.loads(payload)
    assert len(result["files"]) == 1
    assert result["truncated"] is True
    assert "offset=1" in hint


@pytest.mark.asyncio
async def test_search_preserves_upstream_match_shape(tmp_path, monkeypatch):
    from tools.file_tools import search_tool

    async def fake_run_rg(_arguments):
        return 0, "a.py:7:TODO item\n", ""

    monkeypatch.setattr("tools.file_tools._run_rg", fake_run_rg)
    result = json.loads(
        await search_tool("TODO", path=str(tmp_path), task_id="search-shape")
    )

    assert result == {
        "total_count": 1,
        "matches": [{"path": "a.py", "line": 7, "content": "TODO item"}],
    }


@pytest.mark.asyncio
async def test_search_keeps_partial_results_on_rg_error(tmp_path, monkeypatch):
    from tools.file_tools import search_tool

    async def fake_run_rg(_arguments):
        return 2, "a.py:3:needle\n", "rg: locked.txt: Permission denied\n"

    monkeypatch.setattr("tools.file_tools._run_rg", fake_run_rg)
    result = json.loads(
        await search_tool("needle", path=str(tmp_path), task_id="search-partial")
    )

    assert result["matches"] == [
        {"path": "a.py", "line": 3, "content": "needle"}
    ]
    assert "error" not in result


@pytest.mark.asyncio
async def test_search_timeout_returns_partial_results(tmp_path, monkeypatch):
    from tools.file_tools import search_tool

    async def fake_run_rg(_arguments):
        return 124, "a.py:3:needle\n", ""

    monkeypatch.setattr("tools.file_tools._run_rg", fake_run_rg)
    output = await search_tool(
        "needle",
        path=str(tmp_path),
        task_id="search-timeout",
    )
    payload, hint = output.split("\n\n", 1)
    result = json.loads(payload)

    assert result["matches"] == [
        {"path": "a.py", "line": 3, "content": "needle"}
    ]
    assert result["limit_reason"] == "search_timeout"
    assert result["truncated"] is True
    assert "offset=50" in hint


@pytest.mark.asyncio
async def test_search_cancellation_kills_and_drains_process(monkeypatch):
    from tools.file_tools import _run_rg

    killed = asyncio.Event()
    communicate_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_completed = asyncio.Event()

    class FakeProcess:
        returncode = None

        async def communicate(self):
            communicate_started.set()
            await killed.wait()
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_completed.set()
            return b"", b""

        def kill(self):
            self.returncode = -9
            killed.set()

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    task = asyncio.create_task(_run_rg(["needle", "."]))
    await communicate_started.wait()
    task.cancel()
    await cleanup_started.wait()
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cleanup_completed.wait(), timeout=1.0)
    assert killed.is_set()


@pytest.mark.asyncio
async def test_search_communicate_failure_reaps_process(monkeypatch):
    from tools.file_tools import _run_rg

    waited = asyncio.Event()

    class FakeProcess:
        returncode = None
        killed = False

        async def communicate(self):
            raise RuntimeError("pipe failed")

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            waited.set()
            return self.returncode

    process = FakeProcess()

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match="pipe failed"):
        await _run_rg(["needle", "."])
    assert process.killed is True
    assert waited.is_set()


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
