"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import json
import os
import shlex
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import aiofiles
import pytest

from agent.agent_runtime_helpers import convert_to_trajectory_format
from tools.ansi_strip import strip_ansi
from tools.environments.base import BaseEnvironment, _BoundedOutputCollector
from tools.environments.local import LocalEnvironment
from tools.terminal_tool import terminal_tool


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    async def load_config():
        return {
            "tool_output": {
                "max_bytes": 2000,
                "max_lines": 2000,
                "max_line_length": 2000,
            }
        }

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", load_config)
    return tmp_path


@pytest.mark.asyncio
class TestTruncationSpill:
    async def test_truncated_output_has_metadata_and_spill(self, small_cap):
        r = json.loads(await terminal_tool(
            "python3 -c \"print('marker_head'); [print(f'row_{i}', 'x'*80) for i in range(200)]; print('marker_tail')\"",
            task_id="t-spill-1"))
        assert r["exit_code"] == 0
        assert "OUTPUT TRUNCATED" in r["output"]
        assert r["output_total_chars"] > 2000
        p = Path(r["full_output_path"])
        assert p.exists()
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
        full = p.read_text()
        assert "marker_head" in full and "marker_tail" in full
        # The spill contains rows that were cut from the visible window.
        assert "row_100 " in full
        assert "read_file" in r["truncation_note"]

    async def test_small_output_has_no_metadata(self, small_cap):
        r = json.loads(await terminal_tool("echo tiny", task_id="t-spill-2"))
        assert r["exit_code"] == 0
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    async def test_spill_is_redacted(self, small_cap):
        r = json.loads(await terminal_tool(
            "python3 -c \"print('sk-proj-' + 'a1B2c3D4e5F6g7H8i9J0' * 3); [print('pad', 'y'*90) for i in range(200)]\"",
            task_id="t-spill-3"))
        p = Path(r["full_output_path"])
        full = p.read_text()
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in full

    async def test_old_spills_cleaned(self, small_cap, tmp_path):
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        stale = spill_dir / "out-1-2-dead.log"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        json.loads(await terminal_tool(
            "python3 -c \"[print('z'*90) for i in range(200)]\"", task_id="t-spill-4"))
        assert not stale.exists()

    async def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(await terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()


@pytest.mark.asyncio
async def test_local_bounded_capture_caps_the_raw_stream_while_draining(
    tmp_path,
    monkeypatch,
):
    """The foreground opt-in reaches Local's streaming collector, not a post-read cap."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    raw = "HEAD-" + "\x1b[31m" + ("x" * 240) + "\x1b[0m-TAIL"
    code = f"import sys; sys.stdout.write({raw!r})"
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=5)

    try:
        result = await environment._run_bash(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
            bounded_capture=True,
        )
    finally:
        await environment.cleanup()

    assert result["returncode"] == 0
    assert result["output_total_chars"] == len(raw)
    assert len(result["output"]) <= 100
    assert "[OUTPUT TRUNCATED" in result["output"]
    async with aiofiles.open(result["full_output_path"], encoding="utf-8") as handle:
        assert await handle.read() == raw


@pytest.mark.asyncio
async def test_terminal_ansi_heavy_bounded_observation_and_trajectory_parity(
    tmp_path,
    monkeypatch,
):
    """Preserve raw cap → ANSI strip → metadata ordering at a 100-char cap."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)

    async def no_refresh():
        return None

    monkeypatch.setattr(
        "tools.tool_output_limits._refresh_tool_output_limits",
        no_refresh,
    )
    raw = "HEAD-" + "\x1b[31m" + ("m" * 180) + "\x1b[0m-TAIL"
    spill = tmp_path / "raw-stream.log"
    collector = _BoundedOutputCollector(100, spill_path=spill)
    collector.append(raw)
    captured = await BaseEnvironment._finalize_wait_result(
        collector,
        collector.render(),
        0,
    )
    expected_visible = strip_ansi(captured["output"]).strip()
    execute_calls = []

    class Environment:
        cwd = str(tmp_path)

        async def execute(self, command, **kwargs):
            execute_calls.append((command, kwargs))
            return dict(captured)

    environment = Environment()

    async def env_config():
        return {"env_type": "local", "timeout": 5, "cwd": str(tmp_path)}

    async def get_environment(_task_id):
        return environment

    async def approve(*_args, **_kwargs):
        return {"approved": True}

    async def no_evidence(**_kwargs):
        return None

    monkeypatch.setattr("tools.terminal_tool._get_env_config", env_config)
    monkeypatch.setattr(
        "tools.terminal_tool._get_or_create_environment",
        get_environment,
    )
    monkeypatch.setattr("tools.terminal_tool.check_all_command_guards", approve)
    monkeypatch.setattr(
        "agent.verification_evidence.record_terminal_result",
        no_evidence,
    )

    observation = await terminal_tool("printf bounded", task_id="bounded-parity")
    payload = json.loads(observation)

    assert execute_calls == [
        (
            "printf bounded",
            {
                "cwd": str(tmp_path),
                    "timeout": 5,
                "bounded_capture": True,
            },
        )
    ]
    assert payload["output"] == expected_visible
    assert len(payload["output"]) < 100  # ANSI bytes counted before stripping.
    assert "\x1b" not in payload["output"]
    assert payload["output_total_chars"] == len(raw)
    assert payload["full_output_path"] == str(spill)
    assert payload["truncation_note"] == (
        "Output exceeded the capture window (head+tail shown). "
        f"Full output ({len(raw):,} chars) saved to {spill} — "
        "search it with search_files or page it with read_file instead of "
        "re-running the command."
    )
    async with aiofiles.open(spill, encoding="utf-8") as handle:
        assert await handle.read() == strip_ansi(raw)

    messages = [
        {"role": "user", "content": "run bounded output"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-terminal",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "printf bounded"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-terminal",
            "content": observation,
        },
    ]
    agent = SimpleNamespace(_format_tools_for_system_message=lambda: "[]")
    trajectory = convert_to_trajectory_format(
        agent,
        messages,
        "run bounded output",
        completed=True,
    )
    expected_tool_value = (
        "<tool_response>\n"
        + json.dumps(
            {
                "tool_call_id": "call-terminal",
                "name": "terminal",
                "content": payload,
            },
            ensure_ascii=False,
        )
        + "\n</tool_response>"
    )
    assert trajectory[3] == {"from": "tool", "value": expected_tool_value}
