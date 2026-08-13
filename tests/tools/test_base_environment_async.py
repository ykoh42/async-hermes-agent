"""Native-async parity tests for the shared environment base contract."""

from __future__ import annotations

import asyncio
import os
import stat

import aiofiles
import pytest

from tools.environments.base import (
    BaseEnvironment,
    _BoundedOutputCollector,
    _export_dump_excluding_session_vars,
    set_activity_callback,
    touch_activity_if_due,
)


class _TestableEnvironment(BaseEnvironment):
    def __init__(self, cwd: str = "/tmp", timeout: int = 10):
        super().__init__(cwd=cwd, timeout=timeout)

    async def _run_bash(self, *_args, **_kwargs):
        raise NotImplementedError

    async def cleanup(self):
        return None


def test_large_stream_retains_bounded_head_and_tail():
    collector = _BoundedOutputCollector(1_000)
    collector.append("HEAD-SENTINEL\n")
    for _ in range(2_000):
        collector.append("x" * 4_096)
    collector.append("\nTAIL-SENTINEL")

    rendered = collector.render()

    assert collector.total_chars > 8_000_000
    assert collector.buffered_chars <= 1_000
    assert len(rendered) <= 1_000
    assert rendered.startswith("HEAD-SENTINEL")
    assert rendered.endswith("TAIL-SENTINEL")
    assert "[OUTPUT TRUNCATED" in rendered


def test_required_status_suffix_stays_inside_limit():
    collector = _BoundedOutputCollector(120)
    collector.append("A" * 10_000)

    rendered = collector.render(suffix="\n[Command timed out after 1s]")

    assert len(rendered) <= 120
    assert rendered.endswith("[Command timed out after 1s]")
    assert "[OUTPUT TRUNCATED" in rendered


def test_wrap_command_preserves_snapshot_cwd_and_exit_contract():
    environment = _TestableEnvironment()
    environment._snapshot_ready = True

    wrapped = environment._wrap_command("echo 'hello world'", "/tmp")

    assert "source" in wrapped
    assert "builtin cd -- /tmp" in wrapped
    assert "eval 'echo '\\''hello world'\\'''" in wrapped
    assert "__hermes_ec=$?" in wrapped
    assert "export -p" in wrapped
    assert "mktemp " in wrapped
    assert ".tmp.XXXXXXXXXX" in wrapped
    assert "$BASHPID" not in wrapped
    assert environment._cwd_marker in wrapped
    assert "exit $__hermes_ec" in wrapped


def test_wrap_command_without_snapshot_skips_source():
    environment = _TestableEnvironment()
    environment._snapshot_ready = False

    wrapped = environment._wrap_command("echo hello", "/tmp")

    assert "source" not in wrapped


def test_embed_stdin_heredoc_uses_unique_delimiters():
    first = BaseEnvironment._embed_stdin_heredoc("cat", "hello")
    second = BaseEnvironment._embed_stdin_heredoc("cat", "hello")

    assert first.startswith("cat << '")
    assert "hello" in first
    assert first.split("'")[1] != second.split("'")[1]


def test_cwd_marker_is_extracted_and_removed():
    environment = _TestableEnvironment()
    marker = environment._cwd_marker
    result = {"output": f"hello\n{marker}/home/user{marker}\n"}

    environment._extract_cwd_from_output(result)

    assert environment.cwd == "/home/user"
    assert marker not in result["output"]
    assert result["output"] == "hello"


def test_activity_touch_preserves_upstream_elapsed_message(monkeypatch):
    messages: list[str] = []
    set_activity_callback(messages.append)
    monkeypatch.setattr("tools.environments.base.time.monotonic", lambda: 19.9)
    state = {"last_touch": 5.0, "start": 3.0, "interval": 10.0}

    touch_activity_if_due(state, "terminal command running")

    assert messages == ["terminal command running (16s elapsed)"]
    assert state["last_touch"] == 19.9
    set_activity_callback(None)


def test_export_dump_unsets_session_and_profile_names_before_dump():
    snippet = _export_dump_excluding_session_vars(
        '"$__hermes_snap_tmp"',
        ("SAFE_TOKEN", "name; touch /tmp/not-code"),
    )

    assert snippet.startswith("{ ( unset ")
    assert "${!HERMES_SESSION_*}" in snippet
    assert "${!HERMES_CRON_AUTO_DELIVER_*}" in snippet
    assert "HERMES_UI_SESSION_ID" in snippet
    assert "SAFE_TOKEN" in snippet
    assert "'name; touch /tmp/not-code'" in snippet
    assert "grep -vE" not in snippet
    assert snippet.endswith('> "$__hermes_snap_tmp"')


def test_wrap_command_restores_current_profile_values_and_excludes_redump():
    environment = _TestableEnvironment()
    environment._snapshot_ready = True
    environment._snapshot_passthrough_names.update({"PROFILE_TOKEN"})

    wrapped = environment._wrap_command("echo ok", "/tmp")

    save = "_HERMES_RUNTIME_PASSTHROUGH_PROFILE_TOKEN_PRESENT=${PROFILE_TOKEN+x}"
    restore = (
        'if [ "$_HERMES_RUNTIME_PASSTHROUGH_PROFILE_TOKEN_PRESENT" = x ]; '
        "then export PROFILE_TOKEN="
    )
    assert wrapped.index(save) < wrapped.index("source ")
    assert wrapped.index("source ") < wrapped.index(restore)
    assert "unset ${!HERMES_SESSION_*}" in wrapped
    assert "HERMES_UI_SESSION_ID PROFILE_TOKEN" in wrapped


@pytest.mark.asyncio
async def test_spill_is_lazily_persisted_with_full_stream(tmp_path):
    spill = tmp_path / "output.log"
    collector = _BoundedOutputCollector(8, spill_path=spill)
    collector.append("head")
    assert not spill.exists()
    collector.append("-middle-tail")

    result = await _TestableEnvironment._finalize_wait_result(
        collector,
        collector.render(),
        0,
    )

    assert result["full_output_path"] == str(spill)
    assert result["output_total_chars"] == len("head-middle-tail")
    assert spill.read_text() == "head-middle-tail"
    assert stat.S_IMODE(spill.stat().st_mode) == 0o600
    assert await collector.close_spill() == str(spill)


@pytest.mark.asyncio
async def test_spill_cap_and_no_overflow_lifecycle(tmp_path, monkeypatch):
    untouched = _BoundedOutputCollector(64, spill_path=tmp_path / "small.log")
    untouched.append("small")
    assert await untouched.close_spill() is None
    assert not (tmp_path / "small.log").exists()

    monkeypatch.setattr(_BoundedOutputCollector, "_SPILL_CAP_CHARS", 12)
    capped = _BoundedOutputCollector(4, spill_path=tmp_path / "capped.log")
    capped.append("abcdefghijklmnopqrstuvwxyz")
    assert await capped.close_spill() == str(tmp_path / "capped.log")
    contents = (tmp_path / "capped.log").read_text()
    assert contents.startswith("abcdefghijkl")
    assert contents.endswith(_BoundedOutputCollector._SPILL_CAP_MARKER)


@pytest.mark.asyncio
async def test_cancelled_spill_close_removes_raw_final_and_temp_under_recancellation(
    tmp_path,
    monkeypatch,
):
    spill = tmp_path / "raw-output.log"
    collector = _BoundedOutputCollector(4, spill_path=spill)
    collector.append("raw-secret-that-must-not-survive")
    replace_entered = asyncio.Event()
    release_replace = asyncio.Event()
    discard_entered = asyncio.Event()
    release_discard = asyncio.Event()
    real_replace = aiofiles.os.replace
    real_remove = aiofiles.os.remove

    async def gated_replace(source, destination):
        replace_entered.set()
        await release_replace.wait()
        await real_replace(source, destination)

    async def gated_remove(path):
        if path == spill and spill.exists():
            discard_entered.set()
            await release_discard.wait()
        await real_remove(path)

    monkeypatch.setattr("tools.environments.base.aiofiles.os.replace", gated_replace)
    monkeypatch.setattr("tools.environments.base.aiofiles.os.remove", gated_remove)
    close = asyncio.create_task(collector.close_spill())
    await replace_entered.wait()
    close.cancel()
    release_replace.set()
    await discard_entered.wait()
    close.cancel()
    await asyncio.sleep(0)
    close.cancel()
    release_discard.set()

    with pytest.raises(asyncio.CancelledError):
        await close

    assert not spill.exists()
    assert not list(tmp_path.glob(".raw-output.log.*.tmp"))
    assert not [
        task
        for task in asyncio.all_tasks()
        if not task.done()
        and task is not asyncio.current_task()
        and task.get_name().startswith("terminal-spill-")
    ]


@pytest.mark.asyncio
async def test_base_bounded_capture_returns_upstream_spill_metadata(
    tmp_path,
    monkeypatch,
):
    environment = _TestableEnvironment()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 12)

    result = await environment._apply_bounded_capture(
        {"output": "head-middle-tail", "returncode": 7}
    )

    assert result["returncode"] == 7
    assert len(result["output"]) <= 12
    assert result["output_total_chars"] == len("head-middle-tail")
    async with aiofiles.open(result["full_output_path"], encoding="utf-8") as handle:
        assert await handle.read() == "head-middle-tail"


@pytest.mark.asyncio
async def test_init_session_falls_back_to_nonlogin_and_execute_uses_it(monkeypatch):
    environment = _TestableEnvironment()
    calls: list[tuple[str, bool]] = []

    async def run(command, *, login=False, **_kwargs):
        calls.append((command, login))
        if login:
            return {"output": "login failed", "returncode": 1}
        return {"output": "", "returncode": 0}

    async def transform(command):
        return command, None

    monkeypatch.setattr(environment, "_run_bash", run)
    monkeypatch.setattr("tools.terminal_tool._transform_sudo_command", transform)

    await environment.init_session()
    assert environment._prefer_nonlogin is True
    await environment.execute("echo ok")

    assert calls[0][1] is True
    assert calls[1] == ("true", False)
    assert calls[2][1] is False


@pytest.mark.asyncio
async def test_initialization_cancellation_does_not_publish_ready_state(monkeypatch):
    environment = _TestableEnvironment()

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(environment, "_run_bash", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await environment.init_session()

    assert environment._initialized is False
    assert environment._snapshot_ready is False


@pytest.mark.asyncio
async def test_snapshot_runtime_profile_value_isolated_in_real_bash(tmp_path):
    snapshot = tmp_path / "snapshot.sh"
    async with aiofiles.open(snapshot, "w", encoding="utf-8") as handle:
        await handle.write(
            "export PROFILE_TOKEN=old-profile\n"
        )
    environment = _TestableEnvironment(cwd=str(tmp_path))
    environment._snapshot_path = str(snapshot)
    environment._snapshot_ready = True
    environment._snapshot_passthrough_names.add("PROFILE_TOKEN")
    wrapped = environment._wrap_command(
        'printf "%s|%s" "$PROFILE_TOKEN" "$HERMES_SESSION_ID"',
        str(tmp_path),
    )
    child_env = os.environ.copy()
    child_env["PROFILE_TOKEN"] = "current-profile"
    child_env["HERMES_SESSION_ID"] = "current-session"

    process = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        wrapped,
        cwd=tmp_path,
        env=child_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    assert process.returncode == 0, stderr.decode(errors="replace")
    assert stdout.decode(errors="replace").startswith(
        "current-profile|current-session"
    )
    async with aiofiles.open(snapshot, encoding="utf-8") as handle:
        persisted = await handle.read()
    assert "PROFILE_TOKEN" not in persisted
    assert "HERMES_SESSION_ID" not in persisted
