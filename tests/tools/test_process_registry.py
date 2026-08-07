"""Native-async contracts for the background process tool."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys

import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools.process_registry import ProcessRegistry, _handle_process


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(source)}"


@pytest.mark.asyncio
async def test_spawn_wait_poll_and_log_preserve_result_contract(tmp_path):
    process_registry = ProcessRegistry()

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        session = await process_registry.spawn_local(
            _python_command("print('first'); print('second')"),
            cwd=str(tmp_path),
            task_id="turn-1",
        )
        result = await process_registry.wait(session.id, timeout=2)

    assert result["status"] == "exited"
    assert result["exit_code"] == 0
    assert result["output"].splitlines() == ["first", "second"]
    assert session._monitor_task is not None
    assert session._monitor_task.done()
    assert process_registry.poll(session.id)["status"] == "exited"
    assert process_registry.read_log(session.id, limit=1)["output"] == "second"


@pytest.mark.asyncio
async def test_submit_writes_stdin_and_close_sends_eof(tmp_path):
    process_registry = ProcessRegistry()

    async with no_task_leaks(action=LeakAction.RAISE):
        session = await process_registry.spawn_local(
            _python_command(
                "import sys; print(sys.stdin.readline().strip()); print(sys.stdin.read())"
            ),
            cwd=str(tmp_path),
        )
        assert await process_registry.submit_stdin(session.id, "hello") == {
            "status": "ok",
            "bytes_written": 6,
        }
        assert await process_registry.close_stdin(session.id) == {
            "status": "ok",
            "message": "stdin closed",
        }
        result = await process_registry.wait(session.id, timeout=2)

    assert result["status"] == "exited"
    assert result["output"].splitlines()[0] == "hello"
    assert session._monitor_task is not None
    assert session._monitor_task.done()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="stdlib PTY transport is POSIX-only")
async def test_pty_background_process_accepts_interactive_input(tmp_path):
    process_registry = ProcessRegistry()

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        session = await process_registry.spawn_local(
            _python_command("print(input())"),
            cwd=str(tmp_path),
            use_pty=True,
        )
        submitted = await process_registry.submit_stdin(session.id, "interactive")
        result = await process_registry.wait(session.id, timeout=2)

    assert submitted["status"] == "ok"
    assert result["status"] == "exited"
    assert "interactive" in result["output"]


@pytest.mark.asyncio
async def test_wait_cancellation_propagates_without_killing_process(tmp_path):
    process_registry = ProcessRegistry()

    async with no_task_leaks(action=LeakAction.RAISE):
        session = await process_registry.spawn_local(
            _python_command("import time; time.sleep(30)"),
            cwd=str(tmp_path),
        )
        waiter = asyncio.create_task(process_registry.wait(session.id, timeout=30))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert process_registry.poll(session.id)["status"] == "running"
        killed = await process_registry.kill_process(session.id)

    assert killed["status"] == "killed"
    assert session.process is not None
    assert session.process.returncode is not None


@pytest.mark.asyncio
async def test_process_handler_keeps_upstream_actions_and_json_shape(tmp_path):
    process_registry = ProcessRegistry()
    session = await process_registry.spawn_local(
        _python_command("print('handled')"),
        cwd=str(tmp_path),
        task_id="handler-turn",
    )

    from tools import process_registry as process_module

    original = process_module.process_registry
    process_module.process_registry = process_registry
    try:
        waited = json.loads(
            await _handle_process(
                {"action": "wait", "session_id": session.id, "timeout": 2},
                task_id="handler-turn",
            )
        )
        listed = json.loads(
            await _handle_process({"action": "list"}, task_id="handler-turn")
        )
    finally:
        await process_registry.kill_all()
        process_module.process_registry = original

    assert waited["status"] == "exited"
    assert waited["output"].strip() == "handled"
    assert listed["processes"][0]["session_id"] == session.id
