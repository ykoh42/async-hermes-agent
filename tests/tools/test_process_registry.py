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

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    _handle_process,
    format_uptime_short,
)


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(source)}"


def test_process_session_upstream_public_defaults_and_uptime_format():
    session = ProcessSession("proc_contract", "echo contract")

    assert session.started_at == 0.0
    assert session.cwd is None
    assert session.completion_reason == "exited"
    assert session.pid_scope == "host"
    assert session.watch_patterns == []
    assert session.notify_on_complete is False
    assert format_uptime_short(59) == "59s"
    assert format_uptime_short(61) == "1m 1s"
    assert format_uptime_short(3660) == "1h 1m"


@pytest.fixture(autouse=True)
def isolated_process_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr("tools.process_registry.CHECKPOINT_PATH", checkpoint)
    return checkpoint


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
    assert (await process_registry.poll(session.id))["status"] == "exited"
    assert (await process_registry.read_log(session.id, limit=1))["output"] == "second"


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

        assert (await process_registry.poll(session.id))["status"] == "running"
        killed = await process_registry.kill_process(session.id)

    assert killed["status"] == "killed"
    assert session.process is not None
    assert session.process.returncode is not None


@pytest.mark.asyncio
async def test_spawn_cancellation_during_checkpoint_reaps_child(tmp_path, monkeypatch):
    process_registry = ProcessRegistry()
    checkpoint_started = asyncio.Event()
    calls = 0

    async def blocking_first_checkpoint():
        nonlocal calls
        calls += 1
        if calls == 1:
            checkpoint_started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(
        process_registry,
        "_write_checkpoint",
        blocking_first_checkpoint,
    )

    async with no_task_leaks(action=LeakAction.RAISE):
        spawning = asyncio.create_task(
            process_registry.spawn_local(
                _python_command("import time; time.sleep(30)"),
                cwd=str(tmp_path),
            )
        )
        await checkpoint_started.wait()
        session = next(iter(process_registry._running.values()))
        spawning.cancel()
        with pytest.raises(asyncio.CancelledError):
            await spawning

    assert session.process is not None
    assert session.process.returncode is not None
    assert session.id not in process_registry._running
    assert session._monitor_task is not None
    assert session._monitor_task.done()


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


@pytest.mark.asyncio
async def test_notify_on_complete_queues_exactly_one_upstream_event():
    process_registry = ProcessRegistry()
    session = ProcessSession(
        "proc_notify",
        "printf done",
        started_at=1.0,
        notify_on_complete=True,
        output_buffer="done\n",
        exited=True,
        exit_code=0,
    )
    process_registry._running[session.id] = session

    await process_registry._move_to_finished(session)
    await process_registry._move_to_finished(session)
    notifications = process_registry.drain_notifications()

    assert len(notifications) == 1
    event, formatted = notifications[0]
    assert event["type"] == "completion"
    assert event["session_id"] == session.id
    assert "completed normally" in formatted
    assert "done" in formatted


@pytest.mark.asyncio
async def test_watch_pattern_queues_match_and_unblocks_wait_state():
    process_registry = ProcessRegistry()
    session = ProcessSession(
        "proc_watch",
        "serve",
        watch_patterns=["READY"],
    )
    process_registry._running[session.id] = session

    process_registry._append_output(session, "starting\nREADY on port 8000\n")
    notifications = process_registry.drain_notifications()

    assert session._watch_hits == 1
    assert len(notifications) == 1
    event, formatted = notifications[0]
    assert event["type"] == "watch_match"
    assert event["pattern"] == "READY"
    assert "READY on port 8000" in formatted
    assert await process_registry.is_session_waiting(session.id) is False


@pytest.mark.asyncio
async def test_notification_drain_preserves_foreign_session_event():
    process_registry = ProcessRegistry()
    process_registry.completion_queue.put_nowait(
        {
            "type": "watch_match",
            "session_id": "proc_foreign",
            "session_key": "foreign",
            "command": "serve",
            "pattern": "READY",
            "output": "READY",
        }
    )

    assert process_registry.drain_notifications(session_key="local") == []
    claimed = process_registry.drain_notifications(session_key="foreign")
    assert len(claimed) == 1


@pytest.mark.asyncio
async def test_wait_consumption_suppresses_duplicate_completion_notification():
    process_registry = ProcessRegistry()
    session = ProcessSession(
        "proc_consumed",
        "printf done",
        notify_on_complete=True,
        output_buffer="done",
        exited=True,
        exit_code=0,
    )
    process_registry._running[session.id] = session
    await process_registry._move_to_finished(session)

    waited = await process_registry.wait(session.id, timeout=1)

    assert waited["status"] == "exited"
    assert process_registry.drain_notifications() == []


@pytest.mark.asyncio
async def test_poll_observation_can_be_delivered_by_autonomous_consumer():
    process_registry = ProcessRegistry()
    session = ProcessSession(
        "proc_polled",
        "printf done",
        notify_on_complete=True,
        exited=True,
        exit_code=0,
    )
    process_registry._running[session.id] = session
    await process_registry._move_to_finished(session)

    assert (await process_registry.poll(session.id))["status"] == "exited"
    autonomous = process_registry.drain_notifications(skip_poll_observed=False)

    assert len(autonomous) == 1


def test_watch_rate_limit_reports_suppressed_matches_on_next_delivery():
    process_registry = ProcessRegistry()
    session = ProcessSession(
        "proc_rate",
        "serve",
        watch_patterns=["READY"],
    )
    process_registry._running[session.id] = session

    process_registry._append_output(session, "READY first\n")
    assert len(process_registry.drain_notifications()) == 1
    process_registry._append_output(session, "READY suppressed\n")
    assert process_registry.drain_notifications() == []
    session._watch_cooldown_until = 0
    process_registry._append_output(session, "READY next\n")
    event, formatted = process_registry.drain_notifications()[0]

    assert event["suppressed"] == 1
    assert "1 earlier matches were suppressed" in formatted


def test_watch_match_output_is_bounded_to_upstream_limit():
    process_registry = ProcessRegistry()
    session = ProcessSession(
        "proc_bounded_watch",
        "serve",
        watch_patterns=["READY"],
    )
    process_registry._running[session.id] = session

    process_registry._append_output(session, f"READY {'x' * 3000}\n")
    event, _formatted = process_registry.drain_notifications()[0]

    assert len(event["output"]) <= 2015
    assert event["output"].endswith("...(truncated)")


@pytest.mark.asyncio
async def test_running_process_checkpoint_recovers_and_kills_detached_pid(
    tmp_path,
    isolated_process_checkpoint,
):
    original = ProcessRegistry()
    session = await original.spawn_local(
        _python_command("import time; time.sleep(30)"),
        cwd=str(tmp_path),
        task_id="checkpoint-task",
        session_key="checkpoint-session",
    )

    checkpoint_data = json.loads(isolated_process_checkpoint.read_text())
    assert checkpoint_data == [
        {
            "session_id": session.id,
            "command": session.command,
            "pid": session.pid,
            "pid_scope": "host",
            "host_start_time": session.host_start_time,
            "cwd": str(tmp_path),
            "started_at": session.started_at,
            "task_id": "checkpoint-task",
            "session_key": "checkpoint-session",
            "watcher_platform": "",
            "watcher_chat_id": "",
            "watcher_user_id": "",
            "watcher_user_name": "",
            "watcher_thread_id": "",
            "watcher_message_id": "",
            "watcher_interval": 0,
            "notify_on_complete": False,
            "watch_patterns": [],
        }
    ]

    restarted = ProcessRegistry()
    try:
        assert await restarted.recover_from_checkpoint() == 1
        recovered = await restarted.get(session.id)
        assert recovered is not None
        assert recovered.detached is True
        assert (await restarted.poll(session.id))["status"] == "running"

        killed = await restarted.kill_process(session.id)
        assert killed["status"] == "killed"
        assert (await restarted.poll(session.id))["status"] == "exited"
        assert json.loads(isolated_process_checkpoint.read_text()) == []
    finally:
        await original.kill_all()


@pytest.mark.asyncio
async def test_checkpoint_recovery_drops_dead_and_sandbox_pids(
    isolated_process_checkpoint,
):
    isolated_process_checkpoint.write_text(
        json.dumps(
            [
                {
                    "session_id": "proc_dead",
                    "command": "sleep 999",
                    "pid": 999_999_999,
                    "task_id": "t1",
                },
                {
                    "session_id": "proc_sandbox",
                    "command": "sleep 999",
                    "pid": os.getpid(),
                    "pid_scope": "sandbox",
                    "task_id": "t1",
                },
            ]
        )
    )

    restarted = ProcessRegistry()
    assert await restarted.recover_from_checkpoint() == 0
    assert await restarted.get("proc_dead") is None
    assert await restarted.get("proc_sandbox") is None
    assert json.loads(isolated_process_checkpoint.read_text()) == []


@pytest.mark.asyncio
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc start ticks")
async def test_checkpoint_recovery_refuses_recycled_pid(
    isolated_process_checkpoint,
):
    real_start = await ProcessRegistry._safe_host_start_time(os.getpid())
    assert real_start is not None
    isolated_process_checkpoint.write_text(
        json.dumps(
            [
                {
                    "session_id": "proc_recycled",
                    "command": "do-not-adopt",
                    "pid": os.getpid(),
                    "pid_scope": "host",
                    "host_start_time": real_start + 1,
                }
            ]
        )
    )

    restarted = ProcessRegistry()
    assert await restarted.recover_from_checkpoint() == 0
    assert await restarted.get("proc_recycled") is None
