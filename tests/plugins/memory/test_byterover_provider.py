"""Tests for the native-async ByteRover memory provider."""

import asyncio

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from plugins.memory.byterover import ByteRoverMemoryProvider, _run_brv


@pytest.mark.asyncio
async def test_availability_resolves_binary_without_blocking(tmp_path, monkeypatch):
    executable = tmp_path / "brv"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("plugins.memory.byterover._cached_brv_path", None)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            available = await ByteRoverMemoryProvider().is_available()
        finally:
            blockbuster.deactivate()

    assert available is True


@pytest.mark.asyncio
async def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    await provider.initialize("session-1")

    async def run_brv(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("plugins.memory.byterover._run_brv", run_brv)

    await provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []


@pytest.mark.asyncio
async def test_prefetch_awaits_native_subprocess_result(monkeypatch):
    provider = ByteRoverMemoryProvider()
    provider._cwd = "/tmp"

    async def run_brv(*args, **kwargs):
        return {
            "success": True,
            "output": "A sufficiently long remembered project decision.",
        }

    monkeypatch.setattr("plugins.memory.byterover._run_brv", run_brv)

    result = await provider.prefetch("what architecture did we choose?")

    assert result == (
        "## ByteRover Context\n"
        "A sufficiently long remembered project decision."
    )


@pytest.mark.asyncio
async def test_run_brv_timeout_keeps_event_loop_responsive(tmp_path, monkeypatch):
    executable = tmp_path / "brv"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "time.sleep(1)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        "plugins.memory.byterover._cached_brv_path",
        str(executable),
    )
    heartbeat = 0

    async def beat():
        nonlocal heartbeat
        while True:
            heartbeat += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(beat())
    try:
        result = await _run_brv(["status"], timeout=0.01, cwd=str(tmp_path))
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert result == {"success": False, "error": "brv timed out after 0.01s"}
    assert heartbeat > 1


@pytest.mark.asyncio
async def test_run_brv_repeated_cancellation_drains_process(tmp_path, monkeypatch):
    communicate_started = asyncio.Event()
    release_communicate = asyncio.Event()
    communicate_completed = asyncio.Event()

    class BlockingProcess:
        returncode = None
        killed = False

        async def communicate(self):
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            return b"", b""

        async def wait(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr("plugins.memory.byterover._cached_brv_path", "brv")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(_run_brv(["status"], cwd=str(tmp_path)))
    await communicate_started.wait()
    task.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_communicate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert communicate_completed.is_set()
