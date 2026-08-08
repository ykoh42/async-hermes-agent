"""Unit tests for the task-based supervisor registry healthcheck."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as supervisor_module


@pytest.fixture
def isolated_registry():
    return supervisor_module._SupervisorRegistry()


@pytest.fixture
def stub_cdp_supervisor(monkeypatch):
    created = []

    class StubSupervisor:
        def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self.dialog_policy = dialog_policy
            self.dialog_timeout_s = dialog_timeout_s
            self._run_task = None
            self.start_called = False
            self.stop_called = False
            created.append(self)

        async def start(self, timeout: float = 15.0) -> None:
            self.start_called = True
            self._run_task = asyncio.create_task(asyncio.Event().wait())

        async def stop(self) -> None:
            self.stop_called = True
            task = self._run_task
            self._run_task = None
            if task is not None:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    monkeypatch.setattr(supervisor_module, "CDPSupervisor", StubSupervisor)
    return created


@pytest.mark.asyncio
async def test_cache_hit_returns_same_instance_when_healthy(
    isolated_registry, stub_cdp_supervisor
):
    first = await isolated_registry.get_or_start(task_id="t1", cdp_url="http://h/1")
    second = await isolated_registry.get_or_start(task_id="t1", cdp_url="http://h/1")
    assert first is second
    assert len(stub_cdp_supervisor) == 1
    await first.stop()


@pytest.mark.asyncio
async def test_missing_task_attr_triggers_recreate(
    isolated_registry, stub_cdp_supervisor
):
    cdp_url = "http://h/4"
    stop_calls = []

    async def stop():
        stop_calls.append(True)

    broken = SimpleNamespace(cdp_url=cdp_url, stop=stop)
    isolated_registry._by_task["t4"] = broken

    fresh = await isolated_registry.get_or_start(task_id="t4", cdp_url=cdp_url)
    assert fresh is not broken
    assert isolated_registry._by_task["t4"] is fresh
    assert stop_calls == [True]
    await fresh.stop()


@pytest.mark.asyncio
async def test_stop_cancels_retained_background_tasks():
    supervisor = supervisor_module.CDPSupervisor(
        task_id="background-task-test", cdp_url="ws://127.0.0.1:1"
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    supervisor._start_background_task(worker(), name="test-browser-background")
    await started.wait()

    await supervisor.stop()

    assert cancelled.is_set()
    assert not supervisor._background_tasks


@pytest.mark.asyncio
async def test_stop_repeated_cancellation_waits_for_websocket_close():
    supervisor = supervisor_module.CDPSupervisor(
        task_id="stop-cancellation-test", cdp_url="ws://127.0.0.1:1"
    )
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_completed = asyncio.Event()

    class ControlledWebSocket:
        async def close(self):
            close_started.set()
            await release_close.wait()
            close_completed.set()

    supervisor._ws = ControlledWebSocket()
    supervisor._active = True
    stopping = asyncio.create_task(supervisor.stop())
    await close_started.wait()
    stopping.cancel()
    await asyncio.sleep(0)
    stopping.cancel()
    await asyncio.sleep(0)

    try:
        assert stopping.done() is False
    finally:
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        await asyncio.wait_for(close_completed.wait(), timeout=1.0)

    assert supervisor._active is False
