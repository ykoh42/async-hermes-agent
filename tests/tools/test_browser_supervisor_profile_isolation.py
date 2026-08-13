"""Loop/profile ownership tests for the persistent browser supervisor."""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import browser_supervisor


@pytest.fixture
def stub_supervisors(monkeypatch):
    created = []

    class StubSupervisor:
        def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self.dialog_policy = dialog_policy
            self.dialog_timeout_s = dialog_timeout_s
            self.owner_loop = None
            self._run_task = None
            self.stopped = False
            created.append(self)

        async def start(self, timeout=15.0):
            self.owner_loop = asyncio.get_running_loop()
            self._run_task = asyncio.create_task(asyncio.Event().wait())

        async def stop(self):
            assert asyncio.get_running_loop() is self.owner_loop
            self.stopped = True
            task = self._run_task
            self._run_task = None
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(browser_supervisor, "CDPSupervisor", StubSupervisor)
    return created


def test_same_profile_task_id_is_isolated_across_simultaneous_event_loops(
    tmp_path: Path,
    stub_supervisors,
):
    registry = browser_supervisor._SupervisorRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()
    coordination_lock = threading.Lock()
    ready = 0
    allow_second_stop = False
    results = {}
    failures = []

    async def wait_until(predicate):
        while not predicate():
            await asyncio.sleep(0)

    def worker(label: str) -> None:
        async def run() -> None:
            nonlocal ready, allow_second_stop
            token = set_hermes_home_override(profile)
            try:
                supervisor = await registry.get_or_start(
                    task_id="same-task",
                    cdp_url="ws://same-endpoint",
                )
                with coordination_lock:
                    results[label] = supervisor
                    ready += 1
                await wait_until(lambda: ready == 2)
                assert registry.get("same-task") is supervisor
                if label == "first":
                    await registry.stop("same-task")
                    with coordination_lock:
                        results["first_saw_second_stopped"] = results[
                            "second"
                        ].stopped
                        allow_second_stop = True
                else:
                    await wait_until(lambda: allow_second_stop)
                    assert not getattr(supervisor, "stopped")
                    await registry.stop("same-task")
            finally:
                reset_hermes_home_override(token)

        try:
            asyncio.run(run())
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(label,))
        for label in ("first", "second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert results["first"] is not results["second"]
    assert results["first"].task_id == "same-task"
    assert results["second"].task_id == "same-task"
    assert results["first_saw_second_stopped"] is False
    assert results["first"].stopped is True
    assert results["second"].stopped is True


def test_process_final_stop_all_dispatches_shutdown_to_each_owner_loop(
    tmp_path: Path,
    stub_supervisors,
):
    registry = browser_supervisor._SupervisorRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()
    coordination_lock = threading.Lock()
    ready = 0
    emergency_done = False
    results = {}
    failures = []

    async def wait_until(predicate):
        while not predicate():
            await asyncio.sleep(0)

    def worker(label: str) -> None:
        async def run() -> None:
            nonlocal ready, emergency_done
            token = set_hermes_home_override(profile)
            try:
                supervisor = await registry.get_or_start(
                    task_id="same-task",
                    cdp_url="ws://same-endpoint",
                )
                with coordination_lock:
                    results[label] = supervisor
                    ready += 1
                await wait_until(lambda: ready == 2)
                if label == "first":
                    await registry.stop_all()
                    with coordination_lock:
                        emergency_done = True
                else:
                    await wait_until(lambda: emergency_done)
                    assert getattr(supervisor, "stopped")
            finally:
                reset_hermes_home_override(token)

        try:
            asyncio.run(run())
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(label,))
        for label in ("first", "second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert results["first"] is not results["second"]
    assert results["first"].stopped is True
    assert results["second"].stopped is True


@pytest.mark.asyncio
async def test_same_loop_task_id_is_isolated_by_canonical_profile(
    tmp_path: Path,
    stub_supervisors,
):
    registry = browser_supervisor._SupervisorRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)
    other = tmp_path / "other"
    other.mkdir()

    async def start(home: Path):
        token = set_hermes_home_override(home)
        try:
            return await registry.get_or_start(
                task_id="same-task",
                cdp_url="ws://same-endpoint",
            )
        finally:
            reset_hermes_home_override(token)

    first = await start(profile)
    canonical_alias = await start(alias)
    second = await start(other)

    assert canonical_alias is first
    assert second is not first

    token = set_hermes_home_override(other)
    try:
        await registry.stop("same-task")
    finally:
        reset_hermes_home_override(token)
    assert second.stopped is True
    assert first.stopped is False

    token = set_hermes_home_override(profile)
    try:
        await registry.stop("same-task")
    finally:
        reset_hermes_home_override(token)
    assert first.stopped is True


@pytest.mark.asyncio
async def test_stop_all_finishes_owned_stops_before_rethrowing_repeated_cancel(
    monkeypatch,
    tmp_path: Path,
):
    registry = browser_supervisor._SupervisorRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()
    release = asyncio.Event()
    started = set()
    completed = set()

    class ControlledSupervisor:
        def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self._run_task = None

        async def start(self, timeout=15.0):
            self._run_task = asyncio.create_task(asyncio.Event().wait())

        async def stop(self):
            started.add(self.task_id)
            await release.wait()
            task = self._run_task
            self._run_task = None
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            completed.add(self.task_id)

    monkeypatch.setattr(
        browser_supervisor,
        "CDPSupervisor",
        ControlledSupervisor,
    )
    token = set_hermes_home_override(profile)
    try:
        for task_id in ("one", "two"):
            await registry.get_or_start(task_id=task_id, cdp_url=f"ws://{task_id}")
        cleanup = asyncio.create_task(registry.stop_all())
        while started != {"one", "two"}:
            await asyncio.sleep(0)
        cleanup.cancel()
        await asyncio.sleep(0)
        cleanup.cancel()
        await asyncio.sleep(0)
        assert not cleanup.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert completed == {"one", "two"}
        assert registry.get("one") is None
        assert registry.get("two") is None
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_cancelled_concurrent_replacement_rolls_back_new_supervisor(
    monkeypatch,
    tmp_path: Path,
):
    registry = browser_supervisor._SupervisorRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()
    start_release = asyncio.Event()
    stop_release = asyncio.Event()
    start_count = 0
    stop_started = set()
    created = []

    class ControlledSupervisor:
        def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self._run_task = None
            self.stopped = False
            created.append(self)

        async def start(self, timeout=15.0):
            nonlocal start_count
            start_count += 1
            await start_release.wait()
            self._run_task = asyncio.create_task(asyncio.Event().wait())

        async def stop(self):
            stop_started.add(self.cdp_url)
            await stop_release.wait()
            self.stopped = True
            task = self._run_task
            self._run_task = None
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(
        browser_supervisor,
        "CDPSupervisor",
        ControlledSupervisor,
    )
    token = set_hermes_home_override(profile)
    try:
        calls = [
            asyncio.create_task(
                registry.get_or_start(task_id="same", cdp_url=f"ws://{index}")
            )
            for index in range(2)
        ]
        while start_count != 2:
            await asyncio.sleep(0)
        start_release.set()
        while not stop_started:
            await asyncio.sleep(0)
        pending = [call for call in calls if not call.done()]
        assert len(pending) == 1
        replacement = pending[0]
        replacement.cancel()
        await asyncio.sleep(0)
        replacement.cancel()
        assert not replacement.done()
        stop_release.set()
        results = await asyncio.gather(*calls, return_exceptions=True)

        assert sum(isinstance(result, asyncio.CancelledError) for result in results) == 1
        assert all(supervisor.stopped for supervisor in created)
        assert registry.get("same") is None
    finally:
        reset_hermes_home_override(token)


def test_closed_loop_registry_values_are_pruned_and_do_not_retain_loop(
    monkeypatch,
    tmp_path: Path,
):
    registry = browser_supervisor._SupervisorRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()

    class ClosedLoopSupervisor:
        def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self._run_task = None

        async def start(self, timeout=15.0):
            self._run_task = asyncio.create_task(asyncio.Event().wait())

        async def stop(self):
            raise AssertionError("closed-loop supervisors must not be awaited")

    monkeypatch.setattr(
        browser_supervisor,
        "CDPSupervisor",
        ClosedLoopSupervisor,
    )
    loop = asyncio.new_event_loop()

    async def create_and_quiesce() -> None:
        token = set_hermes_home_override(profile)
        try:
            supervisor = await registry.get_or_start(
                task_id="abandoned",
                cdp_url="ws://closed-loop",
            )
            run_task = supervisor._run_task
            assert run_task is not None
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        finally:
            reset_hermes_home_override(token)

    loop.run_until_complete(create_and_quiesce())
    loop_ref = weakref.ref(loop)
    loop.close()

    registry._prune_closed_loops()
    del loop
    gc.collect()

    assert not registry._loop_profile_states
    assert loop_ref() is None
