"""Cancellation contracts for the append-only trajectory writer."""

import asyncio
import gc
import json
import threading
import time
import weakref

import pytest

from agent.trajectory import save_trajectory


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_owned_trajectory_write(monkeypatch):
    """Repeated cancellation cannot detach the shielded JSONL write task."""
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    write_completed = asyncio.Event()

    class ControlledFile:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def write(self, _line):
            write_started.set()
            await release_write.wait()
            write_completed.set()

        async def flush(self):
            return None

    monkeypatch.setattr(
        "agent.trajectory.aiofiles.open",
        lambda *_args, **_kwargs: ControlledFile(),
    )

    task = asyncio.create_task(
        save_trajectory(
            [{"from": "human", "value": "cancel me"}],
            "test-model",
            completed=False,
        )
    )
    await write_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
    finally:
        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(write_completed.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_concurrent_writers_append_complete_ordered_jsonl_records(
    monkeypatch,
    tmp_path,
):
    """One process never interleaves two trajectory records for one file."""
    target = tmp_path / "trajectories.jsonl"
    active_writers = 0
    max_active_writers = 0
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    real_open = __import__("agent.trajectory", fromlist=["aiofiles"]).aiofiles.open

    class ControlledFile:
        def __init__(self, handle, ordinal):
            self._handle = handle
            self._file = None
            self._ordinal = ordinal

        async def __aenter__(self):
            nonlocal active_writers, max_active_writers
            self._file = await self._handle.__aenter__()
            active_writers += 1
            max_active_writers = max(max_active_writers, active_writers)
            if self._ordinal == 1:
                first_entered.set()
                await release_first.wait()
            return self

        async def __aexit__(self, *args):
            nonlocal active_writers
            active_writers -= 1
            return await self._handle.__aexit__(*args)

        async def write(self, value):
            return await self._file.write(value)

        async def flush(self):
            return await self._file.flush()

    open_count = 0

    def controlled_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return ControlledFile(real_open(*args, **kwargs), open_count)

    monkeypatch.setattr("agent.trajectory.aiofiles.open", controlled_open)
    first = asyncio.create_task(
        save_trajectory([{"from": "human", "value": "first"}], "m", True, str(target))
    )
    await first_entered.wait()
    second = asyncio.create_task(
        save_trajectory([{"from": "human", "value": "second"}], "m", True, str(target))
    )
    await asyncio.sleep(0)
    assert second.done() is False
    release_first.set()
    await asyncio.gather(first, second)

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["conversations"][0]["value"] for row in rows] == ["first", "second"]
    assert max_active_writers == 1


def test_new_arrival_cannot_overtake_granted_fifo_waiter(tmp_path):
    trajectory = __import__(
        "agent.trajectory",
        fromlist=["_claim_trajectory_file"],
    )
    target = str(tmp_path / "fifo.jsonl")
    path_key, first_owner, first = trajectory._claim_trajectory_file(target)
    _, second_owner, second = trajectory._claim_trajectory_file(target)
    assert first_owner is True
    assert second_owner is False
    assert first.done() is True
    assert second.done() is False

    trajectory._finish_trajectory_file_claim(path_key, first)
    assert second.done() is True
    _, third_owner, third = trajectory._claim_trajectory_file(target)
    assert third_owner is False
    assert third.done() is False

    trajectory._finish_trajectory_file_claim(path_key, second)
    assert third.done() is True
    trajectory._finish_trajectory_file_claim(path_key, third)
    assert not trajectory._TRAJECTORY_FILE_CLAIMS


@pytest.mark.parametrize("_stress_iteration", range(20))
def test_cross_loop_writers_share_one_append_claim(
    monkeypatch,
    tmp_path,
    _stress_iteration,
):
    """Separate library event loops cannot overlap a single JSONL append."""
    target = tmp_path / "cross-loop.jsonl"
    first_entered = threading.Event()
    release_first = threading.Event()
    state_guard = threading.Lock()
    active_writers = 0
    max_active_writers = 0
    open_count = 0
    errors = []
    loop_refs = []
    real_open = __import__("agent.trajectory", fromlist=["aiofiles"]).aiofiles.open

    class ControlledFile:
        def __init__(self, handle, ordinal):
            self._handle = handle
            self._file = None
            self._ordinal = ordinal

        async def __aenter__(self):
            nonlocal active_writers, max_active_writers
            self._file = await self._handle.__aenter__()
            with state_guard:
                active_writers += 1
                max_active_writers = max(max_active_writers, active_writers)
            if self._ordinal == 1:
                first_entered.set()
                while not release_first.is_set():
                    await asyncio.sleep(0.001)
            return self

        async def __aexit__(self, *args):
            nonlocal active_writers
            with state_guard:
                active_writers -= 1
            return await self._handle.__aexit__(*args)

        async def write(self, value):
            return await self._file.write(value)

        async def flush(self):
            return await self._file.flush()

    def controlled_open(*args, **kwargs):
        nonlocal open_count
        with state_guard:
            open_count += 1
            ordinal = open_count
        return ControlledFile(real_open(*args, **kwargs), ordinal)

    monkeypatch.setattr("agent.trajectory.aiofiles.open", controlled_open)

    def writer(value):
        async def write():
            with state_guard:
                loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            await save_trajectory(
                [{"from": "human", "value": value}],
                "m",
                True,
                str(target),
            )

        try:
            asyncio.run(write())
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=writer, args=("first",))
    second = threading.Thread(target=writer, args=("second",))
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    time.sleep(0.02)
    with state_guard:
        assert active_writers == 1
        assert open_count == 1
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["conversations"][0]["value"] for row in rows] == ["first", "second"]
    assert max_active_writers == 1
    gc.collect()
    trajectory = __import__("agent.trajectory", fromlist=["_TRAJECTORY_FILE_CLAIMS"])
    assert not trajectory._TRAJECTORY_FILE_CLAIMS
    assert [reference() for reference in loop_refs] == [None, None]


@pytest.mark.asyncio
async def test_twenty_same_loop_writers_keep_claim_order(tmp_path):
    target = tmp_path / "twenty.jsonl"
    await asyncio.gather(
        *(
            save_trajectory(
                [{"from": "human", "value": str(index)}],
                "stress-model",
                index % 2 == 0,
                str(target),
            )
            for index in range(20)
        )
    )

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["conversations"][0]["value"] for row in rows] == [
        str(index) for index in range(20)
    ]
    assert [row["completed"] for row in rows] == [
        index % 2 == 0 for index in range(20)
    ]
    assert all(set(row) == {"conversations", "timestamp", "model", "completed"} for row in rows)


@pytest.mark.asyncio
async def test_failed_owner_releases_next_waiter(monkeypatch, tmp_path):
    target = tmp_path / "recovery.jsonl"
    real_open = __import__("agent.trajectory", fromlist=["aiofiles"]).aiofiles.open
    failed_owner_entered = asyncio.Event()
    release_failed_owner = asyncio.Event()
    calls = 0

    class FailedOpen:
        async def __aenter__(self):
            failed_owner_entered.set()
            await release_failed_owner.wait()
            raise OSError("injected append failure")

        async def __aexit__(self, *_args):
            return None

    def fail_first_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FailedOpen()
        return real_open(*args, **kwargs)

    monkeypatch.setattr("agent.trajectory.aiofiles.open", fail_first_open)
    first_task = asyncio.create_task(
        save_trajectory(
            [{"from": "human", "value": "failed"}],
            "m",
            False,
            str(target),
        )
    )
    await failed_owner_entered.wait()
    second_task = asyncio.create_task(
        save_trajectory(
            [{"from": "human", "value": "recovered"}],
            "m",
            True,
            str(target),
        )
    )
    trajectory = __import__("agent.trajectory", fromlist=["_TRAJECTORY_FILE_CLAIMS"])
    while sum(map(len, trajectory._TRAJECTORY_FILE_CLAIMS.values())) < 2:
        await asyncio.sleep(0)
    release_failed_owner.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is second is None
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["conversations"][0]["value"] for row in rows] == ["recovered"]
    assert not trajectory._TRAJECTORY_FILE_CLAIMS


@pytest.mark.asyncio
async def test_repeatedly_cancelled_waiter_still_writes_in_fifo_order(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "cancelled-waiter.jsonl"
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    real_open = __import__("agent.trajectory", fromlist=["aiofiles"]).aiofiles.open
    open_count = 0

    class ControlledFile:
        def __init__(self, handle, ordinal):
            self._handle = handle
            self._file = None
            self._ordinal = ordinal

        async def __aenter__(self):
            self._file = await self._handle.__aenter__()
            if self._ordinal == 1:
                first_entered.set()
                await release_first.wait()
            return self

        async def __aexit__(self, *args):
            return await self._handle.__aexit__(*args)

        async def write(self, value):
            return await self._file.write(value)

        async def flush(self):
            return await self._file.flush()

    def controlled_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return ControlledFile(real_open(*args, **kwargs), open_count)

    monkeypatch.setattr("agent.trajectory.aiofiles.open", controlled_open)
    first = asyncio.create_task(
        save_trajectory(
            [{"from": "human", "value": "first"}],
            "m",
            True,
            str(target),
        )
    )
    await first_entered.wait()
    waiter = asyncio.create_task(
        save_trajectory(
            [{"from": "human", "value": "cancelled-waiter"}],
            "m",
            False,
            str(target),
        )
    )
    trajectory = __import__("agent.trajectory", fromlist=["_TRAJECTORY_FILE_CLAIMS"])
    while sum(map(len, trajectory._TRAJECTORY_FILE_CLAIMS.values())) < 2:
        await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()
    release_first.set()

    await first
    with pytest.raises(asyncio.CancelledError):
        await waiter
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["conversations"][0]["value"] for row in rows] == [
        "first",
        "cancelled-waiter",
    ]
    assert not trajectory._TRAJECTORY_FILE_CLAIMS


def test_completed_claims_do_not_retain_closed_event_loops(tmp_path):
    target = tmp_path / "loop-gc.jsonl"
    loop_refs = []

    async def write(value):
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        await save_trajectory(
            [{"from": "human", "value": value}],
            "m",
            True,
            str(target),
        )

    asyncio.run(write("one"))
    asyncio.run(write("two"))
    gc.collect()

    trajectory = __import__("agent.trajectory", fromlist=["_TRAJECTORY_FILE_CLAIMS"])
    assert not trajectory._TRAJECTORY_FILE_CLAIMS
    assert [reference() for reference in loop_refs] == [None, None]
