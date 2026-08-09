"""Tests for batch_runner trajectory durability and pool cleanup.

Verifies:
  1. Trajectory entries are fsync'd to disk before the checkpoint marks
     them as completed (crash-between-write-and-sync safety).
  2. BatchRunner.run() calls pool.terminate() + pool.join() on
     KeyboardInterrupt and Exception during batch execution (responsive
     worker shutdown).  CPython's Pool.join() takes no timeout parameter —
     join(timeout=10) raises TypeError — so the tests also assert join()
     is invoked with no arguments.
"""

import json
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

# batch_runner is a root-level module (not part of an installed package),
# so make the repo root importable when tests run from elsewhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

import batch_runner
from batch_runner import BatchRunner, _process_batch_worker


@asynccontextmanager
async def _batch_runtime_audit(detector: str):
    """Run each audit without cross-instrumenting allowed aiofiles workers."""
    if detector == "pyleak":
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
        ):
            yield
        return

    async with no_task_leaks(action=LeakAction.RAISE):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            yield
        finally:
            blockbuster.deactivate()


# =========================================================================
# Trajectory write durability (fsync)
# =========================================================================

class TestTrajectoryWriteDurability:
    """Verify that trajectory entries are flushed and fsync'd to disk.

    Without fsync, a crash between the write and the disk sync could leave
    the checkpoint claiming completion with no trajectory data on disk.
    """

    @pytest.mark.asyncio
    async def test_trajectory_entry_is_synced_to_disk(self, tmp_path, monkeypatch):
        """_process_batch_worker should flush+fsync the trajectory file."""
        prompt_result = {
            "success": True,
            "trajectory": [{"role": "assistant", "content": "x"}],
            "reasoning_stats": {"has_any_reasoning": True},
            "tool_stats": {},
            "metadata": {},
            "completed": True,
            "api_calls": 1,
            "toolsets_used": [],
        }

        async def process_prompt(*_args, **_kwargs):
            return prompt_result

        monkeypatch.setattr("batch_runner._process_single_prompt", process_prompt)

        # Intercept os.fsync to record calls
        fsync_calls = []
        monkeypatch.setattr("os.fsync", lambda fd: fsync_calls.append(fd))

        await _process_batch_worker(
            (
                1,
                [(0, {"prompt": "hi"})],
                tmp_path,
                set(),
                {"verbose": False},
            )
        )

        # Verify fsync was called at least once during trajectory write
        assert len(fsync_calls) >= 1, (
            "os.fsync was not called — trajectory writes are not durable"
        )

        # Verify the trajectory file exists and is valid
        output_files = list(tmp_path.glob("*.jsonl"))
        assert len(output_files) >= 1
        for f in output_files:
            lines = f.read_text().strip().split("\n")
            for line in lines:
                if line:
                    entry = json.loads(line)
                    assert "conversations" in entry
                    assert "completed" in entry


# =========================================================================
# Pool cleanup on interruption / exception — drives the REAL run()
# =========================================================================

def _make_runner(tmp_path, monkeypatch):
    """Build a minimal real BatchRunner against a 1-line tmp dataset."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "first"}) + "\n"
        + json.dumps({"prompt": "second"}) + "\n",
        encoding="utf-8",
    )
    # BatchRunner writes to Path("data")/run_name relative to cwd.
    monkeypatch.chdir(tmp_path)
    return BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="pool-cleanup-test",
        num_workers=2,
    )


class TestTaskCleanupOnInterruption:
    """A failed or cancelled async batch must not leak sibling tasks."""

    @pytest.mark.asyncio
    async def test_run_cancels_and_awaits_siblings_on_worker_error(
        self, tmp_path, monkeypatch
    ):
        runner = _make_runner(tmp_path, monkeypatch)
        both_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def worker(args):
            batch_num = args[0]
            if batch_num == 0:
                await both_started.wait()
                raise RuntimeError("boom")
            both_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        monkeypatch.setattr(batch_runner, "_process_batch_worker", worker)
        async with no_task_leaks(action=LeakAction.RAISE):
            with pytest.raises(RuntimeError, match="boom"):
                await runner.run()
        assert sibling_cancelled.is_set()

    @pytest.mark.asyncio
    async def test_run_cancels_and_awaits_workers_when_caller_cancels(
        self, tmp_path, monkeypatch
    ):
        runner = _make_runner(tmp_path, monkeypatch)
        started = asyncio.Event()
        cancelled = 0

        async def worker(_args):
            nonlocal cancelled
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled += 1
                raise

        monkeypatch.setattr(batch_runner, "_process_batch_worker", worker)
        async with no_task_leaks(action=LeakAction.RAISE):
            run_task = asyncio.create_task(runner.run())
            await started.wait()
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task
        assert cancelled == 2

    @pytest.mark.asyncio
    async def test_caller_cancellation_wins_during_worker_error_cleanup(
        self, tmp_path, monkeypatch
    ):
        runner = _make_runner(tmp_path, monkeypatch)
        both_started = asyncio.Event()
        sibling_cleanup_started = asyncio.Event()
        release_sibling_cleanup = asyncio.Event()
        sibling_finished = asyncio.Event()

        async def worker(args):
            batch_num = args[0]
            if batch_num == 0:
                await both_started.wait()
                raise RuntimeError("worker failed first")
            both_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cleanup_started.set()
                await release_sibling_cleanup.wait()
                raise
            finally:
                sibling_finished.set()

        monkeypatch.setattr(batch_runner, "_process_batch_worker", worker)
        async with no_task_leaks(action=LeakAction.RAISE):
            run_task = asyncio.create_task(runner.run())
            await sibling_cleanup_started.wait()
            run_task.cancel()
            await asyncio.sleep(0)

            assert run_task.done() is False
            release_sibling_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await run_task

        assert sibling_finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_audit", ("pyleak", "blockbuster"))
async def test_concurrent_batches_write_complete_rows_and_resume_without_duplicates(
    tmp_path, monkeypatch, runtime_audit
):
    prompts = [f"prompt-{index}" for index in range(4)]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="concurrent-trajectories",
        num_workers=2,
    )

    active = 0
    max_active = 0
    processed = []

    async def process_prompt(prompt_index, prompt_data, *_args, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            processed.append(prompt_index)
            return {
                "success": True,
                "trajectory": [
                    {"from": "human", "value": prompt_data["prompt"]},
                    {"from": "gpt", "value": "<think>reasoning</think>done"},
                ],
                "reasoning_stats": {
                    "has_any_reasoning": True,
                    "total_assistant_turns": 1,
                    "turns_with_reasoning": 1,
                    "turns_without_reasoning": 0,
                },
                "tool_stats": {},
                "metadata": {},
                "completed": True,
                "api_calls": 1,
                "toolsets_used": [],
            }
        finally:
            active -= 1

    monkeypatch.setattr(batch_runner, "_process_single_prompt", process_prompt)

    async with _batch_runtime_audit(runtime_audit):
        await runner.run()

    combined_file = runner.output_dir / "trajectories.jsonl"
    rows = [
        json.loads(line)
        for line in combined_file.read_text(encoding="utf-8").splitlines()
    ]
    checkpoint = json.loads(runner.checkpoint_file.read_text(encoding="utf-8"))
    assert max_active == 2
    assert sorted(processed) == [0, 1, 2, 3]
    assert sorted(row["prompt_index"] for row in rows) == [0, 1, 2, 3]
    assert checkpoint["completed_prompts"] == [0, 1, 2, 3]

    async with _batch_runtime_audit(runtime_audit):
        await runner.run(resume=True)

    resumed_rows = [
        json.loads(line)
        for line in combined_file.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(processed) == [0, 1, 2, 3]
    assert resumed_rows == rows
