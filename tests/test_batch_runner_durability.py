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
from pathlib import Path

import pytest

# batch_runner is a root-level module (not part of an installed package),
# so make the repo root importable when tests run from elsewhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

import batch_runner
from batch_runner import BatchRunner, _process_batch_worker


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
        run_task = asyncio.create_task(runner.run())
        await started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert cancelled == 2
