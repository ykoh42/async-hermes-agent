"""Tests for batch_runner checkpoint behavior — incremental writes, resume, atomicity."""

import asyncio
import json
from pathlib import Path

import pytest

# batch_runner uses relative imports, ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_runner import BatchRunner, _append_jsonl_line, _process_batch_worker


@pytest.fixture
def runner(tmp_path):
    """Create a BatchRunner with all paths pointing at tmp_path."""
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("")
    output_file = tmp_path / "output.jsonl"
    checkpoint_file = tmp_path / "checkpoint.json"
    r = BatchRunner.__new__(BatchRunner)
    r.run_name = "test_run"
    r.checkpoint_file = checkpoint_file
    r.output_file = output_file
    r.prompts_file = prompts_file
    return r


class TestSaveCheckpoint:
    """Verify _save_checkpoint writes valid, atomic JSON."""

    @pytest.mark.asyncio
    async def test_writes_valid_json(self, runner):
        data = {"run_name": "test", "completed_prompts": [1, 2, 3], "batch_stats": {}}
        await runner._save_checkpoint(data)

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["run_name"] == "test"
        assert result["completed_prompts"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_adds_last_updated(self, runner):
        data = {"run_name": "test", "completed_prompts": []}
        await runner._save_checkpoint(data)

        result = json.loads(runner.checkpoint_file.read_text())
        assert "last_updated" in result
        assert result["last_updated"] is not None

    @pytest.mark.asyncio
    async def test_overwrites_previous_checkpoint(self, runner):
        await runner._save_checkpoint({"run_name": "test", "completed_prompts": [1]})
        await runner._save_checkpoint({"run_name": "test", "completed_prompts": [1, 2, 3]})

        result = json.loads(runner.checkpoint_file.read_text())
        assert result["completed_prompts"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        runner_deep = BatchRunner.__new__(BatchRunner)
        runner_deep.checkpoint_file = tmp_path / "deep" / "nested" / "checkpoint.json"

        data = {"run_name": "test", "completed_prompts": []}
        await runner_deep._save_checkpoint(data)

        assert runner_deep.checkpoint_file.exists()

    @pytest.mark.asyncio
    async def test_no_temp_files_left(self, runner):
        await runner._save_checkpoint({"run_name": "test", "completed_prompts": []})

        tmp_files = [f for f in runner.checkpoint_file.parent.iterdir()
                     if ".tmp" in f.name]
        assert len(tmp_files) == 0


@pytest.mark.asyncio
async def test_jsonl_append_completes_current_row_before_cancellation(tmp_path):
    """A cancelled batch worker must leave either no row or a whole JSON row."""
    shard = tmp_path / "batch_0.jsonl"
    task = asyncio.create_task(_append_jsonl_line(shard, {"prompt_index": 7}))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    if shard.exists():
        rows = shard.read_text(encoding="utf-8").splitlines()
        assert rows == ['{"prompt_index": 7}']
        assert json.loads(rows[0]) == {"prompt_index": 7}


@pytest.mark.asyncio
async def test_jsonl_append_repeated_cancellation_waits_for_owned_write(
    tmp_path,
    monkeypatch,
):
    """Repeated cancellation cannot detach the shielded row-write task."""
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

        def fileno(self):
            return 1

    def fake_wrap(_function):
        async def wrapped(*_args, **_kwargs):
            return None

        return wrapped

    monkeypatch.setattr(
        "batch_runner.aiofiles.open",
        lambda *_args, **_kwargs: ControlledFile(),
    )
    monkeypatch.setattr("batch_runner.aiofiles.os.wrap", fake_wrap)

    task = asyncio.create_task(
        _append_jsonl_line(tmp_path / "batch_0.jsonl", {"prompt_index": 8})
    )
    await write_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert write_completed.is_set()


class TestLoadCheckpoint:
    """Verify _load_checkpoint reads existing data or returns defaults."""


    @pytest.mark.asyncio
    async def test_loads_existing_checkpoint(self, runner):
        data = {"run_name": "test_run", "completed_prompts": [5, 10, 15],
                "batch_stats": {"0": {"processed": 3}}}
        runner.checkpoint_file.write_text(json.dumps(data))

        result = await runner._load_checkpoint()
        assert result["completed_prompts"] == [5, 10, 15]
        assert result["batch_stats"]["0"]["processed"] == 3

    @pytest.mark.asyncio
    async def test_handles_corrupt_json(self, runner):
        runner.checkpoint_file.write_text("{broken json!!")

        result = await runner._load_checkpoint()
        # Should return empty/default, not crash
        assert isinstance(result, dict)


class TestResumePreservesProgress:
    """Verify that initializing a run with resume=True loads prior checkpoint."""

    @pytest.mark.asyncio
    async def test_completed_prompts_loaded_from_checkpoint(self, runner):
        # Simulate a prior run that completed prompts 0-4
        prior = {
            "run_name": "test_run",
            "completed_prompts": [0, 1, 2, 3, 4],
            "batch_stats": {"0": {"processed": 5}},
            "last_updated": "2026-01-01T00:00:00",
        }
        runner.checkpoint_file.write_text(json.dumps(prior))

        # Load checkpoint like run() does
        checkpoint_data = await runner._load_checkpoint()
        if checkpoint_data.get("run_name") != runner.run_name:
            checkpoint_data = {
                "run_name": runner.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None,
            }

        completed_set = set(checkpoint_data.get("completed_prompts", []))
        assert completed_set == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_different_run_name_starts_fresh(self, runner):
        prior = {
            "run_name": "different_run",
            "completed_prompts": [0, 1, 2],
            "batch_stats": {},
        }
        runner.checkpoint_file.write_text(json.dumps(prior))

        checkpoint_data = await runner._load_checkpoint()
        if checkpoint_data.get("run_name") != runner.run_name:
            checkpoint_data = {
                "run_name": runner.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None,
            }

        assert checkpoint_data["completed_prompts"] == []
        assert checkpoint_data["run_name"] == "test_run"

class TestBatchWorkerResumeBehavior:
    @pytest.mark.asyncio
    async def test_discarded_no_reasoning_prompts_are_marked_completed(self, tmp_path, monkeypatch):
        batch_file = tmp_path / "batch_1.jsonl"
        prompt_result = {
            "success": True,
            "trajectory": [{"role": "assistant", "content": "x"}],
            "reasoning_stats": {"has_any_reasoning": False},
            "tool_stats": {},
            "metadata": {},
            "completed": True,
            "api_calls": 1,
            "toolsets_used": [],
        }

        async def process_single_prompt(*_args, **_kwargs):
            return prompt_result

        monkeypatch.setattr("batch_runner._process_single_prompt", process_single_prompt)

        result = await _process_batch_worker((
            1,
            [(0, {"prompt": "hi"})],
            tmp_path,
            set(),
            {"verbose": False},
        ))

        assert result["discarded_no_reasoning"] == 1
        assert result["completed_prompts"] == [0]
        assert not batch_file.exists() or batch_file.read_text() == ""


class TestFinalCheckpointNoDuplicates:
    """Regression: the final checkpoint must not contain duplicate prompt
    indices.

    Before PR #15161, `run()` populated `completed_prompts_set` incrementally
    as each batch completed, then at the end built `all_completed_prompts =
    list(completed_prompts_set)` AND extended it again with every batch's
    `completed_prompts` — double-counting every index.
    """

    def _simulate_final_aggregation_fixed(self, batch_results):
        """Mirror the fixed code path in batch_runner.run()."""
        completed_prompts_set = set()
        for result in batch_results:
            completed_prompts_set.update(result.get("completed_prompts", []))
        # This is what the fixed code now writes to the checkpoint:
        return sorted(completed_prompts_set)

    def test_no_duplicates_in_final_list(self):
        batch_results = [
            {"completed_prompts": [0, 1, 2]},
            {"completed_prompts": [3, 4]},
            {"completed_prompts": [5]},
        ]
        final = self._simulate_final_aggregation_fixed(batch_results)
        assert final == [0, 1, 2, 3, 4, 5]
        assert len(final) == len(set(final))  # no duplicates

    @pytest.mark.asyncio
    async def test_persisted_checkpoint_has_unique_prompts(self, runner):
        """Write what run()'s fixed aggregation produces to disk; the file
        must load back with no duplicate indices."""
        batch_results = [
            {"completed_prompts": [0, 1]},
            {"completed_prompts": [2, 3]},
        ]
        final = self._simulate_final_aggregation_fixed(batch_results)
        await runner._save_checkpoint({
            "run_name": runner.run_name,
            "completed_prompts": final,
            "batch_stats": {},
        })
        loaded = json.loads(runner.checkpoint_file.read_text())
        cp = loaded["completed_prompts"]
        assert cp == sorted(set(cp))
        assert len(cp) == 4

    def test_old_buggy_pattern_would_have_duplicates(self):
        """Document the bug this PR fixes: the old code shape produced
        duplicates.  Kept as a sanity anchor so a future refactor that
        re-introduces the pattern is immediately visible."""
        completed_prompts_set = set()
        results = []
        for batch in ({"completed_prompts": [0, 1, 2]},
                      {"completed_prompts": [3, 4]}):
            completed_prompts_set.update(batch["completed_prompts"])
            results.append(batch)
        # Buggy aggregation (pre-fix):
        buggy = list(completed_prompts_set)
        for br in results:
            buggy.extend(br.get("completed_prompts", []))
        # Every index appears twice
        assert len(buggy) == 2 * len(set(buggy))
