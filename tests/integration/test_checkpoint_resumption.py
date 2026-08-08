"""Async integration coverage for durable batch checkpoints and resume."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles
import pytest

import batch_runner
from batch_runner import BatchRunner


pytestmark = pytest.mark.integration


async def _write_dataset(path: Path, prompts: list[str]) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8") as output:
        for prompt in prompts:
            await output.write(json.dumps({"prompt": prompt}) + "\n")


def _successful_result(
    prompt_index: int,
    prompt_data: dict[str, Any],
    batch_num: int,
    model: str,
) -> dict[str, Any]:
    prompt = prompt_data["prompt"]
    return {
        "success": True,
        "prompt_index": prompt_index,
        "trajectory": [
            {"from": "system", "value": "batch integration"},
            {"from": "human", "value": prompt},
            {
                "from": "gpt",
                "value": "<REASONING_SCRATCHPAD>checked</REASONING_SCRATCHPAD>\ndone",
            },
        ],
        "tool_stats": {},
        "reasoning_stats": {
            "total_assistant_turns": 1,
            "turns_with_reasoning": 1,
            "turns_without_reasoning": 0,
            "has_any_reasoning": True,
        },
        "completed": True,
        "partial": False,
        "api_calls": 1,
        "toolsets_used": [],
        "metadata": {"batch_num": batch_num, "model": model},
    }


@pytest.mark.asyncio
async def test_checkpoints_are_persisted_after_each_completed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    prompts = [f"checkpoint prompt {index}" for index in range(6)]
    dataset = tmp_path / "dataset.jsonl"
    await _write_dataset(dataset, prompts)

    async def process_prompt(
        prompt_index: int,
        prompt_data: dict[str, Any],
        batch_num: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        # Make batch completion order deterministic but non-sequential.
        await asyncio.sleep(0.01 * (3 - min(batch_num, 2)))
        return _successful_result(
            prompt_index, prompt_data, batch_num, config["model"]
        )

    monkeypatch.setattr(batch_runner, "_process_single_prompt", process_prompt)
    runner = BatchRunner(
        dataset_file=str(dataset),
        batch_size=2,
        run_name="incremental",
        model="integration-model",
        num_workers=3,
        verbose=False,
    )

    snapshots: list[tuple[int, ...]] = []
    save_checkpoint = runner._save_checkpoint

    async def recording_save(checkpoint: dict[str, Any]) -> None:
        await save_checkpoint(checkpoint)
        snapshots.append(tuple(checkpoint["completed_prompts"]))

    monkeypatch.setattr(runner, "_save_checkpoint", recording_save)
    await runner.run(resume=False)

    # Three per-batch writes plus the final write must all execute.  The first
    # durable checkpoint is necessarily partial, proving this is not an
    # end-of-run-only assertion.
    assert len(snapshots) == 4
    assert 0 < len(snapshots[0]) < len(prompts)
    assert snapshots[-1] == tuple(range(len(prompts)))
    assert all(set(before) <= set(after) for before, after in zip(snapshots, snapshots[1:]))

    checkpoint = json.loads((runner.checkpoint_file).read_text(encoding="utf-8"))
    assert checkpoint["completed_prompts"] == list(range(len(prompts)))


@pytest.mark.asyncio
async def test_resume_uses_durable_trajectory_content_and_keeps_original_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    prompts = [f"resume prompt {index}" for index in range(5)]
    dataset = tmp_path / "dataset.jsonl"
    await _write_dataset(dataset, prompts)

    processed: list[int] = []

    async def process_prompt(
        prompt_index: int,
        prompt_data: dict[str, Any],
        batch_num: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        processed.append(prompt_index)
        await asyncio.sleep(0)
        return _successful_result(
            prompt_index, prompt_data, batch_num, config["model"]
        )

    monkeypatch.setattr(batch_runner, "_process_single_prompt", process_prompt)

    first = BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="resume",
        model="integration-model",
        num_workers=2,
        max_samples=2,
        verbose=False,
    )
    await first.run(resume=False)
    assert processed == [0, 1]

    processed.clear()
    resumed = BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="resume",
        model="integration-model",
        num_workers=2,
        verbose=False,
    )
    await resumed.run(resume=True)

    assert sorted(processed) == [2, 3, 4]
    checkpoint = json.loads(resumed.checkpoint_file.read_text(encoding="utf-8"))
    assert checkpoint["completed_prompts"] == [0, 1, 2, 3, 4]

    rows = [
        json.loads(line)
        for line in (resumed.output_dir / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    human_prompts = {
        message["value"]
        for row in rows
        for message in row["conversations"]
        if message["from"] == "human"
    }
    assert human_prompts == set(prompts)
