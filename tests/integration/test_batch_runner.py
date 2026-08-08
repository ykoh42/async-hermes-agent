"""End-to-end async scheduling coverage for :class:`BatchRunner`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles
import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

import batch_runner
from batch_runner import BatchRunner


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_batch_runner_bounds_concurrency_and_writes_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    async with aiofiles.open(dataset, "w", encoding="utf-8") as output:
        for index in range(6):
            await output.write(json.dumps({"prompt": f"prompt {index}"}) + "\n")

    active = 0
    peak_active = 0

    async def process_prompt(
        prompt_index: int,
        prompt_data: dict[str, Any],
        batch_num: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.03)
        finally:
            active -= 1
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
            "metadata": {"batch_num": batch_num, "model": config["model"]},
        }

    monkeypatch.setattr(batch_runner, "_process_single_prompt", process_prompt)
    runner = BatchRunner(
        dataset_file=str(dataset),
        batch_size=1,
        run_name="bounded",
        model="integration-model",
        num_workers=2,
        verbose=False,
    )

    # Warm the aiofiles worker before measuring the runtime loop; thread
    # startup is an implementation detail of the accepted file backend.
    await aiofiles.os.stat(dataset)
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        await runner.run(resume=False)

    assert peak_active == 2
    checkpoint = json.loads(runner.checkpoint_file.read_text(encoding="utf-8"))
    statistics = json.loads(runner.stats_file.read_text(encoding="utf-8"))
    trajectory_rows = (runner.output_dir / "trajectories.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert checkpoint["completed_prompts"] == list(range(6))
    assert statistics["total_prompts"] == 6
    assert statistics["total_batches"] == 6
    assert len(trajectory_rows) == 6
