"""Opt-in live acceptance test for BatchRunner trajectory and resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles
import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction


LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"
PROVIDER = (os.environ.get("HERMES_LIVE_PROVIDER") or "copilot").strip().lower()
DEFAULT_MODELS = {
    "copilot": "gpt-4.1",
    "openrouter": "google/gemini-2.5-flash",
}

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="live-only — set HERMES_LIVE_TESTS=1",
)


@pytest.mark.asyncio
async def test_live_batch_runner_checkpoint_resume_and_ordered_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from batch_runner import BatchRunner

    monkeypatch.chdir(tmp_path)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True)
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )
    (hermes_home / "config.yaml").write_text(
        f"model:\n  provider: {PROVIDER}\n  default: {model}\n",
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset.jsonl"
    prompt = (
        "Call terminal exactly once with `printf LIVE_BATCH_OBSERVATION`. "
        "Before the tool call include "
        "<REASONING_SCRATCHPAD>live batch reasoning</REASONING_SCRATCHPAD>. "
        "After reading the observation reply exactly LIVE_BATCH_FINAL."
    )
    dataset.write_text(json.dumps({"prompt": prompt}) + "\n", encoding="utf-8")
    runner_kwargs = {
        "dataset_file": str(dataset),
        "batch_size": 1,
        "run_name": "live-batch",
        "distribution": "terminal_only",
        "model": model,
        "num_workers": 1,
        "max_iterations": 4,
        "ephemeral_system_prompt": (
            "For this evaluation, include a <REASONING_SCRATCHPAD> block in "
            "the assistant tool-call turn before using a tool."
        ),
    }
    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")
        runner_kwargs.update(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    runner = BatchRunner(**runner_kwargs)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        assert await runner.run() is None
        output_dir = tmp_path / "data" / "live-batch"
        shard = output_dir / "batch_0.jsonl"
        async with aiofiles.open(shard, "rb") as handle:
            before_resume = await handle.read()
        assert await runner.run(resume=True) is None
        async with aiofiles.open(shard, "rb") as handle:
            after_resume = await handle.read()

    assert before_resume == after_resume
    checkpoint = json.loads((output_dir / "checkpoint.json").read_text())
    assert checkpoint["completed_prompts"] == [0]

    shard_rows = [json.loads(line) for line in before_resume.splitlines()]
    assert len(shard_rows) == 1
    conversations = shard_rows[0]["conversations"]
    assert [turn["from"] for turn in conversations] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert "<think>\nlive batch reasoning\n</think>" in conversations[2]["value"]
    assert '"name": "terminal"' in conversations[2]["value"]
    assert "LIVE_BATCH_OBSERVATION" in conversations[3]["value"]
    assert conversations[4]["value"].endswith("LIVE_BATCH_FINAL")

    merged_rows = [
        json.loads(line)
        for line in (output_dir / "trajectories.jsonl").read_text().splitlines()
    ]
    assert merged_rows == shard_rows
    statistics = json.loads((output_dir / "statistics.json").read_text())
    assert statistics["reasoning_statistics"]["turns_with_reasoning"] >= 1
    assert statistics["tool_statistics"]["terminal"]["count"] == 1
    assert statistics["tool_statistics"]["terminal"]["success"] == 1
    assert statistics["tool_statistics"]["terminal"]["failure"] == 0
