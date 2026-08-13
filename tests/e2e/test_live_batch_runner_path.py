"""Opt-in reasoning-provider acceptance test for BatchRunner and resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles
import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tests.e2e.trajectory_assertions import assert_exact_terminal_trajectory


LIVE = os.environ.get("HERMES_LIVE_REASONING_TESTS") == "1"
PROVIDER = (
    os.environ.get("HERMES_LIVE_REASONING_PROVIDER") or "lmstudio"
).strip().lower()

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="reasoning-live-only — set HERMES_LIVE_REASONING_TESTS=1",
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
    model = (os.environ.get("HERMES_LIVE_REASONING_MODEL") or "").strip()
    if not model:
        pytest.fail("HERMES_LIVE_REASONING_MODEL is required")
    (hermes_home / "config.yaml").write_text(
        f"model:\n  provider: {PROVIDER}\n  default: {model}\n",
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset.jsonl"
    observations = ["LIVE_BATCH_OBSERVATION_0", "LIVE_BATCH_OBSERVATION_1"]
    finals = ["LIVE_BATCH_FINAL_0", "LIVE_BATCH_FINAL_1"]
    commands = [f"printf {observation}" for observation in observations]
    prompts = [
        (
            "Make exactly two model responses. In the first response, emit no "
            "visible text and call terminal exactly once with exactly these "
            f'arguments and no extra keys: {{"command":"{command}"}}. After '
            "the tool observation, make no further tool calls and emit exactly "
            f"this visible final answer with no other text: {final}"
        )
        for command, final in zip(commands, finals, strict=True)
    ]
    dataset.write_text(
        json.dumps({"prompt": prompts[0]}) + "\n",
        encoding="utf-8",
    )
    runner_kwargs = {
        "dataset_file": str(dataset),
        "batch_size": 1,
        "run_name": "live-batch",
        "distribution": "terminal_only",
        "model": model,
        "num_workers": 1,
        "max_iterations": 4,
        "reasoning_config": {"enabled": True, "effort": "high"},
    }

    base_url = (os.environ.get("HERMES_LIVE_REASONING_BASE_URL") or "").strip()
    api_key = (os.environ.get("HERMES_LIVE_REASONING_API_KEY") or "").strip()
    if PROVIDER == "lmstudio":
        base_url = base_url or "http://127.0.0.1:1234/v1"
        api_key = api_key or "lm-studio"
    elif PROVIDER == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")
    if base_url:
        runner_kwargs["base_url"] = base_url
    if api_key:
        runner_kwargs["api_key"] = api_key
    first_runner = BatchRunner(**runner_kwargs)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        assert await first_runner.run() is None
        output_dir = tmp_path / "data" / "live-batch"
        shard = output_dir / "batch_0.jsonl"
        async with aiofiles.open(shard, "rb") as handle:
            first_shard = await handle.read()
        async with aiofiles.open(
            output_dir / "checkpoint.json", encoding="utf-8"
        ) as handle:
            first_checkpoint = json.loads(await handle.read())
        async with aiofiles.open(
            output_dir / "statistics.json", encoding="utf-8"
        ) as handle:
            first_statistics = json.loads(await handle.read())

        async with aiofiles.open(dataset, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps({"prompt": prompts[1]}) + "\n")
            await handle.flush()

        # A process restart constructs a fresh runner. Its lazy dataset load
        # must see the appended prompt while content-based resume skips prompt 0.
        restarted_runner = BatchRunner(**runner_kwargs)
        assert await restarted_runner.run(resume=True) is None
        async with aiofiles.open(shard, "rb") as handle:
            resumed_shard = await handle.read()
        async with aiofiles.open(
            output_dir / "checkpoint.json", encoding="utf-8"
        ) as handle:
            resumed_checkpoint = json.loads(await handle.read())
        async with aiofiles.open(
            output_dir / "trajectories.jsonl", encoding="utf-8"
        ) as handle:
            merged_rows = [
                json.loads(line) for line in (await handle.read()).splitlines()
            ]
        async with aiofiles.open(
            output_dir / "statistics.json", encoding="utf-8"
        ) as handle:
            resumed_statistics = json.loads(await handle.read())

    assert first_checkpoint["completed_prompts"] == [0]
    assert resumed_checkpoint["completed_prompts"] == [0, 1]
    assert resumed_shard.startswith(first_shard)
    dataset_rows = [
        json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
    ]
    assert dataset_rows == [{"prompt": prompts[0]}, {"prompt": prompts[1]}]

    first_rows = [json.loads(line) for line in first_shard.splitlines()]
    resumed_rows = [json.loads(line) for line in resumed_shard.splitlines()]
    assert len(first_rows) == 1
    assert len(resumed_rows) == 2
    assert resumed_rows[0] == first_rows[0]
    assert [row["prompt_index"] for row in resumed_rows] == [0, 1]
    assert merged_rows == resumed_rows

    for index, row in enumerate(resumed_rows):
        assert row["completed"] is True
        assert row["partial"] is False
        assert row["api_calls"] == 2
        assert row["tool_stats"]["terminal"] == {
            "count": 1,
            "success": 1,
            "failure": 0,
        }
        assert row["tool_error_counts"]["terminal"] == 0
        assert_exact_terminal_trajectory(
            row["conversations"],
            prompt=prompts[index],
            command=commands[index],
            observation=observations[index],
            final=finals[index],
        )

    expected_run_terminal_stats = {
        "count": 1,
        "success": 1,
        "failure": 0,
        "success_rate": 100.0,
        "failure_rate": 0.0,
    }
    assert first_statistics["total_prompts"] == 1
    assert first_statistics["tool_statistics"]["terminal"] == (
        expected_run_terminal_stats
    )
    assert resumed_statistics["total_prompts"] == 2
    # Upstream statistics.json describes the current run, while checkpoint and
    # merged shards preserve cross-run progress. The resume run processed one.
    assert resumed_statistics["tool_statistics"]["terminal"] == (
        expected_run_terminal_stats
    )
