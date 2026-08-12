"""Opt-in live acceptance test for reasoning-preserving trajectories.

The selected provider must expose reasoning and tool calls. For example, with
an already-loaded LM Studio model::

    HERMES_LIVE_REASONING_TESTS=1 \
      HERMES_LIVE_REASONING_PROVIDER=lmstudio \
      HERMES_LIVE_REASONING_MODEL=async-hermes-reasoning \
      pytest -q tests/e2e/test_live_reasoning_trajectory_path.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles
import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction


LIVE = os.environ.get("HERMES_LIVE_REASONING_TESTS") == "1"
PROVIDER = (
    os.environ.get("HERMES_LIVE_REASONING_PROVIDER") or "lmstudio"
).strip().lower()

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="reasoning-live-only — set HERMES_LIVE_REASONING_TESTS=1",
)


@pytest.mark.asyncio
async def test_live_reasoning_tool_observation_and_final_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    model = (os.environ.get("HERMES_LIVE_REASONING_MODEL") or "").strip()
    if not model:
        pytest.fail("HERMES_LIVE_REASONING_MODEL is required")

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

    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "reasoning_config": {"enabled": True, "effort": "high"},
        "max_iterations": 4,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "enabled_toolsets": ["terminal"],
        "save_trajectories": True,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    observation = "LIVE_REASONING_OBSERVATION_91D7"
    final = "LIVE_REASONING_FINAL_91D7"
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
        AIAgent(**kwargs) as agent,
    ):
        result = await agent.run_conversation(
            "Call the terminal tool exactly once with command "
            f"`printf {observation}`. After reading its result, call no more "
            f"tools and reply exactly {final}."
        )

    assert result["completed"] is True, result
    trajectory_path = tmp_path / "trajectory_samples.jsonl"
    assert trajectory_path.exists(), result
    async with aiofiles.open(
        trajectory_path,
        encoding="utf-8",
    ) as handle:
        rows = [json.loads(line) for line in (await handle.read()).splitlines()]

    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert len(rows) == 1
    trajectory = rows[0]["conversations"]
    assert [turn["from"] for turn in trajectory] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert trajectory[2]["value"].startswith("<think>\n")
    assert '"name": "terminal"' in trajectory[2]["value"]
    assert observation in trajectory[3]["value"]
    assert trajectory[4]["value"].startswith("<think>\n")
    assert final in trajectory[4]["value"]
