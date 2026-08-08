"""Opt-in live acceptance test for the complete async model/tool path.

Run with an authenticated provider, for example::

    HERMES_LIVE_TESTS=1 HERMES_LIVE_PROVIDER=copilot \
      pytest -q tests/e2e/test_live_provider_tool_path.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
async def test_live_provider_terminal_observation_and_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )

    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "max_iterations": 4,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "enabled_toolsets": ["terminal"],
        "save_trajectories": True,
        "session_db": SessionDB(tmp_path / "state.db"),
    }
    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")
        kwargs.update(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    agent = AIAgent(**kwargs)
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
        agent,
    ):
        result = await agent.run_conversation(
            "Call the terminal tool exactly once with `printf LIVE_ASYNC_OBSERVATION`. "
            "After reading its output, reply with exactly LIVE_ASYNC_FINAL."
        )

    assert result["completed"] is True
    assert "LIVE_ASYNC_FINAL" in result["final_response"]
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "LIVE_ASYNC_OBSERVATION" in result["messages"][2]["content"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectory_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
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
    assert '"name": "terminal"' in trajectory[2]["value"]
    assert "LIVE_ASYNC_OBSERVATION" in trajectory[3]["value"]
    assert "LIVE_ASYNC_FINAL" in trajectory[4]["value"]
