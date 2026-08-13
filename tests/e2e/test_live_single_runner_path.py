"""Opt-in live acceptance test for the retained single-task runner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.auxiliary_client import scoped_runtime_main
from mini_swe_runner import MiniSWERunner
from tests.e2e.trajectory_assertions import assert_exact_terminal_trajectory


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
async def test_live_single_runner_tool_observation_trajectory_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )
    if PROVIDER == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")

    observation = "LIVE_SINGLE_RUNNER_OBSERVATION_8C42"
    final = "LIVE_SINGLE_RUNNER_FINAL_8C42"
    command = f"printf {observation}"
    prompt = (
        "Make exactly two model responses. In the first response, emit no "
        "visible text and call terminal exactly once with exactly these "
        f'arguments and no extra keys: {{"command":"{command}"}}. After the '
        "tool observation, make no further tool calls and emit exactly this "
        f"visible final answer with no other text: {final}"
    )
    runner = MiniSWERunner(
        model=model,
        cwd=str(tmp_path),
        max_iterations=3,
        command_timeout=10,
    )

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        with scoped_runtime_main({"provider": PROVIDER, "model": model}):
            result = await runner.run_task(prompt)

    conversations = result["conversations"]
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert_exact_terminal_trajectory(
        conversations,
        prompt=prompt,
        command=command,
        observation=observation,
        final=final,
    )
    assert runner.client is None
    assert runner.env is None
