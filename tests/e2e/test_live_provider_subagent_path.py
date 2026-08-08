"""Opt-in live acceptance test for provider → subagent → parent reinjection."""

from __future__ import annotations

import asyncio
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
async def test_live_provider_subagent_completion_reenters_parent_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.delegate_tool import list_active_subagents

    monkeypatch.chdir(tmp_path)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "delegation:\n  max_iterations: 2\n",
        encoding="utf-8",
    )
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )

    completion = asyncio.Event()
    completion_payload = None
    event_names: list[str] = []

    def record_event(name: str, payload: dict) -> None:
        nonlocal completion_payload
        event_names.append(name)
        if name == "delegation:complete":
            completion_payload = payload
            completion.set()

    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "max_iterations": 4,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "enabled_toolsets": ["delegation"],
        "save_trajectories": True,
        "session_db": SessionDB(tmp_path / "state.db"),
        "event_callback": record_event,
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
        initial = await agent.run_conversation(
            "Call delegate_task exactly once with goal: Reply exactly "
            "LIVE_SUBAGENT_CHILD. The tool dispatches in the background. "
            "After its immediate dispatch observation, reply exactly "
            "LIVE_SUBAGENT_DISPATCHED. When the async delegation completion is "
            "later injected as a new message, reply exactly "
            "LIVE_SUBAGENT_PARENT_FINAL."
        )
        assert initial["completed"] is True
        assert initial["final_response"].strip() == "LIVE_SUBAGENT_DISPATCHED"
        assert [message["role"] for message in initial["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        dispatch = json.loads(initial["messages"][2]["content"])
        assert dispatch.get("status") == "dispatched", dispatch
        assert dispatch["mode"] == "background"
        try:
            await asyncio.wait_for(completion.wait(), timeout=45)
        except TimeoutError:
            pytest.fail(
                "background subagent did not reinject completion; "
                f"events={event_names!r}, "
                f"background={len(agent._background_delegations)}, "
                f"active_children={len(agent._active_children)}, "
                f"active_subagents={list_active_subagents()!r}"
            )
        await asyncio.sleep(0)

        assert completion_payload is not None
        (child_result,) = completion_payload["results"]
        assert child_result["status"] == "completed"
        assert child_result["summary"].strip() == "LIVE_SUBAGENT_CHILD"
        reinjected = completion_payload["response"]
        assert reinjected["completed"] is True
        assert "LIVE_SUBAGENT_CHILD" in reinjected["final_response"]
        assert [message["role"] for message in reinjected["messages"]] == [
            "user",
            "assistant",
        ]
        assert agent._active_children == []
        assert agent._background_delegations == set()
        assert list_active_subagents() == []
        assert all(
            "SessionDB is closed" not in record.getMessage()
            for record in caplog.records
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectory_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert [turn["from"] for turn in rows[0]["conversations"]] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert '"name": "delegate_task"' in rows[0]["conversations"][2]["value"]
    assert '"status": "dispatched"' in rows[0]["conversations"][3]["value"]
    assert rows[0]["conversations"][4]["value"].endswith(
        "LIVE_SUBAGENT_DISPATCHED"
    )
    assert [turn["from"] for turn in rows[1]["conversations"]] == [
        "system",
        "human",
        "gpt",
    ]
    assert "LIVE_SUBAGENT_CHILD" in rows[1]["conversations"][1]["value"]
    assert rows[1]["conversations"][-1]["value"].endswith(
        reinjected["final_response"]
    )
