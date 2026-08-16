"""Opt-in live acceptance test for provider → subagent → queued turn."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tests.e2e.trajectory_assertions import split_optional_think


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
async def test_live_provider_subagent_completion_queue_reenters_parent_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "delegation:\n  max_iterations: 2\n",
        encoding="utf-8",
    )

    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.async_delegation import (
        claim_event_delivery,
        complete_event_delivery,
        get_durable_delegation,
        release_event_delivery,
    )
    from tools.delegate_tool import list_active_subagents
    from tools.process_registry import (
        format_process_notification,
        process_registry,
    )

    # The singleton can have restored events from another test/profile when
    # this module runs in a shared pytest process. Preserve that queue and give
    # this live acceptance path a clean, turn-local delivery rail.
    monkeypatch.setattr(process_registry, "completion_queue", asyncio.Queue())

    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )

    database = SessionDB(tmp_path / "state.db")
    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "max_iterations": 4,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "enabled_toolsets": ["delegation"],
        "save_trajectories": True,
        "session_db": database,
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
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25)
        )
        await stack.enter_async_context(no_task_leaks(action=LeakAction.RAISE))
        stack.push_async_callback(database.close)
        await stack.enter_async_context(agent)
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
            async with asyncio.timeout(45):
                while True:
                    drained = process_registry.drain_notifications(
                        session_key=agent.session_id
                    )
                    if drained:
                        assert len(drained) == 1
                        completion_event, completion_message = drained[0]
                        break
                    await asyncio.sleep(0.05)
        except TimeoutError:
            pytest.fail(
                "background subagent did not publish its completion; "
                f"background={len(agent._background_delegations)}, "
                f"active_children={len(agent._active_children)}, "
                f"active_subagents={list_active_subagents()!r}"
            )

        assert completion_event["delegation_id"] == dispatch["delegation_id"]
        (child_result,) = completion_event["results"]
        assert child_result["status"] == "completed"
        assert child_result["summary"].strip() == "LIVE_SUBAGENT_CHILD"

        assert completion_message == format_process_notification(completion_event)
        assert completion_message.startswith("[ASYNC DELEGATION BATCH COMPLETE")
        assert "LIVE_SUBAGENT_CHILD" in completion_message

        # Match the upstream driver boundary: completion_queue → formatter →
        # a fresh parent turn. A bare AIAgent does not auto-inject completions
        # through event_callback.
        delivery_claim = await claim_event_delivery(
            completion_event, "live-e2e-subagent"
        )
        assert delivery_claim is not None
        try:
            reinjected = await agent.run_conversation(completion_message)
        except BaseException:
            await release_event_delivery(completion_event, delivery_claim)
            raise
        else:
            await complete_event_delivery(completion_event, delivery_claim)
        durable = await get_durable_delegation(dispatch["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
        assert durable["delivery_attempts"] == 1
        assert reinjected["completed"] is True
        assert reinjected["final_response"].strip() == "LIVE_SUBAGENT_PARENT_FINAL"
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
    _thinking, visible_dispatch = split_optional_think(
        rows[0]["conversations"][4]["value"]
    )
    assert visible_dispatch == "LIVE_SUBAGENT_DISPATCHED"
    assert [turn["from"] for turn in rows[1]["conversations"]] == [
        "system",
        "human",
        "gpt",
    ]
    assert rows[1]["conversations"][1]["value"] == completion_message
    _thinking, visible_parent_final = split_optional_think(
        rows[1]["conversations"][-1]["value"]
    )
    assert visible_parent_final == "LIVE_SUBAGENT_PARENT_FINAL"
