"""Opt-in live acceptance test for provider timeout persistence and recovery."""

from __future__ import annotations

import asyncio
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
async def test_live_provider_timeout_persists_and_next_turn_recovers(
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
    database = SessionDB(tmp_path / "state.db")
    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "max_iterations": 2,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "disabled_toolsets": ["*"],
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
    cancelled_input = "This live provider request must be timed out and persisted."
    try:
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
            no_task_leaks(action=LeakAction.RAISE),
            agent,
        ):
            timeout_trigger = None
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(None) as timeout_scope:

                    async def expire_when_provider_is_active() -> None:
                        await asyncio.wait_for(
                            agent._model_request_active.wait(),
                            timeout=10,
                        )
                        timeout_scope.reschedule(asyncio.get_running_loop().time())

                    timeout_trigger = asyncio.create_task(
                        expire_when_provider_is_active()
                    )
                    try:
                        await agent.run_conversation(cancelled_input)
                    finally:
                        await timeout_trigger

            assert timeout_trigger is not None and timeout_trigger.done()
            assert agent._inflight_turn_id is None
            cancelled_rows = await agent._session_db.get_messages(agent.session_id)
            assert [row["role"] for row in cancelled_rows] == ["user"]
            assert cancelled_rows[0]["content"] == cancelled_input

            recovered = await agent.run_conversation(
                "Ignore the interrupted request and reply exactly LIVE_TIMEOUT_RECOVERED."
            )

            assert recovered["completed"] is True
            assert recovered["final_response"].strip() == "LIVE_TIMEOUT_RECOVERED"
            assert agent._inflight_turn_id is None
            persisted = await agent._session_db.get_messages(agent.session_id)
            assert [row["role"] for row in persisted] == [
                "user",
                "user",
                "assistant",
            ]
            assert persisted[-1]["content"] == "LIVE_TIMEOUT_RECOVERED"
    finally:
        await database.close()
