"""Opt-in live acceptance test for context compression and continuation."""

from __future__ import annotations

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


def _provider_kwargs(model: str) -> dict:
    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
    }
    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")
        kwargs.update(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return kwargs


@pytest.mark.asyncio
async def test_live_provider_compression_summary_and_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.context_compressor import is_compaction_summary_message
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )

    history = []
    for index in range(8):
        history.append(
            {
                "role": "user",
                "content": (
                    f"Historical user turn {index}: "
                    + "alpha beta gamma " * 30
                ),
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Historical assistant turn {index}: "
                    + "delta epsilon zeta " * 30
                    + (" LIVE_COMPRESSION_TAIL_842" if index == 7 else "")
                ),
            }
        )

    database = SessionDB(tmp_path / "state.db")
    agent = AIAgent(
        **_provider_kwargs(model),
        max_iterations=2,
        disabled_toolsets=["*"],
        session_db=database,
        session_id="live-compression-origin",
    )
    try:
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
            no_task_leaks(action=LeakAction.RAISE),
            agent,
        ):
            initial = await agent.run_conversation(
                "Reply with exactly LIVE_COMPRESSION_INIT."
            )
            system_prompt = agent._cached_system_prompt
            compressor = agent.context_compressor
            compressor.protect_first_n = 1
            compressor.protect_last_n = 2
            compressor.threshold_tokens = 500
            compressor.tail_token_budget = 200
            compressor.max_summary_tokens = 400

            compressed, rebuilt_prompt = await agent._compress_context(
                [dict(message) for message in history],
                system_prompt,
                approx_tokens=4_000,
                force=True,
            )
            followup = await agent.run_conversation(
                "From the most recent preserved assistant turn, reply with only "
                "the token beginning LIVE_COMPRESSION_TAIL_.",
                conversation_history=compressed,
            )
            persisted = await database.get_messages(agent.session_id)
    finally:
        await agent.close()
        await database.close()

    summaries = [
        message for message in compressed if is_compaction_summary_message(message)
    ]
    assert initial["completed"] is True
    assert initial["final_response"].strip() == "LIVE_COMPRESSION_INIT"
    assert compressor.compression_count == 1
    assert compressor._last_compress_aborted is False
    assert compressor._last_summary_fallback_used is False
    assert compressor._last_summary_error is None
    assert len(compressed) < len(history)
    assert len(summaries) == 1
    assert rebuilt_prompt == system_prompt
    assert followup["completed"] is True
    assert followup["final_response"].strip() == "LIVE_COMPRESSION_TAIL_842"
    assert [message["role"] for message in persisted] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
