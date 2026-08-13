"""Opt-in live acceptance test for memory persistence and session resume."""

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


def _provider_kwargs(model: str) -> dict:
    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "quiet_mode": True,
        "skip_context_files": True,
    }
    if PROVIDER == "lmstudio":
        # This path validates state and memory rather than reasoning depth.
        # GPT-OSS is less likely to emit a reasoning-only turn at low effort,
        # avoiding a model-parser retry that is unrelated to persistence.
        kwargs["reasoning_config"] = {"enabled": True, "effort": "low"}
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
async def test_live_provider_memory_and_cross_instance_session_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "memory:\n"
        "  memory_enabled: true\n"
        "  user_profile_enabled: true\n",
        encoding="utf-8",
    )
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )
    provider_kwargs = _provider_kwargs(model)
    state_path = tmp_path / "state.db"
    first_session_id = "live-state-origin"
    memory_marker = "LIVE_MEMORY_ORCHID_731"
    session_marker = "LIVE_SESSION_COBALT_942"

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        first_db = SessionDB(state_path)
        async with AIAgent(
            **provider_kwargs,
            session_db=first_db,
            session_id=first_session_id,
            max_iterations=4,
            enabled_toolsets=["memory"],
            save_trajectories=True,
        ) as first_agent:
            first_result = await first_agent.run_conversation(
                f"Remember that this session marker is {session_marker}, but do not "
                "write that session marker to memory. Call the memory tool exactly once "
                f"with target memory, action add, and content {memory_marker}. "
                "After the tool observation, reply exactly LIVE_MEMORY_SAVED."
            )

        memory_file = hermes_home / "memories" / "MEMORY.md"
        async with aiofiles.open(memory_file, encoding="utf-8") as handle:
            memory_text = await handle.read()
        assert memory_marker in memory_text
        assert session_marker not in memory_text
        assert first_result["completed"] is True
        assert first_result["final_response"].strip() == "LIVE_MEMORY_SAVED"
        assert [message["role"] for message in first_result["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]

        async with AIAgent(
            **provider_kwargs,
            session_id="live-memory-fresh",
            max_iterations=2,
            disabled_toolsets=["*"],
        ) as memory_agent:
            memory_result = await memory_agent.run_conversation(
                "Read persistent memory and reply with only the token beginning "
                "LIVE_MEMORY_."
            )
            assert memory_marker in memory_agent._cached_system_prompt

        assert memory_result["completed"] is True
        assert memory_result["final_response"].strip() == memory_marker

        resume_db = SessionDB(state_path)
        tip = await resume_db.resolve_resume_session_id(first_session_id)
        model_history, display_history = await resume_db.get_resume_conversations(tip)
        assert [message["role"] for message in model_history] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert [message["role"] for message in display_history] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]

        async with AIAgent(
            **provider_kwargs,
            session_db=resume_db,
            session_id=tip,
            max_iterations=2,
            disabled_toolsets=["*"],
        ) as resumed_agent:
            resumed_result = await resumed_agent.run_conversation(
                "From the resumed conversation, reply with only the session marker "
                "beginning LIVE_SESSION_.",
                conversation_history=model_history,
            )

        assert resumed_result["completed"] is True
        assert resumed_result["final_response"].strip() == session_marker
        assert [message["role"] for message in resumed_result["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant",
        ]

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
    assert '"name": "memory"' in trajectory[2]["value"]
    assert memory_marker in trajectory[2]["value"]
    assert "Entry added" in trajectory[3]["value"]
