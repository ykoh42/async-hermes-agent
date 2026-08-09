"""Opt-in live acceptance test for native provider streaming lifecycle."""

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
async def test_live_provider_streams_text_and_cleans_owned_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "max_iterations": 2,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "disabled_toolsets": ["*"],
    }
    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")
        kwargs.update(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    deltas: list[str] = []
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
        AIAgent(**kwargs) as agent,
    ):
        result = await agent.run_conversation(
            "Reply with exactly LIVE_NATIVE_STREAM_OK.",
            stream_callback=deltas.append,
        )

    assert result["completed"] is True
    assert result["final_response"].strip() == "LIVE_NATIVE_STREAM_OK"
    assert "LIVE_NATIVE_STREAM_OK" in "".join(deltas)
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("provider-stream-heartbeat-")
    ]
