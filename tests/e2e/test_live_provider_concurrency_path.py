"""Opt-in live acceptance tests for provider concurrency contracts."""

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


def _provider_kwargs(model: str) -> dict:
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
    return kwargs


@pytest.mark.asyncio
async def test_live_provider_instances_overlap_and_shared_agent_serializes_turns(
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
    kwargs = _provider_kwargs(model)

    first = AIAgent(**kwargs, session_id="live-concurrency-first")
    second = AIAgent(**kwargs, session_id="live-concurrency-second")
    shared = AIAgent(**kwargs, session_id="live-concurrency-shared")

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
        first,
        second,
        shared,
    ):
        first_task = asyncio.create_task(
            first.run_conversation("Reply exactly LIVE_CONCURRENT_FIRST."),
            name="live-provider-first",
        )
        second_task = asyncio.create_task(
            second.run_conversation("Reply exactly LIVE_CONCURRENT_SECOND."),
            name="live-provider-second",
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    first._model_request_active.wait(),
                    second._model_request_active.wait(),
                ),
                timeout=10,
            )
            assert first._model_request_active.is_set()
            assert second._model_request_active.is_set()
            first_result, second_result = await asyncio.gather(
                first_task,
                second_task,
            )
        finally:
            for task in (first_task, second_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first_task, second_task, return_exceptions=True)

        assert first_result["final_response"].strip() == "LIVE_CONCURRENT_FIRST"
        assert second_result["final_response"].strip() == "LIVE_CONCURRENT_SECOND"

        shared_first = asyncio.create_task(
            shared.run_conversation("Reply exactly LIVE_SHARED_FIRST."),
            name="live-shared-first",
        )
        await asyncio.wait_for(shared._model_request_active.wait(), timeout=10)
        cached_prompt = shared._cached_system_prompt
        shared_second = asyncio.create_task(
            shared.run_conversation("Reply exactly LIVE_SHARED_SECOND."),
            name="live-shared-second",
        )
        try:
            await asyncio.sleep(0)
            assert shared._turn_lock.locked()
            assert not shared_second.done()
            shared_first_result, shared_second_result = await asyncio.gather(
                shared_first,
                shared_second,
            )
        finally:
            for task in (shared_first, shared_second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(shared_first, shared_second, return_exceptions=True)

        assert shared_first_result["final_response"].strip() == "LIVE_SHARED_FIRST"
        assert shared_second_result["final_response"].strip() == "LIVE_SHARED_SECOND"
        assert shared._cached_system_prompt == cached_prompt
        assert [message["role"] for message in shared_second_result["messages"]] == [
            "user",
            "assistant",
        ]
