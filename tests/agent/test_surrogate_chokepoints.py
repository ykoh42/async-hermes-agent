"""Lone-surrogate chokepoint regression tests for retained async surfaces."""

import json
import inspect

import pytest

from agent.message_sanitization import _sanitize_structure_surrogates, _sanitize_surrogates
from agent.turn_finalizer import finalize_turn
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent


LONE_HIGH = "\ud83d"
LONE_LOW = "\udce7"


@pytest.mark.asyncio
async def test_finalize_turn_scrubs_lone_surrogate_from_final_response(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    dirty = f"answer {LONE_HIGH} and {LONE_LOW} here"
    result = await finalize_turn(
        agent,
        final_response=dirty,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": dirty},
        ],
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )
    final = result["final_response"]
    final.encode("utf-8")
    final.encode("utf-16-le")
    assert "\ufffd" in final
    assert "answer " in final and " here" in final


@pytest.mark.asyncio
async def test_finalize_turn_leaves_non_string_final_response_alone(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    result = await finalize_turn(
        FakeAgent(),
        final_response=None,
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "q"}],
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="interrupted",
    )
    assert result["final_response"] is None


def test_api_kwargs_walk_makes_tool_descriptions_json_safe():
    api_kwargs = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "session_search",
                    "description": f"±5 message window {LONE_HIGH} around the match",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"description": f"nested {LONE_LOW} leaf"}},
                    },
                },
            }
        ],
        "extra_body": {"note": f"deep {LONE_HIGH}"},
    }
    assert _sanitize_structure_surrogates(api_kwargs) is True
    encoded = json.dumps(api_kwargs)
    assert "\\ud83d" not in encoded.lower()
    assert "\ufffd" in api_kwargs["tools"][0]["function"]["description"]
    assert _sanitize_structure_surrogates(api_kwargs) is False


def test_conversation_loop_sanitizes_api_kwargs_after_build():
    import agent.conversation_loop as loop

    source = inspect.getsource(loop.run_conversation)
    build_idx = source.index("api_kwargs = await agent._build_api_kwargs(")
    sanitize_idx = source.index("_sanitize_structure_surrogates(api_kwargs)")
    perform_idx = source.index("async def _perform_api_call")
    assert build_idx < sanitize_idx < perform_idx


def test_sanitize_surrogates_preserves_valid_astral_pairs():
    text = "ok 😀 你好 𝕏"
    assert _sanitize_surrogates(text) == text
