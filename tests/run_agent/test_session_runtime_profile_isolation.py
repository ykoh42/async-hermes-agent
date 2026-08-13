from __future__ import annotations

import asyncio

import pytest

from agent import agent_init, secret_scope
from hermes_state import _cjk_fts_config_enabled
from hermes_state_common import _session_runtime_config_value
from run_agent import AIAgent


def _bare_agent(label: str, *, cjk_fts: bool, search_slow_ms: int) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = f"session-{label}"
    agent._turn_lock = asyncio.Lock()
    agent._active_turn_task = None
    agent._current_turn_id = None
    agent._interrupt_event = asyncio.Event()
    agent._session_db = None
    agent._session_activity_persist_task = None
    agent._session_runtime_config = {
        "cjk_fts": cjk_fts,
        "search_slow_ms": search_slow_ms,
    }
    return agent


def test_runtime_config_does_not_publish_profile_settings_to_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_CJK_FTS", "foreign")
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "999")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    agent = type("Agent", (), {})()

    agent_init._apply_runtime_config(
        agent,
        {"sessions": {"cjk_fts": False, "search_slow_ms": 0}},
    )

    assert agent._session_runtime_config == {
        "cjk_fts": False,
        "search_slow_ms": 0,
    }
    assert _cjk_fts_config_enabled() is True
    assert _session_runtime_config_value(
        "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
    ) == "999"
    assert secret_scope.is_multiplex_active() is True


@pytest.mark.asyncio
async def test_concurrent_agent_turns_bind_distinct_session_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_CJK_FTS", "foreign")
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "999")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    agents = {
        "a": _bare_agent("a", cjk_fts=False, search_slow_ms=0),
        "b": _bare_agent("b", cjk_fts=True, search_slow_ms=60_000),
    }
    observed: dict[str, tuple[bool, str]] = {}
    both_entered = asyncio.Event()
    entered = 0

    async def fake_conversation(agent: AIAgent, *_args, **_kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()
        label = agent.session_id.removeprefix("session-")
        observed[label] = (
            _cjk_fts_config_enabled(),
            _session_runtime_config_value(
                "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
            ),
        )
        return {"final_response": label}

    monkeypatch.setattr(
        "run_agent._conversation_loop.run_conversation", fake_conversation
    )

    results = await asyncio.gather(
        *(agent.run_conversation(label) for label, agent in agents.items())
    )

    assert [result["final_response"] for result in results] == ["a", "b"]
    assert observed == {"a": (False, "0"), "b": (True, "60000")}
    # The binding is turn-local and restores the caller's legacy fallback.
    assert _cjk_fts_config_enabled() is True
    assert _session_runtime_config_value(
        "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
    ) == "999"


@pytest.mark.asyncio
async def test_agent_lifecycle_boundaries_bind_and_restore_session_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_CJK_FTS", "1")
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "999")
    agent = _bare_agent("lifecycle", cjk_fts=False, search_slow_ms=7)
    observed: list[tuple[str, bool, str]] = []

    async def ensure_runtime():
        observed.append(
            (
                "enter",
                _cjk_fts_config_enabled(),
                _session_runtime_config_value(
                    "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
                ),
            )
        )
        raise RuntimeError("stop after deferred runtime")

    async def switch_model(_agent, *_args):
        observed.append(
            (
                "switch",
                _cjk_fts_config_enabled(),
                _session_runtime_config_value(
                    "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
                ),
            )
        )
        return "switched"

    monkeypatch.setattr(agent, "_ensure_provider_runtime", ensure_runtime)
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch_model)

    with pytest.raises(RuntimeError, match="deferred runtime"):
        await agent.__aenter__()
    assert await agent.switch_model("model", "provider") == "switched"
    assert observed == [
        ("enter", False, "7"),
        ("switch", False, "7"),
    ]
    assert _cjk_fts_config_enabled() is True
    assert _session_runtime_config_value(
        "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
    ) == "999"


@pytest.mark.asyncio
async def test_close_keeps_binding_through_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "999")
    agent = _bare_agent("close", cjk_fts=False, search_slow_ms=7)
    agent._closed = False
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    observed: list[str] = []

    async def close_unlocked():
        observed.append(
            _session_runtime_config_value(
                "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
            )
        )
        close_entered.set()
        await release_close.wait()
        agent._closed = True

    monkeypatch.setattr(agent, "_close_unlocked", close_unlocked)
    close_task = asyncio.create_task(agent.close())
    await close_entered.wait()
    close_task.cancel("first")
    await asyncio.sleep(0)
    close_task.cancel("second")
    await asyncio.sleep(0)
    assert not close_task.done()
    release_close.set()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await close_task

    assert cancellation.value.args == ("first",)
    assert observed == ["7"]
    assert agent._closed is True
    assert not agent._get_close_lock().locked()
    assert _session_runtime_config_value(
        "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
    ) == "999"


def test_single_profile_process_bridge_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("HERMES_CJK_FTS", "1")
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "1000")
    agent = type("Agent", (), {})()

    agent_init._apply_runtime_config(
        agent,
        {"sessions": {"cjk_fts": False, "search_slow_ms": 7}},
    )

    assert _cjk_fts_config_enabled() is False
    assert _session_runtime_config_value(
        "search_slow_ms", "HERMES_SEARCH_SLOW_MS", "1000"
    ) == "7"
