"""AIAgent enters turns only after acquiring and reloading durable state."""

from __future__ import annotations

import asyncio

import pytest

from run_agent import AIAgent


class _DB:
    def __init__(self, session_exists: bool = True) -> None:
        self.events: list[tuple] = []
        self.session_exists = session_exists

    async def get_session(self, session_id: str):
        return {"id": session_id} if self.session_exists else None

    async def acquire_session_turn_lease(
        self, session_id: str, holder: str, **kwargs
    ) -> bool:
        self.events.append(("acquire", session_id, holder))
        on_wait = kwargs.get("on_wait")
        if on_wait is not None:
            on_wait(0.0)
        return True

    async def resolve_resume_session_id(self, session_id: str) -> str:
        self.events.append(("resolve", session_id))
        return "compressed-tip"

    async def get_messages_as_conversation(self, session_id: str, **kwargs):
        self.events.append(("reload", session_id, kwargs))
        return [{"role": "user", "content": "durable latest"}]

    async def refresh_session_turn_lease(
        self, session_id: str, holder: str, **kwargs
    ) -> bool:
        return True

    async def release_session_turn_lease(self, session_id: str, holder: str) -> None:
        self.events.append(("release", session_id, holder))


def _bare_agent(db: _DB, *, session_id: str, session_created: bool) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.platform = "desktop"
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = session_created
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._session_runtime_config = None
    agent._interrupt_event = asyncio.Event()
    agent._active_turn_task = None
    agent._current_turn_id = None
    agent.status_callback = None
    agent._get_turn_lock = lambda: asyncio.Lock()

    async def reset_activity_labels() -> None:
        return None

    async def conversation_root() -> str:
        return "stale-parent"

    agent._reset_activity_labels_after_turn = reset_activity_labels
    agent._conversation_root_id = conversation_root
    return agent


@pytest.mark.asyncio
async def test_run_conversation_acquires_then_reloads_latest_tip(monkeypatch):
    db = _DB()
    agent = _bare_agent(db, session_id="stale-parent", session_created=True)
    observed: dict[str, object] = {}

    async def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        observed["session_id"] = _agent.session_id
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("run_agent._conversation_loop.run_conversation", fake_run)
    result = await agent.run_conversation(
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )

    assert result["final_response"] == "ok"
    assert observed == {
        "history": [{"role": "user", "content": "durable latest"}],
        "session_id": "compressed-tip",
    }
    assert [event[0] for event in db.events] == [
        "acquire",
        "resolve",
        "reload",
        "release",
    ]


@pytest.mark.asyncio
async def test_fresh_session_keeps_caller_seed_without_durable_lease(monkeypatch):
    db = _DB(session_exists=False)
    agent = _bare_agent(db, session_id="fresh", session_created=False)
    agent.platform = "subagent"
    agent._parent_session_id = "parent"
    observed: dict[str, object] = {}

    async def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("run_agent._conversation_loop.run_conversation", fake_run)
    seed = [{"role": "user", "content": "delegated context"}]

    await agent.run_conversation("work", conversation_history=seed)

    assert observed["history"] is seed
    assert db.events == []
