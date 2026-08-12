
import pytest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import json
import sys

from run_agent import AIAgent


def _mock_response(*, usage: dict, content: str = "done"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(**usage),
    )


def _make_agent(session_db, *, platform: str):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=f"{platform}-session",
            platform=platform,
        )
    agent.client = MagicMock()
    agent._deferred_provider_runtime = None
    agent.client.chat.completions.create = AsyncMock(return_value=_mock_response(
        usage={
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }
    ))
    return agent


@pytest.mark.asyncio
async def test_run_conversation_persists_tokens_for_telegram_sessions(tmp_path):
    from hermes_state import SessionDB

    session_db = SessionDB(tmp_path / "state.db")
    agent = _make_agent(session_db, platform="telegram")
    try:
        result = await agent.run_conversation("hello")

        assert result["final_response"] == "done"
        session = await session_db.get_session("telegram-session")
        assert session is not None
        assert session["input_tokens"] == 11
        assert session["output_tokens"] == 7
    finally:
        await agent.close()
        await session_db.close()


@pytest.mark.asyncio
async def test_session_search_lazily_opens_db_when_entrypoint_did_not_pass_one(
    monkeypatch,
):
    sentinel_db = object()

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    agent = _make_agent(None, platform="acp")

    resolved = await agent._get_session_db_for_recall()

    assert resolved is sentinel_db
    assert agent._session_db is sentinel_db
    assert await agent._get_session_db_for_recall() is sentinel_db
