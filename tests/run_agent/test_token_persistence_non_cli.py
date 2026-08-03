
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
