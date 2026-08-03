"""Test that tool_name is correctly persisted to the session DB for tool-result messages.

make_tool_result_message() sets tool_name on every tool-result dict at construction
time. This test verifies that the value survives the flush path into the session DB.
"""

import pytest
from unittest.mock import patch

from run_agent import AIAgent
from agent.tool_dispatch_helpers import make_tool_result_message


def _make_agent(session_db):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
        )


@pytest.mark.asyncio
async def test_tool_name_persisted_to_session_db(tmp_path):
    """tool_name set by make_tool_result_message must be passed through to
    append_message so the column is populated on first flush to the session DB."""
    from hermes_state import SessionDB

    session_db = SessionDB(tmp_path / "state.db")
    agent = _make_agent(session_db)
    try:
        messages = [
            {"role": "user", "content": "run a command"},
            make_tool_result_message("terminal", "$ ls\nfile.txt", "c1"),
        ]
        await agent._flush_messages_to_session_db(messages)

        stored = await session_db.get_messages(agent.session_id)
        tool_messages = [message for message in stored if message["role"] == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_name"] == "terminal"
    finally:
        await agent.close()
        await session_db.close()
