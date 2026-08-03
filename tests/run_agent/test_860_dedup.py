"""Tests for issue #860 — SQLite session transcript deduplication.

Verifies that:
1. _flush_messages_to_session_db uses _last_flushed_db_idx to avoid re-writing
2. Multiple _persist_session calls don't duplicate messages
3. _last_flushed_db_idx starts at zero
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Test: _flush_messages_to_session_db only writes new messages
# ---------------------------------------------------------------------------

class TestFlushDeduplication:
    """Verify _flush_messages_to_session_db tracks what it already wrote."""

    async def _make_agent(self, session_db):
        """Create a minimal AIAgent with a real session DB."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-session-860",
                skip_context_files=True,
                skip_memory=True,
            )
        # Simulate lazy session creation (normally done by run_conversation)
        await agent._ensure_db_session()
        return agent

    @pytest.mark.asyncio
    async def test_flush_writes_only_new_messages(self):
        """First flush writes all new messages, second flush writes none."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path)
            try:
                agent = await self._make_agent(db)

                conversation_history = [
                    {"role": "user", "content": "old message"},
                ]
                messages = list(conversation_history) + [
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "new answer"},
                ]

                # First flush — should write 2 new messages
                await agent._flush_messages_to_session_db(messages, conversation_history)

                rows = await db.get_messages(agent.session_id)
                assert len(rows) == 2, f"Expected 2 messages, got {len(rows)}"

                # Second flush with SAME messages — should write 0 new messages
                await agent._flush_messages_to_session_db(messages, conversation_history)

                rows = await db.get_messages(agent.session_id)
                assert len(rows) == 2, f"Expected still 2 messages after second flush, got {len(rows)}"
            finally:
                await agent.close()
                await db.close()



    @pytest.mark.asyncio
    async def test_flush_reset_after_compression(self):
        """After compression creates a new session, flush index resets."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path)
            try:
                agent = await self._make_agent(db)

                # Write some messages
                messages = [
                    {"role": "user", "content": "msg1"},
                    {"role": "assistant", "content": "reply1"},
                ]
                await agent._flush_messages_to_session_db(messages, [])

                old_session = agent.session_id
                assert agent._last_flushed_db_idx == 2

                # Simulate what _compress_context does: new session, reset idx
                agent.session_id = "compressed-session-new"
                await db.create_session(session_id=agent.session_id, source="test")
                agent._last_flushed_db_idx = 0

                # Now flush compressed messages to new session
                compressed_messages = [
                    {"role": "user", "content": "summary of conversation"},
                ]
                await agent._flush_messages_to_session_db(compressed_messages, [])

                new_rows = await db.get_messages(agent.session_id)
                assert len(new_rows) == 1

                # Old session should still have its 2 messages
                old_rows = await db.get_messages(old_session)
                assert len(old_rows) == 2
            finally:
                await agent.close()
                await db.close()


# ---------------------------------------------------------------------------
# Test: _last_flushed_db_idx initialization
# ---------------------------------------------------------------------------

class TestFlushIdxInit:
    """Verify _last_flushed_db_idx is properly initialized."""

    @pytest.mark.asyncio
    async def test_init_zero(self):
        """Agent starts with _last_flushed_db_idx = 0."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert agent._last_flushed_db_idx == 0
        await agent.close()
