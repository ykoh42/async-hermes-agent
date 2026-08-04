"""Tests for the interrupt system.

Run with: python -m pytest tests/test_interrupt.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# Unit tests: shared interrupt module
# ---------------------------------------------------------------------------

class TestInterruptModule:
    """Tests for tools/interrupt.py"""

    def test_set_and_check(self):
        from tools.interrupt import set_interrupt, is_interrupted
        set_interrupt(False)
        assert not is_interrupted()

        set_interrupt(True)
        assert is_interrupted()

        set_interrupt(False)
        assert not is_interrupted()


    @pytest.mark.asyncio
    async def test_child_task_inherits_current_interrupt_event(self):
        from tools.interrupt import set_interrupt, is_interrupted

        set_interrupt(True)

        async def observe():
            await asyncio.sleep(0)
            return is_interrupted()

        assert await asyncio.create_task(observe()) is True
        set_interrupt(False)


# ---------------------------------------------------------------------------
# Unit tests: pre-tool interrupt check
# ---------------------------------------------------------------------------

class TestPreToolCheck:
    """Verify that _execute_tool_calls skips all tools when interrupted."""

    @pytest.mark.asyncio
    async def test_all_tools_skipped_when_interrupted(self):
        """Mock an interrupted agent and verify no tools execute."""
        from unittest.mock import MagicMock

        # Build a fake assistant_message with 3 tool calls
        tc1 = MagicMock()
        tc1.id = "tc_1"
        tc1.function.name = "terminal"
        tc1.function.arguments = '{"command": "rm -rf /"}'

        tc2 = MagicMock()
        tc2.id = "tc_2"
        tc2.function.name = "terminal"
        tc2.function.arguments = '{"command": "echo hello"}'

        tc3 = MagicMock()
        tc3.id = "tc_3"
        tc3.function.name = "todo"
        tc3.function.arguments = '{"todos": []}'

        assistant_msg = MagicMock()
        assistant_msg.tool_calls = [tc1, tc2, tc3]

        messages = []

        # Create a minimal mock agent with _interrupt_requested = True
        agent = MagicMock()
        agent._interrupt_requested = True
        agent.log_prefix = ""
        agent._flush_messages_to_session_db = AsyncMock(return_value=True)
        # PR #72425: execute_tool_calls_* read _incremental_persistence_failed
        # via getattr at loop top. A bare MagicMock auto-creates a truthy value
        # for any attribute access, which would short-circuit the interrupt
        # skip path before any cancelled-tool messages are appended.
        agent._incremental_persistence_failed = False

        # Import and call the method
        from run_agent import AIAgent
        await AIAgent._execute_tool_calls(agent, assistant_msg, messages, "default")

        # All 3 should be skipped
        assert len(messages) == 3
        for msg in messages:
            assert msg["role"] == "tool"
            assert "cancelled" in msg["content"].lower() or "interrupted" in msg["content"].lower()

        # No actual tool handlers should have been called
        # (handle_function_call should NOT have been invoked)
