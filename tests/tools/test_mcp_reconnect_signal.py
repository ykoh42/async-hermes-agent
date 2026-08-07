"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio

import pytest
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction


@pytest.mark.asyncio
async def test_reconnect_event_attribute_exists():
    """MCPServerTask has a _reconnect_event alongside _shutdown_event."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")
    assert hasattr(task, "_reconnect_event")
    assert isinstance(task._reconnect_event, asyncio.Event)
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_shutdown_wins_when_both_set():
    """If both events are set simultaneously, shutdown takes precedence."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.parametrize(
    "wait_method",
    (
        "_wait_for_lifecycle_event",
        "_wait_for_reconnect_or_shutdown",
        "_wait_for_lazy_reconnect",
    ),
)
@pytest.mark.asyncio
async def test_lifecycle_wait_cancellation_propagates_without_task_leaks(wait_method):
    """External cancellation must reap both event waiters before propagating."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("test")
    async with no_task_leaks(action=LeakAction.RAISE):
        waiter = asyncio.create_task(getattr(server, wait_method)())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
