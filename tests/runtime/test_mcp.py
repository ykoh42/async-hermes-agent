"""MCP tool dispatch contract tests."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools import mcp_tool


class _Session:
    async def call_tool(self, _name, arguments):
        return SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="text", text=f"ok:{arguments['value']}")],
            structuredContent=None,
        )


class _Server:
    def __init__(self):
        self.session = _Session()
        self._rpc_lock = asyncio.Lock()
        self.tool_timeout = 5

    def _is_recycled_stdio(self):
        return False

    def mark_tool_call(self):
        pass

@pytest.mark.asyncio
async def test_mcp_tool_dispatch_does_not_block_or_leak(monkeypatch):
    server = _Server()
    monkeypatch.setitem(mcp_tool._servers, "srv", server)
    monkeypatch.setattr(mcp_tool, "_server_error_counts", {})
    monkeypatch.setattr(mcp_tool, "_server_breaker_opened_at", {})

    handler = mcp_tool._make_tool_handler("srv", "echo", 5)

    assert inspect.iscoroutinefunction(handler)
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            result = await handler({"value": "hello"})
        finally:
            blockbuster.deactivate()

    assert result == '{"result": "ok:hello"}'


@pytest.mark.asyncio
async def test_mcp_lifecycle_stops_after_last_agent_releases(monkeypatch):
    class Owner:
        pass

    first = Owner()
    second = Owner()
    shutdown_calls = 0

    async def shutdown():
        nonlocal shutdown_calls
        shutdown_calls += 1

    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", shutdown)
    mcp_tool._mcp_lifecycle_consumers.clear()
    try:
        await mcp_tool.retain_mcp_lifecycle(first)
        await mcp_tool.retain_mcp_lifecycle(second)

        await mcp_tool.release_mcp_lifecycle(first)
        assert shutdown_calls == 0

        await mcp_tool.release_mcp_lifecycle(second)
        assert shutdown_calls == 1

        await mcp_tool.release_mcp_lifecycle(second)
        assert shutdown_calls == 1
    finally:
        mcp_tool._mcp_lifecycle_consumers.clear()
