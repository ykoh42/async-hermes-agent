"""Coroutine-native MCP tool dispatch tests."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

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
async def test_mcp_tool_uses_async_cross_loop_bridge(monkeypatch):
    server = _Server()
    monkeypatch.setitem(mcp_tool._servers, "srv", server)
    monkeypatch.setattr(mcp_tool, "_server_error_counts", {})
    monkeypatch.setattr(mcp_tool, "_server_breaker_opened_at", {})

    async def run_inline(coro_or_factory, timeout=30):
        result = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return await result

    monkeypatch.setattr(mcp_tool, "_await_mcp_operation", run_inline)
    handler = mcp_tool._make_tool_handler("srv", "echo", 5)

    assert inspect.iscoroutinefunction(handler)
    result = await handler({"value": "hello"})

    assert result == '{"result": "ok:hello"}'
