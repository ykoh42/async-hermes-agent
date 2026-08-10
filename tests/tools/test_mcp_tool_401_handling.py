"""Tests for MCP tool-handler auth-failure detection.

When a tool call raises UnauthorizedError / OAuthNonInteractiveError /
httpx.HTTPStatusError(401), the handler should:
  1. Ask MCPOAuthManager.handle_401 if recovery is viable.
  2. If yes, trigger MCPServerTask._reconnect_event and retry once.
  3. If no, return a structured needs_reauth error so the model stops
     hallucinating manual refresh attempts.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


def test_is_auth_error_detects_oauth_flow_error():
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    assert _is_auth_error(OAuthFlowError("expired")) is True


@pytest.mark.asyncio
async def test_auth_recovery_retries_even_when_reconnect_wait_times_out(
    monkeypatch,
):
    """Match upstream: successful token recovery always earns one retry."""
    from tools import mcp_tool
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.client.auth import OAuthFlowError

    reset_manager_for_tests()
    manager = get_manager()
    monkeypatch.setattr(manager, "handle_401", AsyncMock(return_value=True))
    monkeypatch.setattr(
        mcp_tool,
        "_await_native_mcp_reconnect",
        AsyncMock(return_value=False),
    )
    server = MagicMock()
    mcp_tool._servers["srv"] = server
    retry = AsyncMock(return_value=json.dumps({"result": "recovered"}))

    try:
        result = await mcp_tool._handle_auth_error_and_retry(
            "srv", OAuthFlowError("expired"), retry, "tools/call tool1"
        )
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)
        reset_manager_for_tests()

    assert json.loads(result) == {"result": "recovered"}
    retry.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_unrecoverable_auth_error_preserves_upstream_message(monkeypatch):
    from tools import mcp_tool
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.client.auth import OAuthFlowError

    reset_manager_for_tests()
    manager = get_manager()
    monkeypatch.setattr(manager, "handle_401", AsyncMock(return_value=False))

    try:
        result = await mcp_tool._handle_auth_error_and_retry(
            "srv",
            OAuthFlowError("expired"),
            AsyncMock(),
            "tools/call tool1",
        )
    finally:
        mcp_tool._server_error_counts.pop("srv", None)
        reset_manager_for_tests()

    assert json.loads(result) == {
        "error": (
            "MCP server 'srv' requires re-authentication. "
            "Run `hermes mcp login srv` (or delete the tokens file under "
            "~/.hermes/mcp-tokens/ and restart). Do NOT retry this tool — "
            "ask the user to re-authenticate."
        ),
        "needs_reauth": True,
        "server": "srv",
    }


@pytest.mark.parametrize(
    ("factory_name", "session_method", "arguments", "operation"),
    [
        ("_make_list_resources_handler", "list_resources", {}, "resources/list"),
        (
            "_make_read_resource_handler",
            "read_resource",
            {"uri": "file://fact"},
            "resources/read",
        ),
        ("_make_list_prompts_handler", "list_prompts", {}, "prompts/list"),
        (
            "_make_get_prompt_handler",
            "get_prompt",
            {"name": "summarize"},
            "prompts/get",
        ),
    ],
)
@pytest.mark.asyncio
async def test_utility_handlers_route_auth_errors_through_recovery(
    monkeypatch,
    factory_name,
    session_method,
    arguments,
    operation,
):
    from tools import mcp_tool

    async def raise_auth(*args, **kwargs):
        raise RuntimeError("auth failure")

    session = SimpleNamespace(**{session_method: raise_auth})
    server = MagicMock()
    server.name = "srv"
    server.session = session
    server._rpc_lock = asyncio.Lock()
    server._is_recycled_stdio.return_value = False
    mcp_tool._servers["srv"] = server

    recovered_payload = json.dumps({"recovered": operation})
    auth_recovery = AsyncMock(return_value=recovered_payload)
    session_recovery = AsyncMock(
        side_effect=AssertionError("auth recovery must run before session recovery")
    )
    monkeypatch.setattr(
        mcp_tool, "_handle_auth_error_and_retry", auth_recovery
    )
    monkeypatch.setattr(
        mcp_tool, "_handle_session_expired_and_retry", session_recovery
    )

    try:
        handler = getattr(mcp_tool, factory_name)("srv", 10.0)
        result = await handler(arguments)
    finally:
        mcp_tool._servers.pop("srv", None)

    assert result == recovered_payload
    assert auth_recovery.await_args.args[0] == "srv"
    assert auth_recovery.await_args.args[3] == operation
    session_recovery.assert_not_awaited()
