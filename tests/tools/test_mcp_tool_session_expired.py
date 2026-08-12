"""Tests for MCP tool-handler transport-session auto-reconnect.

When a Streamable HTTP MCP server garbage-collects its server-side
session (idle TTL, server restart, pod rotation, …) it rejects
subsequent requests with a JSON-RPC error containing phrases like
``"Invalid or expired session"``.  The OAuth token remains valid —
only the transport session state needs rebuilding.

Before the #13383 fix, this class of failure fell through as a plain
tool error with no recovery path, so every subsequent call on the
affected MCP server failed until the gateway was manually restarted.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _is_session_expired_error — unit coverage
# ---------------------------------------------------------------------------


def test_is_session_expired_detects_invalid_or_expired_session():
    """Reporter's exact wpcom-mcp error message (#13383)."""
    from tools.mcp_tool import _is_session_expired_error
    exc = RuntimeError("Invalid params: Invalid or expired session")
    assert _is_session_expired_error(exc) is True


def test_is_session_expired_detects_expired_session_variant():
    """Generic ``session expired`` / ``expired session`` phrasings used
    by other SDK servers."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("Session expired")) is True
    assert _is_session_expired_error(RuntimeError("expired session: abc")) is True


def test_is_session_expired_detects_session_not_found():
    """Server-side GC produces ``session not found`` / ``unknown session``
    on some implementations."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("session not found")) is True
    assert _is_session_expired_error(RuntimeError("Unknown session: abc123")) is True


def test_is_session_expired_traversal_is_budget_bounded():
    """Pathologically long chains stop at the node budget without spinning."""
    import tools.mcp_tool as mcp_mod
    from tools.mcp_tool import _is_session_expired_error

    exc: BaseException = RuntimeError("leaf")
    for i in range(mcp_mod._EXC_TRAVERSAL_MAX_NODES * 2):
        wrapper = RuntimeError(f"layer {i}")
        wrapper.__cause__ = exc
        exc = wrapper

    # Terminates quickly and classifies false (no transport signal within
    # budget). The exact outcome past the budget is unspecified; the
    # invariant under test is termination.
    assert _is_session_expired_error(exc) is False


# ---------------------------------------------------------------------------
# Handler integration — verify the recovery plumbing wires end-to-end
# ---------------------------------------------------------------------------


def _install_stub_server(mcp_tool, name):
    server = MagicMock()
    server.name = name
    server._rpc_lock = asyncio.Lock()
    server._is_recycled_stdio.return_value = False
    server.session = SimpleNamespace()
    mcp_tool._servers[name] = server
    mcp_tool._server_error_counts.pop(name, None)
    return server


@pytest.mark.asyncio
async def test_session_expired_reconnect_waits_for_distinct_session():
    from tools.mcp_tool import _await_native_mcp_reconnect

    old_session = object()
    new_session = object()
    server = SimpleNamespace(
        session=old_session,
        _ready=asyncio.Event(),
    )
    server._ready.set()
    replacement_task = None

    class ReconnectEvent:
        def set(self):
            nonlocal replacement_task

            async def replace_session():
                # A reconnect lifecycle can publish readiness before its stale
                # session reference has been swapped out. The waiter must not
                # retry against that old object.
                server._ready.set()
                await asyncio.sleep(0.01)
                server.session = new_session

            replacement_task = asyncio.create_task(replace_session())

    server._reconnect_event = ReconnectEvent()

    assert await _await_native_mcp_reconnect(
        "srv",
        server,
        operation_description="tools/call health",
        timeout=0.5,
    ) is True
    assert server.session is new_session
    assert replacement_task is not None
    await replacement_task



# ---------------------------------------------------------------------------
# Parallel coverage for resources/list, resources/read, prompts/list,
# prompts/get — all four handlers share the same exception path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_factory, handler_kwargs, session_method, op_label",
    [
        ("_make_list_resources_handler", {"tool_timeout": 10.0}, "list_resources", "list_resources"),
        ("_make_read_resource_handler", {"tool_timeout": 10.0}, "read_resource", "read_resource"),
        ("_make_list_prompts_handler", {"tool_timeout": 10.0}, "list_prompts", "list_prompts"),
        ("_make_get_prompt_handler", {"tool_timeout": 10.0}, "get_prompt", "get_prompt"),
    ],
)
@pytest.mark.asyncio
async def test_non_tool_handlers_also_reconnect_on_session_expired(
    monkeypatch, tmp_path, handler_factory, handler_kwargs, session_method, op_label
):
    """All four non-``tools/call`` MCP handlers share the recovery
    pattern and must reconnect the same way on session-expired."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    server_name = f"srv-{op_label}"
    server = _install_stub_server(mcp_tool, server_name)

    call_count = {"n": 0}

    async def _sequence(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Invalid or expired session")
        # Return something with the shapes each handler expects.
        # Explicitly set primitive attrs — MagicMock's default auto-attr
        # behaviour surfaces ``MagicMock`` values for optional fields
        # like ``description``, which break ``json.dumps`` downstream.
        return SimpleNamespace(
            resources=[],
            prompts=[],
            contents=[],
            messages=[],
            description=None,
            nextCursor=None,
        )

    setattr(server.session, session_method, _sequence)

    async def reconnect(*_args, **_kwargs):
        return True

    monkeypatch.setattr(mcp_tool, "_await_native_mcp_reconnect", reconnect)
    factory = getattr(mcp_tool, handler_factory)
    # list_resources / list_prompts take (server_name, timeout).
    # read_resource / get_prompt take the same signature.
    try:
        handler = factory(server_name, **handler_kwargs)
        if op_label == "read_resource":
            out = await handler({"uri": "file://foo"})
        elif op_label == "get_prompt":
            out = await handler({"name": "p1"})
        else:
            out = await handler({})
        parsed = json.loads(out)
        assert "error" not in parsed, (
            f"{op_label}: expected retry success, got {parsed}"
        )
        assert call_count["n"] == 2, (
            f"{op_label}: expected 1 original + 1 retry"
        )
    finally:
        mcp_tool._servers.pop(server_name, None)
        mcp_tool._server_error_counts.pop(server_name, None)
