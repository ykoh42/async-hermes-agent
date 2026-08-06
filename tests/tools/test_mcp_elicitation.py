"""Async-library MCP elicitation contracts."""

import asyncio
import contextvars
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp.types")

from mcp.types import ElicitResult

from tools.mcp_tool import ElicitationHandler, _format_elicitation_schema_summary


def test_schema_summary_describes_requested_fields() -> None:
    summary = _format_elicitation_schema_summary(
        {
            "type": "object",
            "properties": {
                "amount": {"type": "string", "description": "USD amount"},
            },
        },
        "pay",
    )
    assert "amount (string): USD amount" in summary


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["form", "url"])
async def test_elicitation_declines_without_native_human_approval_surface(mode) -> None:
    handler = ElicitationHandler("pay", {"timeout": 5})
    params = SimpleNamespace(mode=mode, message="confirm", requested_schema={})

    result = await handler(context=None, params=params)

    assert isinstance(result, ElicitResult)
    assert result.action == "decline"
    assert handler.metrics == {
        "requests": 1,
        "accepted": 0,
        "declined": 1,
        "errors": 0,
    }


@pytest.mark.asyncio
async def test_form_elicitation_awaits_native_clarify_callback() -> None:
    calls = []

    async def clarify(question, choices):
        calls.append((question, choices))
        await asyncio.sleep(0)
        return "Approve"

    owner = SimpleNamespace(
        _pending_call_context=contextvars.copy_context(),
        _pending_elicitation_callback=clarify,
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)
    params = SimpleNamespace(
        mode="form",
        message="Authorize payment",
        requested_schema={
            "properties": {
                "amount": {"type": "string", "description": "USD amount"}
            }
        },
    )

    result = await handler(context=None, params=params)

    assert result.action == "accept"
    assert result.content == {}
    assert calls == [
        (
            "Authorize payment\n\nFields requested by MCP server 'pay':\n"
            "  - amount (string): USD amount",
            ["Approve", "Decline"],
        )
    ]
    assert handler.metrics["accepted"] == 1


@pytest.mark.asyncio
async def test_form_elicitation_replays_call_context() -> None:
    probe = contextvars.ContextVar("mcp_elicitation_probe", default="")
    seen = []

    async def clarify(_question, _choices):
        seen.append(probe.get())
        return "Decline"

    token = probe.set("agent-turn")
    try:
        captured = contextvars.copy_context()
    finally:
        probe.reset(token)
    owner = SimpleNamespace(
        _pending_call_context=captured,
        _pending_elicitation_callback=clarify,
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)

    result = await handler(
        context=None,
        params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
    )

    assert result.action == "decline"
    assert seen == ["agent-turn"]


@pytest.mark.asyncio
async def test_form_elicitation_propagates_caller_cancel() -> None:
    async def clarify(_question, _choices):
        return "Cancel"

    owner = SimpleNamespace(
        _pending_call_context=None,
        _pending_elicitation_callback=clarify,
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)

    result = await handler(
        context=None,
        params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
    )

    assert result.action == "cancel"
    assert handler.metrics["errors"] == 1


@pytest.mark.asyncio
async def test_form_elicitation_exception_fails_closed() -> None:
    async def clarify(_question, _choices):
        raise RuntimeError("approval unavailable")

    owner = SimpleNamespace(
        _pending_call_context=None,
        _pending_elicitation_callback=clarify,
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)

    result = await handler(
        context=None,
        params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
    )

    assert result.action == "decline"
    assert handler.metrics["errors"] == 1


@pytest.mark.asyncio
async def test_form_elicitation_timeout_cancels_callback() -> None:
    cancelled = asyncio.Event()

    async def clarify(_question, _choices):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    owner = SimpleNamespace(
        _pending_call_context=None,
        _pending_elicitation_callback=clarify,
    )
    handler = ElicitationHandler("pay", {"timeout": 0.01}, owner=owner)
    handler.timeout = 0.01

    result = await handler(
        context=None,
        params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
    )

    assert result.action == "cancel"
    assert cancelled.is_set()
    assert handler.metrics["errors"] == 1


@pytest.mark.asyncio
async def test_sync_elicitation_callback_fails_closed() -> None:
    owner = SimpleNamespace(
        _pending_call_context=None,
        _pending_elicitation_callback=lambda *_args: "Approve",
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)

    result = await handler(
        context=None,
        params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
    )

    assert result.action == "decline"
    assert handler.metrics["errors"] == 1


def test_session_kwargs_installs_callback() -> None:
    handler = ElicitationHandler("pay", {})
    assert handler.session_kwargs() == {"elicitation_callback": handler}
