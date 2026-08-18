"""Async-library MCP elicitation contracts."""

import asyncio
import contextvars
import inspect
from unittest.mock import patch
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp.types")

from mcp.types import ElicitResult

from tools.approval import (
    _elicitation_approval_callback,
    request_elicitation_consent,
)
from tools.mcp_tool import ElicitationHandler, _format_elicitation_schema_summary


def _callback_context(callback, context=None):
    captured = context.copy() if context is not None else contextvars.copy_context()
    captured.run(_elicitation_approval_callback.set, callback)
    return captured


def test_public_consent_router_is_native_async() -> None:
    assert inspect.iscoroutinefunction(request_elicitation_consent)


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
        _pending_call_context=_callback_context(clarify),
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
        _pending_call_context=_callback_context(clarify, captured),
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
        _pending_call_context=_callback_context(clarify),
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
        _pending_call_context=_callback_context(clarify),
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)

    result = await handler(
        context=None,
        params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
    )

    assert result.action == "decline"
    assert handler.metrics["declined"] == 1
    assert handler.metrics["errors"] == 0


@pytest.mark.asyncio
async def test_form_elicitation_propagates_task_cancellation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def clarify(_question, _choices):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    owner = SimpleNamespace(
        _pending_call_context=_callback_context(clarify),
    )
    handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)
    task = asyncio.create_task(
        handler(
            context=None,
            params=SimpleNamespace(mode="form", message="Confirm", requested_schema={}),
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_form_elicitation_timeout_cancels_callback() -> None:
    cancelled = asyncio.Event()

    async def clarify(_question, _choices):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    owner = SimpleNamespace(
        _pending_call_context=_callback_context(clarify),
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
        _pending_call_context=_callback_context(lambda *_args: "Approve"),
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


class TestRequestedSchemaFieldName:
    """The requested schema must be read off the *real* SDK model.

    Every other test in this file builds a duck-typed ``SimpleNamespace``
    stand-in for the params object. That keeps them cheap, but it means none
    of them can catch the handler reading a field name the SDK model does not
    actually have -- the stand-in simply has whatever name the test wrote.

    The SDK spells this field ``requestedSchema`` on mcp 1.x and
    ``requested_schema`` on 2.0 (which renamed model fields to snake_case and
    kept camelCase only as a serialization alias, which pydantic does not
    expose to attribute access). Constructing with the camelCase spelling
    works on both -- 2.0 accepts it as the alias -- so this test pins the
    behaviour to the real model on whichever SDK is installed.
    """

    def test_real_sdk_params_schema_reaches_the_consent_description(self):
        from mcp.types import ElicitRequestFormParams

        params = ElicitRequestFormParams(
            message="authorize a payment of $0.50",
            requestedSchema={
                "type": "object",
                "properties": {
                    "card_number": {
                        "type": "string",
                        "description": "card to charge",
                    },
                },
            },
        )
        async def approve(_question, _choices):
            return "decline"

        owner = SimpleNamespace(_pending_call_context=_callback_context(approve))
        handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)
        captured: dict = {}

        async def _capture(*args, **kwargs):
            captured["description"] = kwargs.get("description") or (
                args[1] if len(args) > 1 else ""
            )
            return "decline"

        with patch("tools.approval.request_elicitation_consent", _capture):
            asyncio.run(handler(context=None, params=params))

        # An empty schema renders the generic "Approval requested by ..."
        # fallback, so the field name is what proves the schema was read.
        assert "card_number" in (captured.get("description") or ""), captured
