"""Async-library MCP elicitation contracts."""

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


def test_session_kwargs_installs_callback() -> None:
    handler = ElicitationHandler("pay", {})
    assert handler.session_kwargs() == {"elicitation_callback": handler}
