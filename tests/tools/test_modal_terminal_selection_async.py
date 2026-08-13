"""Terminal selects direct versus managed Modal at an awaited boundary."""

from __future__ import annotations

import pytest

from tools import terminal_tool as terminal


@pytest.mark.asyncio
async def test_get_modal_backend_state_prefers_ready_managed(monkeypatch):
    async def yes(*_args, **_kwargs):  # noqa: ASYNC124 - coroutine test double
        return True

    monkeypatch.setattr(
        "tools.tool_backend_helpers.has_direct_modal_credentials", yes
    )
    monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", yes)
    monkeypatch.setattr(
        "tools.managed_tool_gateway.is_managed_tool_gateway_ready", yes
    )

    state = await terminal._get_modal_backend_state("auto")

    assert state["selected_backend"] == "managed"
