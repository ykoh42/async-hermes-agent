"""Native async parity tests for plugin-driven tool approvals."""

from unittest.mock import AsyncMock, Mock

import pytest

import tools.approval as approval
from tools.approval import request_tool_approval


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_session_yolo", set())
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(
        approval,
        "_load_approval_config_snapshot",
        AsyncMock(),
    )
    monkeypatch.setattr(
        approval,
        "get_current_session_key",
        lambda default="default": "test-session",
    )
    monkeypatch.setattr(approval, "is_approved", lambda session, pattern: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: None,
    )


@pytest.mark.asyncio
async def test_session_cached_approval_short_circuits(monkeypatch):
    monkeypatch.setattr(approval, "is_approved", lambda session, pattern: True)
    prompt = AsyncMock(side_effect=AssertionError("cached approval must not prompt"))
    monkeypatch.setattr(approval, "prompt_dangerous_approval", prompt)

    result = await request_tool_approval(
        "write_file",
        "sensitive path",
        rule_key="ssh",
    )

    assert result == {"approved": True, "message": None}
    prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_approve_once(monkeypatch):
    callback = AsyncMock(return_value="once")
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: callback,
    )

    result = await request_tool_approval(
        "write_file",
        "writing ~/.ssh/authorized_keys",
    )

    assert result == {"approved": True, "message": None}
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_deny_blocks(monkeypatch):
    callback = AsyncMock(return_value="deny")
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: callback,
    )

    result = await request_tool_approval("terminal", "curl PUT to external API")

    assert result["approved"] is False
    assert "denied" in result["message"].lower()
    assert result["pattern_key"].startswith("plugin_rule:")


@pytest.mark.asyncio
async def test_session_choice_persists_session_only(monkeypatch):
    callback = AsyncMock(return_value="session")
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: callback,
    )
    approve_session = Mock()
    approve_permanent = Mock()
    monkeypatch.setattr(approval, "approve_session", approve_session)
    monkeypatch.setattr(approval, "approve_permanent", approve_permanent)

    result = await request_tool_approval(
        "write_file",
        "reason",
        rule_key="ssh-writes",
    )

    assert result["approved"] is True
    approve_session.assert_called_once_with(
        "test-session",
        "plugin_rule:ssh-writes",
    )
    approve_permanent.assert_not_called()


@pytest.mark.asyncio
async def test_always_choice_persists_allowlist(monkeypatch):
    callback = AsyncMock(return_value="always")
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: callback,
    )
    save = AsyncMock()
    approve_session = Mock()
    approve_permanent = Mock()
    monkeypatch.setattr(approval, "approve_session", approve_session)
    monkeypatch.setattr(approval, "approve_permanent", approve_permanent)
    monkeypatch.setattr(approval, "save_permanent_allowlist", save)

    result = await request_tool_approval(
        "write_file",
        "reason",
        rule_key="ssh-writes",
    )

    assert result["approved"] is True
    approve_session.assert_called_once_with(
        "test-session",
        "plugin_rule:ssh-writes",
    )
    approve_permanent.assert_called_once_with("plugin_rule:ssh-writes")
    save.assert_awaited_once()


@pytest.mark.asyncio
async def test_distinct_reasons_get_distinct_keys(monkeypatch):
    callback = AsyncMock(return_value="deny")
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: callback,
    )

    first = await request_tool_approval("write_file", "write to ~/.ssh")
    second = await request_tool_approval("write_file", "send an email")

    assert first["pattern_key"] != second["pattern_key"]


@pytest.mark.asyncio
async def test_explicit_rule_key_overrides_derivation(monkeypatch):
    callback = AsyncMock(return_value="deny")
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback",
        lambda: callback,
    )

    result = await request_tool_approval(
        "terminal",
        "any",
        rule_key="my-rule",
    )

    assert result["pattern_key"] == "plugin_rule:my-rule"


@pytest.mark.asyncio
async def test_no_callback_fails_closed():
    result = await request_tool_approval("terminal", "smtp send")

    assert result["approved"] is False
    assert "no native async approval callback" in result["message"].lower()


@pytest.mark.asyncio
async def test_yolo_session_bypasses_gate(monkeypatch):
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
    prompt = AsyncMock(side_effect=AssertionError("yolo must not prompt"))
    monkeypatch.setattr(approval, "prompt_dangerous_approval", prompt)

    result = await request_tool_approval(
        "terminal",
        "curl PUT",
        rule_key="ext",
    )

    assert result == {"approved": True, "message": None}
    prompt.assert_not_awaited()
