"""Async terminal-policy tests for user-defined deny rules."""

from unittest.mock import AsyncMock

import pytest

from tools.approval import check_all_command_guards


def set_approval_config(monkeypatch, approvals):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value={"approvals": approvals}),
    )


@pytest.mark.asyncio
async def test_matching_deny_rule_is_blocked(monkeypatch):
    set_approval_config(monkeypatch, {"deny": ["git push *"]})

    result = await check_all_command_guards("git push origin main", "local")

    assert result["approved"] is False
    assert result["user_deny"] is True
    assert "Do NOT retry" in result["message"]


@pytest.mark.asyncio
async def test_non_matching_command_is_allowed(monkeypatch):
    set_approval_config(monkeypatch, {"deny": ["git push *"]})

    result = await check_all_command_guards("git status", "local")

    assert result == {"approved": True, "message": None}


@pytest.mark.asyncio
async def test_hardline_policy_precedes_user_deny(monkeypatch):
    set_approval_config(monkeypatch, {"deny": ["rm *"]})

    result = await check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    assert result["hardline"] is True
    assert "user_deny" not in result


@pytest.mark.asyncio
async def test_empty_deny_list_allows_benign_command(monkeypatch):
    set_approval_config(monkeypatch, {"deny": []})

    result = await check_all_command_guards("printf hello", "local")

    assert result["approved"] is True
