"""The async approval boundary reads configuration once without a deepcopy."""

from __future__ import annotations

import pytest

from tools import approval


@pytest.mark.asyncio
async def test_guard_awaits_readonly_config_once(monkeypatch):
    calls = 0

    async def load_config_readonly():
        nonlocal calls
        calls += 1
        return {"approvals": {"deny": ["git push --force*"]}}

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )

    result = await approval.check_all_command_guards(
        "git push --force origin main", "local"
    )

    assert calls == 1
    assert result["approved"] is False
    assert result["user_deny"] is True


@pytest.mark.asyncio
async def test_guard_does_not_mutate_readonly_config(monkeypatch):
    config = {"approvals": {"deny": ["rm secret*"]}}

    async def load_config_readonly():
        return config

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )

    await approval.check_all_command_guards("echo safe", "local")

    assert config == {"approvals": {"deny": ["rm secret*"]}}
