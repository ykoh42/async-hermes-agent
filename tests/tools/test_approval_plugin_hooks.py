"""Native-async parity tests for Hermes approval observer hooks."""

import pytest

import model_tools
import tools.approval as approval


@pytest.fixture(autouse=True)
def _isolated_approval_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    approval._permanent_approved.clear()
    approval._session_approved.clear()
    yield
    approval._permanent_approved.clear()
    approval._session_approved.clear()


@pytest.mark.asyncio
async def test_manual_approval_fires_correlated_pre_and_post_hooks(monkeypatch):
    events = []

    async def load_config_readonly():
        return {"approvals": {"mode": "manual"}}

    async def invoke_hook(name, **kwargs):
        events.append((name, kwargs))
        return []

    async def approve_once(*_args, **_kwargs):
        return "once"

    async def dispatch(*_args, **_kwargs):
        return await approval.check_all_command_guards(
            "rm -rf build",
            "local",
            approval_callback=approve_once,
        )

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)

    result = await model_tools.handle_function_call(
        "terminal",
        {"command": "rm -rf build"},
        tool_call_id="call-approval",
        turn_id="turn-approval",
        skip_pre_tool_call_hook=True,
        skip_tool_execution_middleware=True,
    )

    assert result["approved"] is True
    assert [name for name, _ in events] == [
        "pre_approval_request",
        "post_approval_response",
    ]
    for _, payload in events:
        assert payload["turn_id"] == "turn-approval"
        assert payload["tool_call_id"] == "call-approval"
        assert payload["surface"] == "cli"
    assert events[-1][1]["choice"] == "once"


@pytest.mark.asyncio
async def test_smart_approval_fires_redacted_observer_hooks(monkeypatch):
    events = []

    async def load_config_readonly():
        return {"approvals": {"mode": "smart"}}

    async def invoke_hook(name, **kwargs):
        events.append((name, kwargs))
        return []

    async def approve(*_args):
        return "approve"

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    monkeypatch.setattr(approval, "_smart_approve", approve)

    result = await approval.check_all_command_guards(
        "rm -rf build",
        "local",
    )

    assert result["approved"] is True
    assert [name for name, _ in events] == [
        "pre_approval_request",
        "post_approval_response",
    ]
    assert events[0][1]["surface"] == "smart"
    assert events[1][1]["choice"] == "smart_approve"
    assert events[1][1]["decided_by"] == "aux_llm"


@pytest.mark.asyncio
async def test_observer_failure_does_not_change_approval(monkeypatch):
    async def load_config_readonly():
        return {"approvals": {"mode": "manual"}}

    async def invoke_hook(*_args, **_kwargs):
        raise RuntimeError("observer failed")

    async def approve_once(*_args, **_kwargs):
        return "once"

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)

    result = await approval.check_all_command_guards(
        "rm -rf build",
        "local",
        approval_callback=approve_once,
    )

    assert result["approved"] is True
