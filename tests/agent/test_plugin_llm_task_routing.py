"""Native-async port of upstream plugin auxiliary-task routing tests.

The upstream source tests both sync and async facades. This distribution keeps
the retained plugin facade coroutine-only, so the assertions are unchanged in
meaning and every completion is awaited at the public boundary.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from agent.plugin_llm import (
    PluginLlmTextInput,
    PluginLlmTrustError,
    _TrustPolicy,
    _check_task,
    _resolve_task_ownership,
    make_plugin_llm_for_test,
)


def _fake_response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


def _policy(plugin_id: str = "my-plugin", *, allow_task_override: bool = False):
    return _TrustPolicy(
        plugin_id=plugin_id,
        allow_provider_override=True,
        allow_model_override=True,
        allow_agent_id_override=True,
        allow_profile_override=True,
        allow_task_override=allow_task_override,
    )


def _set_registry(monkeypatch, entries: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_auxiliary_tasks", lambda: list(entries)
    )


def _set_builtins(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setattr(
        "agent.plugin_llm._builtin_auxiliary_task_keys", lambda: frozenset(keys)
    )


def _capturing_caller(captured: dict[str, Any]):
    async def caller(**kwargs: Any):
        captured.update(kwargs)
        if kwargs.get("task"):
            return "aux-provider", "aux-model", _fake_response()
        return "main-provider", "main-model", _fake_response()

    return caller


class TestCheckTask:
    def test_none_auto_and_blank_are_default_route(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, [])
        for value in (None, "", "   ", "auto", " AUTO "):
            assert _check_task(
                _policy(), plugin_id="my-plugin", requested_task=value
            ) is None

    def test_own_registered_key_allowed_and_stripped(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        assert (
            _check_task(
                _policy(), plugin_id="my-plugin", requested_task=" classifier "
            )
            == "classifier"
        )

    def test_foreign_key_rejected_and_named(self, monkeypatch, caplog):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "other-plugin"}])
        _set_builtins(monkeypatch, [])
        with caplog.at_level(logging.WARNING):
            with pytest.raises(PluginLlmTrustError, match="classifier"):
                _check_task(
                    _policy(), plugin_id="my-plugin", requested_task="classifier"
                )
        assert any("my-plugin" in record.getMessage() for record in caplog.records)

    def test_unknown_key_rejected(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, ["vision"])
        with pytest.raises(PluginLlmTrustError):
            _check_task(_policy(), plugin_id="my-plugin", requested_task="nope")

    def test_builtin_key_requires_explicit_trust(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, ["vision"])
        with pytest.raises(PluginLlmTrustError, match="allow_task_override"):
            _check_task(_policy(), plugin_id="my-plugin", requested_task="vision")
        assert (
            _check_task(
                _policy(allow_task_override=True),
                plugin_id="my-plugin",
                requested_task="vision",
            )
            == "vision"
        )


@pytest.mark.asyncio
async def test_task_routes_complete_and_reports_slot(monkeypatch):
    _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
    _set_builtins(monkeypatch, [])
    captured: dict[str, Any] = {}
    llm = make_plugin_llm_for_test(
        plugin_id="my-plugin",
        policy=_policy(),
        async_caller=_capturing_caller(captured),
    )

    result = await llm.complete(
        [{"role": "user", "content": "hi"}], task="classifier"
    )
    assert captured["task"] == "classifier"
    assert (result.provider, result.model) == ("aux-provider", "aux-model")
    assert result.audit["task"] == "classifier"


@pytest.mark.asyncio
async def test_default_complete_keeps_main_route(monkeypatch):
    _set_registry(monkeypatch, [])
    _set_builtins(monkeypatch, [])
    captured: dict[str, Any] = {}
    llm = make_plugin_llm_for_test(
        plugin_id="my-plugin",
        policy=_policy(),
        async_caller=_capturing_caller(captured),
    )
    result = await llm.complete([{"role": "user", "content": "hi"}])
    assert captured["task"] is None
    assert (result.provider, result.model) == ("main-provider", "main-model")
    assert result.audit["task"] == ""


@pytest.mark.asyncio
async def test_rejected_task_never_invokes_caller(monkeypatch):
    _set_registry(monkeypatch, [{"key": "classifier", "plugin": "other-plugin"}])
    _set_builtins(monkeypatch, [])
    captured: dict[str, Any] = {}
    llm = make_plugin_llm_for_test(
        plugin_id="my-plugin",
        policy=_policy(),
        async_caller=_capturing_caller(captured),
    )
    with pytest.raises(PluginLlmTrustError):
        await llm.complete([{"role": "user", "content": "hi"}], task="classifier")
    assert captured == {}


@pytest.mark.asyncio
async def test_structured_task_routes(monkeypatch):
    _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
    _set_builtins(monkeypatch, [])
    captured: dict[str, Any] = {}
    llm = make_plugin_llm_for_test(
        plugin_id="my-plugin",
        policy=_policy(),
        async_caller=_capturing_caller(captured),
    )
    result = await llm.complete_structured(
        instructions="classify this",
        input=[PluginLlmTextInput(text="payload")],
        task="classifier",
    )
    assert captured["task"] == "classifier"
    assert result.audit["task"] == "classifier"


@pytest.mark.asyncio
async def test_task_route_is_logged(monkeypatch, caplog):
    _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
    _set_builtins(monkeypatch, [])
    llm = make_plugin_llm_for_test(
        plugin_id="my-plugin",
        policy=_policy(),
        async_caller=_capturing_caller({}),
    )
    with caplog.at_level(logging.INFO, logger="agent.plugin_llm"):
        await llm.complete(
            [{"role": "user", "content": "hi"}],
            task="classifier",
            purpose="plain",
        )
    assert any(
        "plugin_llm.complete" in record.getMessage()
        and "aux-provider" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_production_invoke_forwards_task(monkeypatch):
    _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
    _set_builtins(monkeypatch, [])
    seen: dict[str, Any] = {}

    async def fake_call_llm(**kwargs: Any):
        seen.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    llm = make_plugin_llm_for_test(plugin_id="my-plugin", policy=_policy())
    await llm.complete([{"role": "user", "content": "hi"}], task="classifier")
    assert seen["task"] == "classifier"


def test_ownership_uses_canonical_plugin_key(monkeypatch):
    _set_registry(
        monkeypatch,
        [{"key": "classifier", "plugin": "my_key"}],
    )
    _set_builtins(monkeypatch, [])
    owned, builtin = _resolve_task_ownership("my_key")
    assert owned == {"classifier"}
    assert builtin == set()
