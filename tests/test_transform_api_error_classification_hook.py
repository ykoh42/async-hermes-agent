"""Parity tests for the retained API-error transform hook.

The upstream hook is synchronous because ``classify_api_error`` is a pure
classifier.  This fork keeps the same payload and first-valid result contract
while deliberately limiting dispatch to callbacks already loaded by the
awaited native-async plugin boundary.
"""

import logging

import hermes_cli.plugins as plugins_mod
from agent.error_classifier import FailoverReason, classify_api_error


class _FakeAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.body = body or {}


_UNCLAIMED_MESSAGE = "flux capacitor drift detected in shard seven"


def _classify_unclaimed_error(**kwargs):
    return classify_api_error(
        _FakeAPIError(_UNCLAIMED_MESSAGE),
        provider="acmecloud",
        model="acme/large-1",
        **kwargs,
    )


def _fresh_manager(monkeypatch):
    manager = plugins_mod.PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    plugins_mod._ACTIVE_PLUGIN_MANAGER.set(None)
    return manager


def test_no_hook_falls_through_to_builtin(monkeypatch):
    _fresh_manager(monkeypatch)
    result = _classify_unclaimed_error()
    assert result.reason is FailoverReason.unknown
    assert result.retryable is True


def test_transform_wins_and_sanitizes(monkeypatch):
    monkeypatch.setattr(
        plugins_mod,
        "get_plugin_error_classification",
        lambda **_kwargs: {
            "reason": FailoverReason.model_not_found,
            "retryable": False,
            "should_fallback": True,
            "message": "custom guidance",
            "error_context": {"upstream_provider": "AcmeCloud"},
        },
    )
    result = _classify_unclaimed_error()
    assert result.reason is FailoverReason.model_not_found
    assert result.retryable is False
    assert result.should_fallback is True
    assert result.message == "custom guidance"
    assert result.error_context == {"upstream_provider": "AcmeCloud"}


def test_transform_can_override_builtin(monkeypatch):
    monkeypatch.setattr(
        plugins_mod,
        "get_plugin_error_classification",
        lambda **_kwargs: {"reason": FailoverReason.overloaded},
    )
    result = classify_api_error(
        _FakeAPIError("too many requests", status_code=429),
        provider="zai",
    )
    assert result.reason is FailoverReason.overloaded


def test_invalid_transform_is_ignored(monkeypatch):
    monkeypatch.setattr(
        plugins_mod,
        "get_plugin_error_classification",
        lambda **_kwargs: None,
    )
    result = _classify_unclaimed_error()
    assert result.reason is FailoverReason.unknown


def test_helper_exception_never_breaks_classification(monkeypatch):
    def _boom(**_kwargs):
        raise RuntimeError("plugin infrastructure exploded")

    monkeypatch.setattr(plugins_mod, "get_plugin_error_classification", _boom)
    result = _classify_unclaimed_error()
    assert result.reason is FailoverReason.unknown


def test_transform_payload_contains_parsed_error_context(monkeypatch):
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(plugins_mod, "get_plugin_error_classification", _capture)
    _classify_unclaimed_error(approx_tokens=1234, num_messages=7)
    assert seen["provider"] == "acmecloud"
    assert seen["model"] == "acme/large-1"
    assert seen["status_code"] is None
    assert seen["error_type"] == "_FakeAPIError"
    assert "flux capacitor drift" in seen["error_message"]
    assert seen["approx_tokens"] == 1234
    assert seen["num_messages"] == 7
    assert isinstance(seen["error_body"], dict)
    assert isinstance(seen["error"], _FakeAPIError)


def test_loaded_sync_callbacks_first_valid_wins_and_warns(monkeypatch, caplog):
    manager = _fresh_manager(monkeypatch)

    def invalid(**_kwargs):
        return {"reason": "not-a-real-reason"}

    def first(**_kwargs):
        return {"reason": "billing"}

    def second(**_kwargs):
        return {"reason": "rate_limit"}

    manager._hooks["transform_api_error_classification"] = [invalid, first, second]
    with caplog.at_level(logging.WARNING, logger=plugins_mod.logger.name):
        result = _classify_unclaimed_error()
    assert result.reason is FailoverReason.billing
    assert any("skipped 1 valid" in record.getMessage() for record in caplog.records)


def test_loaded_async_callback_is_not_run_from_sync_classifier(monkeypatch):
    manager = _fresh_manager(monkeypatch)
    called = False

    async def callback(**_kwargs):
        nonlocal called
        called = True
        return {"reason": "billing"}

    manager._hooks["transform_api_error_classification"] = [callback]
    result = _classify_unclaimed_error()
    assert result.reason is FailoverReason.unknown
    assert called is False


def test_hook_name_is_registered():
    assert "transform_api_error_classification" in plugins_mod.VALID_HOOKS
