"""Async parity tests for the Actual Computer provider wiring."""

from __future__ import annotations

import io
import json
import urllib.request
import urllib.response
from email.message import Message

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from agent.transports.codex import ResponsesApiTransport
from hermes_cli.auth import (
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
    DEFAULT_ACTUAL_BASE_URL,
    DEFAULT_ACTUAL_LOCAL_BASE_URL,
    get_api_key_provider_status,
    normalize_actual_base_url,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.providers import determine_api_mode, get_label
from providers import get_provider_profile

pytestmark = pytest.mark.asyncio


def _clear_actual_env(monkeypatch):
    monkeypatch.delenv("ACTUAL_API_KEY", raising=False)
    monkeypatch.delenv("ACTUAL_BASE_URL", raising=False)


async def test_actual_aliases_and_profile_metadata():
    profile = await get_provider_profile("actual-computer")

    assert profile is not None
    assert profile.name == "actual"
    assert profile.display_name == "Actual Computer"
    assert profile.base_url == DEFAULT_ACTUAL_BASE_URL
    assert profile.api_mode == "codex_responses"
    assert profile.auth_type == "api_key"
    assert profile.env_vars == ("ACTUAL_API_KEY", "ACTUAL_BASE_URL")
    assert await resolve_provider("actual-computer") == "actual"
    assert get_label("actual") == "Actual Computer"
    assert determine_api_mode("actual", DEFAULT_ACTUAL_BASE_URL) == "codex_responses"


async def test_actual_base_url_normalization():
    assert normalize_actual_base_url("https://api.actual.inc") == DEFAULT_ACTUAL_BASE_URL
    assert normalize_actual_base_url("https://api.actual.inc/v1") == DEFAULT_ACTUAL_BASE_URL
    assert normalize_actual_base_url("http://127.0.0.1:8080") == DEFAULT_ACTUAL_LOCAL_BASE_URL
    assert normalize_actual_base_url("http://localhost:8080/") == (
        "http://localhost:8080/v1"
    )


async def test_actual_credentials_default_to_hosted_api(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")

    creds = await resolve_api_key_provider_credentials("actual")

    assert creds == {
        "provider": "actual",
        "api_key": "actual-test-key",
        "base_url": DEFAULT_ACTUAL_BASE_URL,
        "source": "ACTUAL_API_KEY",
    }


async def test_actual_local_loopback_allows_no_auth(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")

    creds = await resolve_api_key_provider_credentials("actual")
    status = await get_api_key_provider_status("actual")

    assert creds["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert creds["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL
    assert creds["source"] == "local-offline"
    assert status["configured"] is True
    assert status["logged_in"] is True
    assert status["key_source"] == "local-offline"
    assert status["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


async def test_actual_scoped_base_url_does_not_read_foreign_process_env(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "https://foreign.example/v1")
    token = set_secret_scope({"ACTUAL_BASE_URL": "http://127.0.0.1:8080"})
    try:
        creds = await resolve_api_key_provider_credentials("actual")
        assert creds["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL
    finally:
        reset_secret_scope(token)


async def test_actual_profile_fetch_models_uses_native_async_catalog(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    profile = await get_provider_profile("actual")
    captured: dict[str, object] = {}

    async def open_catalog(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return urllib.response.addinfourl(
            io.BytesIO(json.dumps({"data": [{"id": "actual/local-model"}]}).encode()),
            Message(),
            request.full_url,
            200,
        )

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", open_catalog)
    assert await profile.fetch_models(timeout=1.5) == ["actual/local-model"]

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == DEFAULT_ACTUAL_LOCAL_BASE_URL + "/models"
    assert "authorization" not in {
        name.lower() for name, _value in request.header_items()
    }
    assert captured["timeout"] == 1.5


async def test_actual_codex_transport_clamps_reasoning_effort():
    transport = ResponsesApiTransport()
    for requested, expected in (("xhigh", "high"), ("ultra", "max"), ("high", "high")):
        kwargs = transport.build_kwargs(
            model="actual/local-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            provider="actual",
            reasoning_config={"effort": requested},
            base_url=DEFAULT_ACTUAL_BASE_URL,
        )
        assert kwargs["reasoning"]["effort"] == expected
