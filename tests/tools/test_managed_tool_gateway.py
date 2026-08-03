"""Contracts for the intentionally disabled managed-gateway boundary."""

from tools.managed_tool_gateway import (
    build_vendor_gateway_url,
    is_managed_tool_gateway_ready,
    peek_nous_access_token,
    read_nous_access_token,
    resolve_managed_tool_gateway,
)


def test_vendor_url_uses_shared_domain(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "gateway.example")
    monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "https")

    assert build_vendor_gateway_url("firecrawl") == (
        "https://firecrawl-gateway.gateway.example"
    )


def test_vendor_url_honors_explicit_override(monkeypatch):
    monkeypatch.setenv(
        "BROWSER_USE_GATEWAY_URL", "http://browser-gateway.local:3009/"
    )

    assert build_vendor_gateway_url("browser-use") == (
        "http://browser-gateway.local:3009"
    )


def test_token_readers_only_expose_explicit_environment_token(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", " token ")

    assert peek_nous_access_token() == "token"
    assert read_nous_access_token() == "token"


def test_managed_gateway_remains_disabled():
    assert resolve_managed_tool_gateway(
        "firecrawl",
        token_reader=lambda: "token",
    ) is None
    assert is_managed_tool_gateway_ready("firecrawl") is False
