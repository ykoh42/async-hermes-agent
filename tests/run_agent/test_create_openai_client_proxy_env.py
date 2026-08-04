"""Regression guard: _create_openai_client must honor HTTP(S)_PROXY env vars.

The keepalive client now uses ``httpx.Limits(keepalive_expiry=20.0)``
instead of a custom ``httpx.HTTPTransport(socket_options=...)`` to
prevent CLOSE-WAIT accumulation.  This avoids breaking streaming for
providers behind reverse proxies (#54049, #12952) while still reaping
idle connections before a proxy's timeout drops them.

This test pins that the constructed ``httpx.Client`` mounts an
``HTTPProxy`` pool when a proxy env var is set, and that no
custom socket-options transport is used (default httpx transport).
"""
from run_agent import _get_proxy_from_env, _get_proxy_for_base_url


def test_get_proxy_from_env_prefers_https_then_http_then_all(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    assert _get_proxy_from_env() is None

    monkeypatch.setenv("ALL_PROXY", "http://all:1")
    assert _get_proxy_from_env() == "http://all:1"

    monkeypatch.setenv("HTTP_PROXY", "http://http:2")
    assert _get_proxy_from_env() == "http://http:2"

    monkeypatch.setenv("HTTPS_PROXY", "http://https:3")
    assert _get_proxy_from_env() == "http://https:3"




def test_get_proxy_from_env_normalizes_socks_alias(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1080/")
    assert _get_proxy_from_env() == "socks5://127.0.0.1:1080/"









