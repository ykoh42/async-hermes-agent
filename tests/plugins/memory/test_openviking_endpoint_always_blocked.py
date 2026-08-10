"""OpenViking endpoint always-blocked floor."""

import pytest

from plugins.memory.openviking import (
    _OpenVikingEndpointError,
    _local_openviking_bind,
    _normalize_openviking_url,
    _openviking_endpoint_is_always_blocked,
)


pytestmark = pytest.mark.asyncio


async def test_openviking_blocks_metadata_endpoint():
    with pytest.raises(_OpenVikingEndpointError, match="blocked metadata address"):
        await _normalize_openviking_url("http://169.254.169.254/")


async def test_openviking_keeps_default_loopback():
    assert await _normalize_openviking_url("http://127.0.0.1:1933") == "http://127.0.0.1:1933"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
async def test_openviking_bare_loopback_health_and_autostart_use_same_default_port(host):
    endpoint = await _normalize_openviking_url(host)

    assert endpoint == f"http://{host}:1933"
    assert await _local_openviking_bind(endpoint) == (host, 1933)


async def test_openviking_explicit_loopback_url_preserves_implicit_http_port():
    assert await _normalize_openviking_url("http://localhost") == "http://localhost"


async def test_openviking_blocks_ecs_metadata_hostname():
    with pytest.raises(_OpenVikingEndpointError, match="blocked metadata address"):
        await _normalize_openviking_url("http://metadata.google.internal/computeMetadata/v1/")


async def test_openviking_rejects_endpoint_credentials_and_query():
    with pytest.raises(_OpenVikingEndpointError, match="cannot contain user info"):
        await _normalize_openviking_url("https://user:secret@example.com?api_key=secret")


async def test_openviking_validates_shorthand_ipv6_port():
    assert await _normalize_openviking_url("::1:1934") == "http://[::1]:1934"
    with pytest.raises(_OpenVikingEndpointError, match="Port could not be cast"):
        await _normalize_openviking_url("::1:not-a-port")


async def test_openviking_caches_safety_check_for_unchanged_endpoint(monkeypatch):
    import tools.url_safety as url_safety

    calls = []
    _openviking_endpoint_is_always_blocked.cache_clear()
    async def allow_url(value):
        calls.append(value)
        return False

    monkeypatch.setattr(url_safety, "is_always_blocked_url", allow_url)

    assert await _normalize_openviking_url("https://openviking.example.test") == (
        "https://openviking.example.test"
    )
    assert await _normalize_openviking_url("https://openviking.example.test") == (
        "https://openviking.example.test"
    )
    assert calls == ["https://openviking.example.test"]
    _openviking_endpoint_is_always_blocked.cache_clear()
