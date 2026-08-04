"""Native-async regression coverage for TCP-connect blackhole suppression."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent import model_metadata


@pytest.fixture(autouse=True)
def clear_blackhole_cache():
    model_metadata._endpoint_blackhole_cache.clear()
    model_metadata._endpoint_probe_path_cache.clear()
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()


def test_blackhole_cache_is_shared_by_host_and_port(monkeypatch):
    url = "http://10.0.0.9:30080/v1"
    model_metadata._note_endpoint_blackholed(url)

    assert model_metadata._endpoint_blackholed("http://10.0.0.9:30080/api")
    assert not model_metadata._endpoint_blackholed("http://10.0.0.9:11434")

    seen = model_metadata._endpoint_blackhole_cache["10.0.0.9:30080"]
    monkeypatch.setattr(
        model_metadata.time,
        "monotonic",
        lambda: seen + model_metadata._ENDPOINT_BLACKHOLE_TTL_SECONDS + 1,
    )
    assert not model_metadata._endpoint_blackholed(url)


class AsyncClientStub:
    def __init__(self, error):
        self.get = AsyncMock(side_effect=error)
        self.post = AsyncMock(side_effect=error)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_detect_timeout_aborts_probe_waterfall():
    url = "http://10.0.0.9:30080/v1"
    client = AsyncClientStub(httpx.ConnectTimeout("timed out"))

    with (
        patch("httpx.AsyncClient", return_value=client),
        patch.object(model_metadata, "_local_probe_disk_get", AsyncMock(return_value=None)),
    ):
        assert await model_metadata.detect_local_server_type(url) is None

    assert client.get.await_count == 1
    assert model_metadata._endpoint_blackholed(url)


@pytest.mark.asyncio
async def test_read_timeout_does_not_mark_endpoint_blackholed():
    url = "http://10.0.0.9:30080/v1"
    client = AsyncClientStub(httpx.ReadTimeout("slow"))

    with (
        patch("httpx.AsyncClient", return_value=client),
        patch.object(model_metadata, "_local_probe_disk_get", AsyncMock(return_value=None)),
    ):
        assert await model_metadata.detect_local_server_type(url) is None

    assert not model_metadata._endpoint_blackholed(url)


@pytest.mark.asyncio
async def test_metadata_timeout_skips_remaining_candidate():
    url = "http://10.0.0.9:30080/v1"
    client = AsyncClientStub(httpx.ConnectTimeout("timed out"))

    with (
        patch("httpx.AsyncClient", return_value=client),
        patch.object(model_metadata, "detect_local_server_type", AsyncMock(return_value=None)),
    ):
        assert await model_metadata.fetch_endpoint_model_metadata(url) == {}

    assert client.get.await_count == 1
    assert model_metadata._endpoint_blackholed(url)


@pytest.mark.asyncio
async def test_ollama_show_short_circuits_after_timeout():
    url = "http://10.0.0.9:30080/v1"
    first = AsyncClientStub(httpx.ConnectTimeout("timed out"))
    with patch("httpx.AsyncClient", return_value=first):
        assert await model_metadata._query_ollama_api_show_uncached("model", url) is None

    assert model_metadata._endpoint_blackholed(url)
    with patch("httpx.AsyncClient") as client_class:
        assert await model_metadata._query_ollama_api_show_uncached("model", url) is None
    client_class.assert_not_called()
