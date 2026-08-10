from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster

from plugins.memory.mem0._native_oss import _create_ollama_client


class _FakeOllamaBase:
    def __init__(
        self,
        client_factory,
        host=None,
        *,
        follow_redirects=True,
        timeout=None,
        headers=None,
        **kwargs,
    ):
        merged_headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "ollama-python/0.6.2 test",
            **(headers or {}),
        }
        self._client = client_factory(
            base_url=host or "http://127.0.0.1:11434",
            follow_redirects=follow_redirects,
            timeout=timeout,
            headers=merged_headers,
            **kwargs,
        )


class _FakeOllamaClient(_FakeOllamaBase):
    pass


@pytest.mark.asyncio
async def test_create_ollama_client_reuses_sdk_http_arguments_without_blocking(
    monkeypatch,
):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    blocker = BlockBuster()
    blocker.activate()
    try:
        client = await _create_ollama_client(
            _FakeOllamaClient,
            host="http://ollama.test:11434",
        )
    finally:
        blocker.deactivate()

    try:
        assert client._client.base_url == "http://ollama.test:11434"
        assert client._client.follow_redirects is True
        assert client._client.timeout.connect is None
        assert client._client.headers["content-type"] == "application/json"
        assert client._client.headers["accept"] == "application/json"
        assert client._client.headers["user-agent"] == "ollama-python/0.6.2 test"
    finally:
        await client._client.aclose()
