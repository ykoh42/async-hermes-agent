import asyncio
from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster

from tools.fal_common import _close_fal_client, _create_fal_client


class _FakeFALClient:
    default_timeout = 120.0

    def _get_auth(self):
        return SimpleNamespace(header_value="Key test-id:test-secret")


@pytest.mark.asyncio
async def test_create_fal_client_materializes_http_transport_without_blocking(
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

    fal_client = SimpleNamespace(
        AsyncClient=_FakeFALClient,
        client=SimpleNamespace(USER_AGENT="fal-client/0.13.1 (python)"),
    )
    blocker = BlockBuster()
    blocker.activate()
    try:
        client = await _create_fal_client(fal_client)
    finally:
        blocker.deactivate()

    try:
        assert client._client.timeout.connect == 120.0
        assert client._client.timeout.read == 120.0
        assert client._client.headers["authorization"] == "Key test-id:test-secret"
        assert client._client.headers["user-agent"] == "fal-client/0.13.1 (python)"
    finally:
        await client._client.aclose()


@pytest.mark.asyncio
async def test_close_fal_client_survives_repeated_cancellation():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class HttpClient:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    client = SimpleNamespace(_client=HttpClient())
    task = asyncio.create_task(_close_fal_client(client))
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_close_fal_client_preserves_cancellation_when_close_fails():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class HttpClient:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            raise RuntimeError("close failed")

    client = SimpleNamespace(_client=HttpClient())
    task = asyncio.create_task(_close_fal_client(client))
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task
