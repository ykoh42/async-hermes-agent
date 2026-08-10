from unittest.mock import patch

import httpx
import pytest
from blockbuster import BlockBuster

from hermes_cli.urllib_security import open_credentialed_url


@pytest.mark.asyncio
async def test_default_credentialed_client_setup_does_not_block(monkeypatch):
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

    request = httpx.Request(
        "GET",
        "https://catalog.example.test/models",
        headers={"Authorization": "Bearer secret"},
    )

    async def send(_client, outgoing):
        return httpx.Response(200, json={"data": []}, request=outgoing)

    blocker = BlockBuster()
    blocker.activate()
    try:
        with patch.object(httpx.AsyncClient, "send", new=send):
            response = await open_credentialed_url(request, timeout=5.0)
    finally:
        blocker.deactivate()

    assert response.status_code == 200
