"""Profile isolation for the retained SearXNG provider."""

from __future__ import annotations

import asyncio

import aiofiles
import aiofiles.os
import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.web.searxng import provider as searxng_provider
from plugins.web.searxng.provider import SearXNGWebSearchProvider, _searxng_url


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_concurrent_profiles_use_their_own_searxng_endpoint(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "https://process.invalid")
    set_multiplex_active(True)
    requests: list[tuple[str, str]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"results": []}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, *, params, headers):
            requests.append((url, params["q"]))
            await asyncio.sleep(0)
            return Response()

    async def create_client(**_kwargs):
        return Client()

    monkeypatch.setattr(
        "plugins.web.searxng.provider._create_httpx_client",
        create_client,
    )

    async def search(profile: str):
        token = set_secret_scope(
            {"SEARXNG_URL": f"https://{profile}.example"}
        )
        try:
            return await SearXNGWebSearchProvider().search(profile)
        finally:
            reset_secret_scope(token)

    result_a, result_b = await asyncio.gather(search("alpha"), search("beta"))

    assert result_a == {"success": True, "data": {"web": []}}
    assert result_b == {"success": True, "data": {"web": []}}
    assert set(requests) == {
        ("https://alpha.example/search", "alpha"),
        ("https://beta.example/search", "beta"),
    }


@pytest.mark.asyncio
async def test_unscoped_multiplex_lookup_fails_closed(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "https://process.invalid")
    set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError, match="SEARXNG_URL"):
        await SearXNGWebSearchProvider().is_available()
    with pytest.raises(UnscopedSecretError, match="SEARXNG_URL"):
        await SearXNGWebSearchProvider().search("must not leak")


@pytest.mark.asyncio
async def test_single_profile_config_and_explicit_empty_precedence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "profile"
    await aiofiles.os.makedirs(home)
    async with aiofiles.open(home / ".env", "w", encoding="utf-8") as handle:
        await handle.write("SEARXNG_URL=https://config.example/\n")
    home_token = set_hermes_home_override(home)
    try:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        set_multiplex_active(False)
        assert await _searxng_url() == "https://config.example/"

        monkeypatch.setenv("SEARXNG_URL", "  https://process.example/  ")
        assert await _searxng_url() == "https://process.example/"

        monkeypatch.setenv("SEARXNG_URL", "")
        assert await _searxng_url() == ""

        monkeypatch.setenv("SEARXNG_URL", "https://process.invalid")
        set_multiplex_active(True)
        scope_token = set_secret_scope({})
        try:
            assert await _searxng_url() == "https://config.example/"
        finally:
            reset_secret_scope(scope_token)

        scope_token = set_secret_scope({"SEARXNG_URL": ""})
        try:
            assert await _searxng_url() == ""
            assert await SearXNGWebSearchProvider().is_available() is False
        finally:
            reset_secret_scope(scope_token)

        scope_token = set_secret_scope(
            {"SEARXNG_URL": "  https://scope.example/  "}
        )
        try:
            assert await _searxng_url() == "https://scope.example/"
        finally:
            reset_secret_scope(scope_token)
    finally:
        reset_hermes_home_override(home_token)


async def _close_test_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        pass


@pytest.mark.asyncio
async def test_real_http_client_is_closed_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_closed = asyncio.Event()
    handler_done = asyncio.Event()

    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            body = b'{"results": []}'
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                + b"Connection: keep-alive\r\n\r\n"
                + body
            )
            await writer.drain()
            await reader.read()
            connection_closed.set()
        finally:
            await _close_test_writer(writer)
            handler_done.set()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("SEARXNG_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    clients = []
    create_client = searxng_provider._create_httpx_client

    async def capture_client(**kwargs):
        client = await create_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(searxng_provider, "_create_httpx_client", capture_client)
    try:
        result = await SearXNGWebSearchProvider().search("cleanup")
        await asyncio.wait_for(connection_closed.wait(), timeout=2)
        await asyncio.wait_for(handler_done.wait(), timeout=2)
    finally:
        server.close()
        await server.wait_closed()

    assert result == {"success": True, "data": {"web": []}}
    assert len(clients) == 1
    assert clients[0].is_closed is True


@pytest.mark.asyncio
async def test_real_http_client_closes_connection_before_propagating_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_started = asyncio.Event()
    connection_closed = asyncio.Event()
    handler_done = asyncio.Event()

    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            request_started.set()
            await reader.read()
            connection_closed.set()
        finally:
            await _close_test_writer(writer)
            handler_done.set()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("SEARXNG_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    clients = []
    create_client = searxng_provider._create_httpx_client

    async def capture_client(**kwargs):
        client = await create_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(searxng_provider, "_create_httpx_client", capture_client)
    task = asyncio.create_task(SearXNGWebSearchProvider().search("cancel"))
    try:
        await asyncio.wait_for(request_started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(connection_closed.wait(), timeout=2)
        await asyncio.wait_for(handler_done.wait(), timeout=2)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        server.close()
        await server.wait_closed()

    assert len(clients) == 1
    assert clients[0].is_closed is True
