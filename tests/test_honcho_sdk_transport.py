"""Pinned honcho-ai native transport and lifecycle integration tests."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
from honcho.http.exceptions import TimeoutError as HonchoTimeoutError

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
)


pytestmark = pytest.mark.asyncio


@dataclass
class _ServerState:
    requests: list[tuple[str, str, dict[str, str], dict]] = field(default_factory=list)
    stall_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_stall: asyncio.Event = field(default_factory=asyncio.Event)


@asynccontextmanager
async def _honcho_server(*, response_delay: float = 0.0):
    state = _ServerState()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header_data = await reader.readuntil(b"\r\n\r\n")
            header_lines = header_data.decode("latin-1").split("\r\n")
            method, path, _ = header_lines[0].split(" ", 2)
            headers = {}
            for line in header_lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.lower()] = value.strip()
            content_length = int(headers.get("content-length", "0"))
            body_data = await reader.readexactly(content_length) if content_length else b""
            body = json.loads(body_data) if body_data else {}
            state.requests.append((method, path, headers, body))

            if path == "/stall":
                state.stall_started.set()
                await state.release_stall.wait()
                payload = {"released": True}
            elif path == "/v3/workspaces":
                payload = {
                    "id": body["id"],
                    "metadata": {},
                    "configuration": {},
                    "created_at": "2026-08-10T00:00:00Z",
                }
            elif path.startswith("/v3/workspaces/async-test/peers"):
                payload = {
                    "id": body["id"],
                    "workspace_id": "async-test",
                    "metadata": {},
                    "configuration": {},
                    "created_at": "2026-08-10T00:00:00Z",
                }
            else:
                payload = {"ok": True}

            if response_delay:
                await asyncio.sleep(response_delay)
            encoded = json.dumps(payload).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(encoded)}\r\n".encode()
                + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
                + encoded
            )
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        yield f"http://{host}:{port}", state
    finally:
        state.release_stall.set()
        server.close()
        await server.wait_closed()


async def test_pinned_sdk_uses_only_async_transport_and_closes_it():
    await reset_honcho_client()
    async with _honcho_server() as (base_url, state):
        cfg = HonchoClientConfig(
            enabled=True,
            base_url=base_url,
            workspace_id="async-test",
            timeout=1.0,
        )
        client = await get_honcho_client(cfg)
        async_http = client._async_http

        assert getattr(client, "_http", None) is None
        assert async_http is not None
        assert not async_http._client.is_closed

        peer = await client.aio.peer("peer-1")
        assert peer.id == "peer-1"
        assert [request[1] for request in state.requests] == [
            "/v3/workspaces",
            "/v3/workspaces/async-test/peers",
        ]

        await reset_honcho_client()
        assert async_http._client.is_closed


async def test_pinned_sdk_concurrent_requests_do_not_block_event_loop():
    await reset_honcho_client()
    async with _honcho_server(response_delay=0.02) as (base_url, state):
        cfg = HonchoClientConfig(
            enabled=True,
            base_url=base_url,
            workspace_id="async-test",
            timeout=1.0,
        )
        client = await get_honcho_client(cfg)
        heartbeat_count = 0
        running = True

        async def heartbeat() -> None:
            nonlocal heartbeat_count
            while running:
                heartbeat_count += 1
                await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            peers = await asyncio.gather(
                *(client.aio.peer(f"peer-{index}") for index in range(12))
            )
        finally:
            running = False
            await heartbeat_task
            await reset_honcho_client()

        assert [peer.id for peer in peers] == [f"peer-{index}" for index in range(12)]
        peer_requests = [path for _, path, _, _ in state.requests if path.endswith("/peers")]
        assert len(peer_requests) == 12
        assert heartbeat_count > 10


async def test_pinned_sdk_cancellation_and_timeout_propagate_and_close():
    await reset_honcho_client()
    async with _honcho_server() as (base_url, state):
        cfg = HonchoClientConfig(
            enabled=True,
            base_url=base_url,
            workspace_id="async-test",
            timeout=0.05,
        )
        client = await get_honcho_client(cfg)
        async_http = client._async_http
        assert async_http is not None
        async_http.max_retries = 0

        request_task = asyncio.create_task(async_http.get("/stall"))
        await asyncio.wait_for(state.stall_started.wait(), timeout=1)
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        state.stall_started.clear()
        timeout_task = asyncio.create_task(async_http.get("/stall"))
        await asyncio.wait_for(state.stall_started.wait(), timeout=1)
        with pytest.raises(HonchoTimeoutError, match="timed out"):
            await timeout_task

        await reset_honcho_client()
        assert async_http._client.is_closed
