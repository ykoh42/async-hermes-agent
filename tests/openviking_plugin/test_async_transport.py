"""Native-async OpenViking transport, cancellation, and lifecycle tests."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


pytestmark = pytest.mark.asyncio


@dataclass
class _ServerState:
    requests: list[tuple[str, str, dict]] = field(default_factory=list)
    stall_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_stall: asyncio.Event = field(default_factory=asyncio.Event)


@asynccontextmanager
async def _openviking_server(*, response_delay: float = 0.0):
    state = _ServerState()
    handlers: set[asyncio.Task[None]] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            handlers.add(task)
        try:
            header_data = await reader.readuntil(b"\r\n\r\n")
            header_lines = header_data.decode("latin-1").split("\r\n")
            method, path, _version = header_lines[0].split(" ", 2)
            headers = {}
            for line in header_lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.lower()] = value.strip()
            content_length = int(headers.get("content-length", "0"))
            body_data = await reader.readexactly(content_length) if content_length else b""
            body = json.loads(body_data) if body_data else {}
            state.requests.append((method, path, body))

            if body.get("content") == "stall":
                state.stall_started.set()
                await state.release_stall.wait()
            if response_delay:
                await asyncio.sleep(response_delay)

            payload = {
                "status": "ok",
                "result": {
                    "uri": body.get("uri", ""),
                    "written_bytes": len(str(body.get("content", ""))),
                },
            }
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
            if task is not None:
                handlers.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        yield f"http://{host}:{port}", state
    finally:
        state.release_stall.set()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*tuple(handlers), return_exceptions=True)


async def test_provider_tool_transport_is_concurrent_nonblocking_and_closes():
    async with _openviking_server(response_delay=0.02) as (endpoint, state):
        provider = OpenVikingMemoryProvider()
        client = _VikingClient(endpoint, account="acct", user="user", agent="hermes")
        provider._client = client
        provider._endpoint = endpoint
        provider._account = "acct"
        provider._user = "user"
        provider._agent = "hermes"
        heartbeat_count = 0
        running = True

        async def heartbeat() -> None:
            nonlocal heartbeat_count
            while running:
                heartbeat_count += 1
                await asyncio.sleep(0)

        async with no_task_leaks(action=LeakAction.RAISE):
            heartbeat_task = asyncio.create_task(heartbeat())
            blocker = BlockBuster()
            blocker.activate()
            try:
                results = await asyncio.gather(*(
                    provider.handle_tool_call(
                        "viking_remember",
                        {"content": f"fact-{index}", "category": "event"},
                    )
                    for index in range(12)
                ))
            finally:
                blocker.deactivate()
                running = False
                await heartbeat_task
                await provider.shutdown()

        assert all(json.loads(result)["status"] == "stored" for result in results)
        assert len(state.requests) == 12
        assert all(path == "/api/v1/content/write" for _method, path, _body in state.requests)
        assert heartbeat_count > 10
        assert client._http.is_closed


async def test_provider_tool_cancellation_propagates_without_task_leaks():
    async with _openviking_server() as (endpoint, state):
        provider = OpenVikingMemoryProvider()
        provider._client = _VikingClient(endpoint)
        provider._endpoint = endpoint

        async with no_task_leaks(action=LeakAction.RAISE):
            request = asyncio.create_task(
                provider.handle_tool_call("viking_remember", {"content": "stall"})
            )
            await asyncio.wait_for(state.stall_started.wait(), timeout=1.0)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            state.release_stall.set()
            await provider.shutdown()


async def test_viking_client_timeout_propagates_and_client_closes():
    async with _openviking_server() as (endpoint, state):
        client = _VikingClient(endpoint)
        async with no_task_leaks(action=LeakAction.RAISE):
            with pytest.raises(httpx.TimeoutException):
                await client.post(
                    "/api/v1/content/write",
                    {"content": "stall"},
                    timeout=0.02,
                )
            state.release_stall.set()
            await client.close()
        assert client._http.is_closed


async def test_shutdown_wins_health_refresh_race_and_closes_candidate(monkeypatch):
    health_started = asyncio.Event()
    release_health = asyncio.Event()
    candidates = []

    class Candidate:
        def __init__(self, *args, **kwargs):
            self.closed = False
            candidates.append(self)

        async def health(self):
            health_started.set()
            await release_health.wait()
            return True

        async def close(self):
            self.closed = True

    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933")
    monkeypatch.setattr(openviking_module, "_VikingClient", Candidate)
    provider = OpenVikingMemoryProvider()
    provider._env_refresh_enabled = True

    refresh_task = asyncio.create_task(provider._ensure_client())
    await asyncio.wait_for(health_started.wait(), timeout=1.0)
    shutdown_task = asyncio.create_task(provider.shutdown())
    await asyncio.sleep(0)
    release_health.set()

    refreshed, _closed = await asyncio.wait_for(
        asyncio.gather(refresh_task, shutdown_task),
        timeout=1.0,
    )
    assert refreshed is None
    assert provider._client is None
    assert len(candidates) == 1
    assert candidates[0].closed is True
