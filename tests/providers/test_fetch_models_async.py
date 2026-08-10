"""Native-async parity tests for provider model catalog discovery."""

from __future__ import annotations

import asyncio
import io
import inspect
import json
import urllib.request
import urllib.response
from contextlib import asynccontextmanager
from email.message import Message

import httpx
import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

from providers import get_provider_profile
from providers.base import ProviderProfile


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, dict[str, str]]:
    raw = await reader.readuntil(b"\r\n\r\n")
    lines = raw.decode("latin-1").split("\r\n")
    path = lines[0].split(" ", 2)[1]
    headers = {
        name.strip().lower(): value.strip()
        for line in lines[1:]
        if ":" in line
        for name, value in [line.split(":", 1)]
    }
    return path, headers


async def _respond(
    writer: asyncio.StreamWriter,
    *,
    status: str = "200 OK",
    headers: dict[str, str] | None = None,
    payload: dict | None = None,
) -> None:
    body = json.dumps(payload or {}).encode()
    response_headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Connection": "close",
        **(headers or {}),
    }
    head = "".join(
        [f"HTTP/1.1 {status}\r\n"]
        + [f"{name}: {value}\r\n" for name, value in response_headers.items()]
        + ["\r\n"]
    ).encode("latin-1")
    writer.write(head + body)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@asynccontextmanager
async def _http_server(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_fetch_models_uses_base_url_override_without_blocking_or_leaking():
    received_headers: dict[str, str] = {}

    async def handle(reader, writer):
        path, headers = await _read_request(reader)
        assert path == "/models"
        received_headers.update(headers)
        await _respond(writer, payload={"data": [{"id": "proxy-model-a"}]})

    async with _http_server(handle) as base_url:
        profile = ProviderProfile(
            name="test",
            base_url="http://127.0.0.1:1",
            default_headers={
                "Authorization": "Bearer profile-override",
                "User-Agent": "profile-agent",
            },
        )
        async with no_task_leaks(action=LeakAction.RAISE):
            blockbuster = BlockBuster()
            blockbuster.activate()
            try:
                result = await profile.fetch_models(
                    api_key="test-key", base_url=base_url
                )
            finally:
                blockbuster.deactivate()

    assert result == ["proxy-model-a"]
    assert received_headers["authorization"] == "Bearer profile-override"
    assert received_headers["user-agent"] == "profile-agent"


@pytest.mark.asyncio
@pytest.mark.parametrize("cross_origin", [False, True])
async def test_fetch_models_only_forwards_credentials_within_original_origin(
    cross_origin,
):
    received_headers: dict[str, str] = {}

    async def sink(reader, writer):
        _path, headers = await _read_request(reader)
        received_headers.update(headers)
        await _respond(writer, payload={"data": [{"id": "redirected-model"}]})

    async with _http_server(sink) as sink_url:
        if cross_origin:
            redirect_url = f"{sink_url}/redirected"

            async def source(reader, writer):
                await _read_request(reader)
                await _respond(
                    writer,
                    status="302 Found",
                    headers={"Location": redirect_url},
                )

            source_context = _http_server(source)
        else:
            source_url = ""

            async def source(reader, writer):
                path, headers = await _read_request(reader)
                if path == "/models":
                    await _respond(
                        writer,
                        status="302 Found",
                        headers={"Location": f"{source_url}/redirected"},
                    )
                else:
                    received_headers.update(headers)
                    await _respond(
                        writer,
                        payload={"data": [{"id": "redirected-model"}]},
                    )

            source_context = _http_server(source)

        async with source_context as resolved_source_url:
            source_url = resolved_source_url
            profile = ProviderProfile(
                name="test",
                base_url=source_url,
                default_headers={"x-api-key": "default-header-secret"},
            )
            result = await profile.fetch_models(api_key="bearer-secret")

    assert result == ["redirected-model"]
    if cross_origin:
        assert "authorization" not in received_headers
        assert "x-api-key" not in received_headers
    else:
        assert received_headers["authorization"] == "Bearer bearer-secret"
        assert received_headers["x-api-key"] == "default-header-secret"


@pytest.mark.asyncio
async def test_provider_specific_fetch_models_contracts(monkeypatch):
    profiles: dict[str, ProviderProfile] = {}
    for name in ("anthropic", "bedrock", "custom", "kimi", "openrouter", "vertex"):
        profile = await get_provider_profile(name)
        assert isinstance(profile, ProviderProfile)
        profiles[name] = profile

    assert all(
        inspect.iscoroutinefunction(type(profile).fetch_models)
        for profile in profiles.values()
    )
    assert await profiles["bedrock"].fetch_models() is None
    assert await profiles["vertex"].fetch_models() is None
    assert await profiles["custom"].fetch_models() is None
    assert await profiles["anthropic"].fetch_models() is None

    async def models_with_k3(self, **kwargs):
        return ["k3", "kimi-k2"]

    monkeypatch.setattr(ProviderProfile, "fetch_models", models_with_k3)
    assert await profiles["kimi"].fetch_models(
        base_url="https://api.moonshot.ai/v1"
    ) == ["kimi-k2"]


@pytest.mark.asyncio
async def test_anthropic_fetch_models_preserves_native_catalog_contract(monkeypatch):
    captured: dict[str, object] = {}

    async def open_catalog(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return urllib.response.addinfourl(
            io.BytesIO(b'{"data":[{"id":"claude-test"}]}'),
            Message(),
            request.full_url,
            200,
        )

    monkeypatch.setattr(
        "hermes_cli.urllib_security.open_credentialed_url", open_catalog
    )
    profile = await get_provider_profile("anthropic")
    assert isinstance(profile, ProviderProfile)

    assert await profile.fetch_models(api_key="anthropic-secret", timeout=3.0) == [
        "claude-test"
    ]
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "https://api.anthropic.com/v1/models"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["x-api-key"] == "anthropic-secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert captured["timeout"] == 3.0


@pytest.mark.asyncio
async def test_openrouter_fetch_models_uses_public_catalog_and_cache(monkeypatch):
    profile = await get_provider_profile("openrouter")
    assert isinstance(profile, ProviderProfile)
    module = inspect.getmodule(type(profile))
    assert module is not None
    monkeypatch.setattr(module, "_CACHE", None)
    calls: list[dict] = []

    async def fetch_public_catalog(self, **kwargs):
        calls.append(kwargs)
        return ["provider/model"]

    monkeypatch.setattr(ProviderProfile, "fetch_models", fetch_public_catalog)
    assert await profile.fetch_models(api_key="must-not-be-forwarded") == [
        "provider/model"
    ]
    assert await profile.fetch_models(api_key="still-not-forwarded") == [
        "provider/model"
    ]
    assert calls == [{"api_key": None, "base_url": None, "timeout": 8.0}]
