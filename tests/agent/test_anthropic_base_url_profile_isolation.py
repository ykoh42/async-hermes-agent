"""Profile isolation for Anthropic-compatible endpoint settings."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from agent import anthropic_adapter
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_default_http_transport_uses_active_profile_base_url(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.invalid")
    set_multiplex_active(True)

    base_client = SimpleNamespace(AsyncHttpxClientWrapper=object())
    monkeypatch.setitem(sys.modules, "anthropic._base_client", base_client)
    sdk = SimpleNamespace(DEFAULT_CONNECTION_LIMITS=httpx.Limits())

    async def create_client(**kwargs):
        await asyncio.sleep(0)
        return kwargs["base_url"]

    monkeypatch.setattr("agent.ssl_verify._create_httpx_client", create_client)

    async def resolve(url: str):
        token = set_secret_scope({"ANTHROPIC_BASE_URL": url})
        try:
            return await anthropic_adapter._build_anthropic_default_http_client(
                sdk,
                base_url=None,
                timeout=httpx.Timeout(1.0),
            )
        finally:
            reset_secret_scope(token)

    result_a, result_b = await asyncio.gather(
        resolve("https://alpha.example"),
        resolve("https://beta.example"),
    )

    assert result_a == "https://alpha.example"
    assert result_b == "https://beta.example"


@pytest.mark.asyncio
async def test_bedrock_transport_uses_active_profile_base_url(monkeypatch):
    monkeypatch.setenv(
        "ANTHROPIC_BEDROCK_BASE_URL",
        "https://process-bedrock.invalid",
    )
    set_multiplex_active(True)

    class FakeBedrockClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sdk = SimpleNamespace(AsyncAnthropicBedrock=FakeBedrockClient)
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", sdk)

    class FakeHTTPClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        async def aclose(self) -> None:
            return None

    async def build_http_client(_sdk, *, base_url, timeout):
        await asyncio.sleep(0)
        return FakeHTTPClient(base_url)

    monkeypatch.setattr(
        anthropic_adapter,
        "_build_anthropic_default_http_client",
        build_http_client,
    )

    async def resolve(url: str):
        token = set_secret_scope(
            {
                "AWS_ACCESS_KEY_ID": "access",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "AWS_SESSION_TOKEN": "token",
                "ANTHROPIC_BEDROCK_BASE_URL": url,
            }
        )
        try:
            return await anthropic_adapter.build_anthropic_bedrock_client(
                "us-west-2"
            )
        finally:
            reset_secret_scope(token)

    client_a, client_b = await asyncio.gather(
        resolve("https://alpha-bedrock.example"),
        resolve("https://beta-bedrock.example"),
    )

    assert client_a is not client_b
    assert client_a.kwargs["http_client"].base_url == (
        "https://alpha-bedrock.example"
    )
    assert client_b.kwargs["http_client"].base_url == (
        "https://beta-bedrock.example"
    )


@pytest.mark.asyncio
async def test_unscoped_multiplex_base_url_read_fails_closed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.invalid")
    set_multiplex_active(True)
    base_client = SimpleNamespace(AsyncHttpxClientWrapper=object())
    monkeypatch.setitem(sys.modules, "anthropic._base_client", base_client)
    sdk = SimpleNamespace(DEFAULT_CONNECTION_LIMITS=httpx.Limits())

    with pytest.raises(UnscopedSecretError, match="ANTHROPIC_BASE_URL"):
        await anthropic_adapter._build_anthropic_default_http_client(
            sdk,
            base_url=None,
            timeout=httpx.Timeout(1.0),
        )


@pytest.mark.asyncio
async def test_explicit_base_url_bypasses_unscoped_multiplex_env_lookup(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.invalid")
    set_multiplex_active(True)
    base_client = SimpleNamespace(AsyncHttpxClientWrapper=object())
    monkeypatch.setitem(sys.modules, "anthropic._base_client", base_client)
    sdk = SimpleNamespace(DEFAULT_CONNECTION_LIMITS=httpx.Limits())

    async def create_client(**kwargs):
        return kwargs["base_url"]

    monkeypatch.setattr("agent.ssl_verify._create_httpx_client", create_client)
    token = set_secret_scope(None)
    try:
        result = await anthropic_adapter._build_anthropic_default_http_client(
            sdk,
            base_url="https://explicit.example",
            timeout=httpx.Timeout(1.0),
        )
    finally:
        reset_secret_scope(token)

    assert result == "https://explicit.example"


@pytest.mark.asyncio
async def test_scoped_empty_base_url_does_not_borrow_process_environment(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.invalid")
    set_multiplex_active(True)
    base_client = SimpleNamespace(AsyncHttpxClientWrapper=object())
    monkeypatch.setitem(sys.modules, "anthropic._base_client", base_client)
    sdk = SimpleNamespace(DEFAULT_CONNECTION_LIMITS=httpx.Limits())

    async def create_client(**kwargs):
        return kwargs["base_url"]

    monkeypatch.setattr("agent.ssl_verify._create_httpx_client", create_client)
    token = set_secret_scope({"ANTHROPIC_BASE_URL": ""})
    try:
        result = await anthropic_adapter._build_anthropic_default_http_client(
            sdk,
            base_url=None,
            timeout=httpx.Timeout(1.0),
        )
    finally:
        reset_secret_scope(token)

    assert result == ""


@pytest.mark.asyncio
async def test_bedrock_unscoped_multiplex_base_url_read_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv(
        "ANTHROPIC_BEDROCK_BASE_URL",
        "https://process-bedrock.invalid",
    )
    set_multiplex_active(True)

    sdk = SimpleNamespace(AsyncAnthropicBedrock=object())
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", sdk)
    build_http_client = AsyncMock()
    monkeypatch.setattr(
        anthropic_adapter,
        "_build_anthropic_default_http_client",
        build_http_client,
    )

    token = set_secret_scope(None)
    try:
        with pytest.raises(
            UnscopedSecretError,
            match="AWS Bedrock credential resolution",
        ):
            await anthropic_adapter.build_anthropic_bedrock_client("us-west-2")
    finally:
        reset_secret_scope(token)

    build_http_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_base_url_cancellation_finishes_owned_http_cleanup(
    monkeypatch,
):
    set_multiplex_active(True)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()
    seen_base_urls: list[str] = []

    class OwnedHTTPClient:
        async def aclose(self) -> None:
            close_started.set()
            await release_close.wait()
            close_finished.set()

    class FailingAsyncAnthropic:
        def __init__(self, **_kwargs):
            raise RuntimeError("constructor failed")

    base_client = SimpleNamespace(AsyncHttpxClientWrapper=object())
    monkeypatch.setitem(sys.modules, "anthropic._base_client", base_client)
    sdk = SimpleNamespace(
        AsyncAnthropic=FailingAsyncAnthropic,
        DEFAULT_CONNECTION_LIMITS=httpx.Limits(),
    )
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", sdk)

    async def create_client(**kwargs):
        seen_base_urls.append(kwargs["base_url"])
        return OwnedHTTPClient()

    monkeypatch.setattr("agent.ssl_verify._create_httpx_client", create_client)

    token = set_secret_scope(
        {"ANTHROPIC_BASE_URL": "https://cancel-profile.example"}
    )
    build = None
    try:
        build = asyncio.create_task(
            anthropic_adapter.build_anthropic_client(
                "sk-ant-api03-cancellation-test"
            )
        )
        await close_started.wait()
        build.cancel()
        await asyncio.sleep(0)
        build.cancel()
        await asyncio.sleep(0)

        assert build.done() is False
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await build
    finally:
        release_close.set()
        if build is not None and not build.done():
            build.cancel()
            await asyncio.gather(build, return_exceptions=True)
        reset_secret_scope(token)

    assert seen_base_urls == ["https://cancel-profile.example"]
    assert close_finished.is_set()
