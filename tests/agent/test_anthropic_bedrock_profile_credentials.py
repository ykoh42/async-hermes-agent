"""Profile-scoped AWS credentials for Anthropic's Bedrock transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent import anthropic_adapter
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)


@dataclass
class _FrozenCredentials:
    access_key: str
    secret_key: str
    token: str | None


class _DefaultCredentials:
    def __init__(self) -> None:
        self.frozen_calls = 0

    async def get_frozen_credentials(self) -> _FrozenCredentials:
        self.frozen_calls += 1
        return _FrozenCredentials("DEFAULT", "DEFAULT-SECRET", "DEFAULT-TOKEN")


class _FakeBedrockClient:
    instances: list[_FakeBedrockClient] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.aws_access_key = kwargs["aws_access_key"]
        self.aws_secret_key = kwargs["aws_secret_key"]
        self.aws_session_token = kwargs["aws_session_token"]
        self._client = kwargs["http_client"]
        self.closed = 0
        self.instances.append(self)

    async def close(self) -> None:
        self.closed += 1
        await self._client.aclose()


class _FakeHTTPClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous = is_multiplex_active()
    token = set_secret_scope(None)
    set_multiplex_active(False)
    _FakeBedrockClient.instances.clear()
    try:
        yield
    finally:
        set_multiplex_active(previous)
        reset_secret_scope(token)


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = SimpleNamespace(AsyncAnthropicBedrock=_FakeBedrockClient)
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", sdk)

    async def build_http_client(_sdk, *, base_url, timeout):
        await asyncio.sleep(0)
        return _FakeHTTPClient(base_url)

    monkeypatch.setattr(
        anthropic_adapter,
        "_build_anthropic_default_http_client",
        build_http_client,
    )
    return sdk


async def _build_under_scope(secrets):
    token = set_secret_scope(secrets)
    try:
        return await anthropic_adapter.build_anthropic_bedrock_client("us-west-2")
    finally:
        reset_secret_scope(token)


@pytest.mark.asyncio
async def test_concurrent_profiles_pass_only_their_static_aws_credentials(
    monkeypatch,
    fake_sdk,
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FOREIGN")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "FOREIGN-SECRET")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "FOREIGN-TOKEN")
    monkeypatch.setenv("AWS_PROFILE", "foreign-profile")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "FOREIGN-BEARER")
    set_multiplex_active(True)

    client_a, client_b = await asyncio.gather(
        _build_under_scope({
            "AWS_ACCESS_KEY_ID": "PROFILE-A",
            "AWS_SECRET_ACCESS_KEY": "SECRET-A",
            "AWS_SESSION_TOKEN": "TOKEN-A",
            "ANTHROPIC_BEDROCK_BASE_URL": "https://a.example",
        }),
        _build_under_scope({
            "AWS_ACCESS_KEY_ID": "PROFILE-B",
            "AWS_SECRET_ACCESS_KEY": "SECRET-B",
            "ANTHROPIC_BEDROCK_BASE_URL": "https://b.example",
        }),
    )

    assert {
        (
            client.aws_access_key,
            client.aws_secret_key,
            client.aws_session_token,
            client._client.base_url,
        )
        for client in (client_a, client_b)
    } == {
        ("PROFILE-A", "SECRET-A", "TOKEN-A", "https://a.example"),
        ("PROFILE-B", "SECRET-B", None, "https://b.example"),
    }
    assert all(
        client.kwargs["aws_region"] == "us-west-2"
        and client.kwargs["max_retries"] == 0
        and "context-1m-2025-08-07"
        in client.kwargs["default_headers"]["anthropic-beta"]
        for client in (client_a, client_b)
    )
    await asyncio.gather(client_a.close(), client_b.close())
    assert [client.closed for client in (client_a, client_b)] == [1, 1]
    assert [client._client.closed for client in (client_a, client_b)] == [1, 1]


@pytest.mark.asyncio
async def test_multiplex_unscoped_empty_unsafe_and_bearer_fail_closed(
    monkeypatch,
    fake_sdk,
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FOREIGN")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "FOREIGN-SECRET")
    set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError):
        await anthropic_adapter.build_anthropic_bedrock_client("us-west-2")
    with pytest.raises(RuntimeError, match="explicit profile-scoped AWS"):
        await _build_under_scope({})
    with pytest.raises(RuntimeError, match="AWS_PROFILE"):
        await _build_under_scope({"AWS_PROFILE": "profile-a"})
    with pytest.raises(RuntimeError, match="supports only SigV4"):
        await _build_under_scope({"AWS_BEARER_TOKEN_BEDROCK": "profile-bearer"})

    assert _FakeBedrockClient.instances == []


@pytest.mark.asyncio
async def test_single_profile_retains_async_default_chain(
    monkeypatch,
    fake_sdk,
):
    credentials = _DefaultCredentials()
    session = SimpleNamespace(get_credentials=lambda: None)

    async def get_credentials():
        return credentials

    session.get_credentials = get_credentials
    monkeypatch.setattr(
        "agent.bedrock_adapter._aiobotocore_get_session",
        lambda: session,
    )
    monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", "https://legacy.example")

    client = await anthropic_adapter.build_anthropic_bedrock_client("eu-west-1")

    assert (
        client.aws_access_key,
        client.aws_secret_key,
        client.aws_session_token,
        client.kwargs["aws_region"],
        client._client.base_url,
    ) == (
        "DEFAULT",
        "DEFAULT-SECRET",
        "DEFAULT-TOKEN",
        "eu-west-1",
        "https://legacy.example",
    )
    assert credentials.frozen_calls == 1
    assert client._hermes_aws_credentials is credentials
    await client.close()


@pytest.mark.asyncio
async def test_constructor_failure_repeated_cancellation_closes_owned_http(
    monkeypatch,
):
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class BlockingHTTPClient:
        async def aclose(self) -> None:
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    class FailingBedrockClient:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("constructor failed")

    sdk = SimpleNamespace(AsyncAnthropicBedrock=FailingBedrockClient)
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", sdk)

    async def build_http_client(_sdk, *, base_url, timeout):
        return BlockingHTTPClient()

    monkeypatch.setattr(
        anthropic_adapter,
        "_build_anthropic_default_http_client",
        build_http_client,
    )
    set_multiplex_active(True)
    task = asyncio.create_task(
        _build_under_scope({
            "AWS_ACCESS_KEY_ID": "PROFILE-A",
            "AWS_SECRET_ACCESS_KEY": "SECRET-A",
        })
    )
    await close_started.wait()
    task.cancel("first")
    await asyncio.sleep(0)
    task.cancel("second")
    assert not task.done()

    allow_close.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert cancelled.value.args == ("first",)
    assert close_finished.is_set()
    assert not [
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
        and pending.get_name() == "anthropic-http-client-construction-cleanup"
        and not pending.done()
    ]
