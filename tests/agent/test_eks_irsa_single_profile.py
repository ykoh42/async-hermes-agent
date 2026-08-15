"""Realistic single-profile EKS IRSA coverage through aiobotocore."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import os
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from agent import bedrock_adapter as bedrock


_AWS_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_SESSION_NAME",
    "AWS_EC2_METADATA_DISABLED",
)


def _expiration(seconds: int = 300) -> str:
    return (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@contextlib.asynccontextmanager
async def _loopback_sts_server(response_factory):
    requests: list[dict[str, str]] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            content_length = next(
                int(line.split(":", 1)[1].strip())
                for line in headers.split("\r\n")
                if line.lower().startswith("content-length:")
            )
            body = (await reader.readexactly(content_length)).decode("ascii")
            request = {
                key: values[-1]
                for key, values in parse_qs(body, keep_blank_values=True).items()
            }
            requests.append(request)
            payload = response_factory(len(requests) - 1).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/xml\r\n"
                b"Content-Length: "
                + str(len(payload)).encode("ascii")
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    socket = server.sockets[0]
    assert socket is not None
    try:
        yield socket.getsockname()[1], requests
    finally:
        server.close()
        await server.wait_closed()


def _sts_response(
    *,
    access_key: str,
    secret_key: str,
    session_token: str,
    expiration: str,
) -> str:
    return f"""\
<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleWithWebIdentityResult>
    <Credentials>
      <AccessKeyId>{access_key}</AccessKeyId>
      <SecretAccessKey>{secret_key}</SecretAccessKey>
      <SessionToken>{session_token}</SessionToken>
      <Expiration>{expiration}</Expiration>
    </Credentials>
    <SubjectFromWebIdentityToken>system:serviceaccount:default:hermes</SubjectFromWebIdentityToken>
    <AssumedRoleUser>
      <AssumedRoleId>AROAEXAMPLE:botocore-session</AssumedRoleId>
      <Arn>arn:aws:iam::123456789012:role/hermes</Arn>
    </AssumedRoleUser>
    <Audience>sts.amazonaws.com</Audience>
    <Provider>https://oidc.eks.example/id/cluster</Provider>
  </AssumeRoleWithWebIdentityResult>
  <ResponseMetadata><RequestId>loopback-request</RequestId></ResponseMetadata>
</AssumeRoleWithWebIdentityResponse>
"""


@pytest.fixture
def aiobotocore_modules():
    return pytest.importorskip("aiobotocore.credentials"), pytest.importorskip(
        "aiobotocore.session"
    )


@pytest.fixture
def irsa_environment(monkeypatch, tmp_path):
    for name in _AWS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    token_file = tmp_path / "var" / "run" / "secrets" / "eks" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("jwt-eks-a", encoding="utf-8")
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/hermes")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    return token_file


def _route_sts_to_loopback(monkeypatch, credentials_module, port):
    from aiobotocore.utils import create_nested_client as real_nested_client

    def nested_client(session, service_name, **kwargs):
        if service_name == "sts":
            kwargs["endpoint_url"] = f"http://127.0.0.1:{port}"
        return real_nested_client(session, service_name, **kwargs)

    monkeypatch.setattr(credentials_module, "create_nested_client", nested_client)


@pytest.mark.asyncio
async def test_single_profile_irsa_uses_actual_aiobotocore_sts_exchange(
    aiobotocore_modules,
    irsa_environment,
    monkeypatch,
):
    credentials_module, session_module = aiobotocore_modules
    async with _loopback_sts_server(
        lambda _index: _sts_response(
            access_key="AKIA_LOOPBACK",
            secret_key="secret-loopback",
            session_token="session-loopback",
            expiration=_expiration(3600),
        )
    ) as (port, requests):
        _route_sts_to_loopback(monkeypatch, credentials_module, port)
        assert await bedrock.resolve_aws_auth_env_var() == "AWS_WEB_IDENTITY_TOKEN_FILE"

        session = session_module.get_session()
        credentials = await session.get_credentials()
        assert credentials is not None
        frozen = await credentials.get_frozen_credentials()

    assert frozen.access_key == "AKIA_LOOPBACK"
    assert frozen.secret_key == "secret-loopback"
    assert frozen.token == "session-loopback"
    assert len(requests) == 1
    assert requests[0]["Action"] == "AssumeRoleWithWebIdentity"
    assert requests[0]["RoleArn"] == "arn:aws:iam::123456789012:role/hermes"
    assert requests[0]["WebIdentityToken"] == "jwt-eks-a"


@pytest.mark.asyncio
async def test_single_profile_irsa_refresh_reloads_rotated_projected_token(
    aiobotocore_modules,
    irsa_environment,
    monkeypatch,
):
    credentials_module, session_module = aiobotocore_modules
    irsa_environment.write_text("jwt-eks-a", encoding="utf-8")
    async with _loopback_sts_server(
        lambda index: _sts_response(
            access_key=f"AKIA_{index}",
            secret_key=f"secret-{index}",
            session_token=f"session-{index}",
            expiration=_expiration(300),
        )
    ) as (port, requests):
        _route_sts_to_loopback(monkeypatch, credentials_module, port)
        session = session_module.get_session()
        credentials = await session.get_credentials()
        assert credentials is not None
        first = await credentials.get_frozen_credentials()
        irsa_environment.write_text("jwt-eks-b", encoding="utf-8")
        second = await credentials.get_frozen_credentials()

    assert first.access_key == "AKIA_0"
    assert second.access_key == "AKIA_1"
    assert [request["WebIdentityToken"] for request in requests] == [
        "jwt-eks-a",
        "jwt-eks-b",
    ]


@pytest.mark.asyncio
async def test_single_profile_irsa_concurrent_first_requests_single_flight(
    aiobotocore_modules,
    irsa_environment,
    monkeypatch,
):
    credentials_module, session_module = aiobotocore_modules
    async with _loopback_sts_server(
        lambda _index: _sts_response(
            access_key="AKIA_CONCURRENT",
            secret_key="secret-concurrent",
            session_token="session-concurrent",
            expiration=_expiration(3600),
        )
    ) as (port, requests):
        _route_sts_to_loopback(monkeypatch, credentials_module, port)
        session = session_module.get_session()
        credentials = await session.get_credentials()
        assert credentials is not None
        frozen = await asyncio.gather(
            *(credentials.get_frozen_credentials() for _ in range(20))
        )

    assert len(requests) == 1
    assert {item.access_key for item in frozen} == {"AKIA_CONCURRENT"}


@pytest.mark.asyncio
async def test_multiplexed_irsa_fails_closed_before_sdk_chain(
    aiobotocore_modules,
    irsa_environment,
    monkeypatch,
):
    pytest.importorskip("agent.bedrock_adapter")
    from agent import secret_scope

    credentials_module, _session_module = aiobotocore_modules
    previous = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope({
        "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/profile",
        "AWS_WEB_IDENTITY_TOKEN_FILE": str(irsa_environment),
    })
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(bedrock._BedrockProfileIsolationError):
            await bedrock.resolve_aws_auth_env_var()
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_irsa_server_handles_fastapi_like_concurrent_requests(
    aiobotocore_modules,
    irsa_environment,
    monkeypatch,
):
    credentials_module, session_module = aiobotocore_modules
    async with _loopback_sts_server(
        lambda _index: _sts_response(
            access_key="AKIA_FASTAPI",
            secret_key="secret-fastapi",
            session_token="session-fastapi",
            expiration=_expiration(3600),
        )
    ) as (port, requests):
        _route_sts_to_loopback(monkeypatch, credentials_module, port)
        session = session_module.get_session()
        credentials = await session.get_credentials()
        assert credentials is not None

        async def request_using_shared_credentials():
            return await credentials.get_frozen_credentials()

        frozen = await asyncio.gather(
            *(request_using_shared_credentials() for _ in range(20))
        )

    assert len(requests) == 1
    assert {item.access_key for item in frozen} == {"AKIA_FASTAPI"}
    assert all(item.token == "session-fastapi" for item in frozen)
