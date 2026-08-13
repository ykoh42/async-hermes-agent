"""AWS credential and region isolation for the Bedrock async adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from agent import bedrock_adapter as bedrock
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


@dataclass
class _FrozenCredentials:
    access_key: str
    secret_key: str = "resolved-secret"
    token: str | None = None


@dataclass
class _Credentials:
    frozen: _FrozenCredentials

    async def get_frozen_credentials(self):
        return self.frozen


@dataclass
class _FakeClient:
    service: str
    region: str
    auth_label: str
    closed: int = 0

    async def converse(self, **_kwargs):
        return {
            "output": {"message": {"content": [{"text": self.auth_label}]}},
            "stopReason": "end_turn",
            "usage": {},
        }

    async def list_foundation_models(self):
        return {
            "modelSummaries": [
                {
                    "modelId": self.auth_label,
                    "modelName": self.auth_label,
                    "providerName": "Test",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                }
            ]
        }

    async def list_inference_profiles(self, **_kwargs):
        return {"inferenceProfileSummaries": []}


@dataclass
class _FakeManager:
    client: _FakeClient

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *_exc):
        self.client.closed += 1


@dataclass
class _FakeSession:
    sessions: list[_FakeSession]
    default_access_key: str = "DEFAULT-CHAIN"
    components: dict = field(default_factory=dict)
    create_calls: list[dict] = field(default_factory=list)
    clients: list[_FakeClient] = field(default_factory=list)
    credential_calls: int = 0

    def __post_init__(self):
        self.sessions.append(self)

    def register_component(self, name, component):
        self.components[name] = component

    async def get_credentials(self):
        self.credential_calls += 1
        return _Credentials(_FrozenCredentials(self.default_access_key))

    def create_client(self, service, **kwargs):
        self.create_calls.append({"service": service, **kwargs})
        access_key = kwargs.get("aws_access_key_id")
        if getattr(kwargs.get("config"), "signature_version", None) == "bearer":
            token_provider = self.components.get("token_provider")
            auth_token = (
                token_provider.load_token(signing_name="bedrock")
                if token_provider is not None
                else None
            )
            auth_label = (
                str(getattr(auth_token, "token", ""))
                if auth_token is not None
                else self.default_access_key
            )
        elif access_key:
            auth_label = str(access_key)
        else:
            auth_label = self.default_access_key
        client = _FakeClient(
            service=service,
            region=str(kwargs.get("region_name") or ""),
            auth_label=auth_label,
        )
        self.clients.append(client)
        return _FakeManager(client)


class _Owner:
    pass


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous = is_multiplex_active()
    token = set_secret_scope(None)
    set_multiplex_active(False)
    try:
        yield
    finally:
        set_multiplex_active(previous)
        reset_secret_scope(token)


@pytest.fixture
def fake_sessions(monkeypatch):
    sessions: list[_FakeSession] = []

    def get_session():
        return _FakeSession(sessions)

    monkeypatch.setattr(bedrock, "_aiobotocore_get_session", get_session)
    monkeypatch.setattr(bedrock, "_aiobotocore_import_error", None)
    return sessions


async def _under_profile(home, secrets, operation, *args):
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(secrets)
    try:
        return await operation(*args)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_concurrent_profiles_pass_static_credentials_and_regions_explicitly(
    tmp_path,
    monkeypatch,
    fake_sessions,
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FOREIGN-ACCESS")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "FOREIGN-SECRET")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "FOREIGN-SESSION")
    monkeypatch.setenv("AWS_REGION", "foreign-region-1")
    monkeypatch.setenv("AWS_PROFILE", "foreign-profile")
    set_multiplex_active(True)
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    owner_a = _Owner()
    owner_b = _Owner()
    secrets_a = {
        "AWS_ACCESS_KEY_ID": "PROFILE-A",
        "AWS_SECRET_ACCESS_KEY": "SECRET-A",
        "AWS_SESSION_TOKEN": "SESSION-A",
        "AWS_REGION": "eu-west-1",
    }
    secrets_b = {
        "AWS_ACCESS_KEY_ID": "PROFILE-B",
        "AWS_SECRET_ACCESS_KEY": "SECRET-B",
        "AWS_DEFAULT_REGION": "ap-southeast-2",
    }
    await asyncio.gather(
        _under_profile(
            profile_a,
            secrets_a,
            bedrock._retain_bedrock_lifecycle,
            owner_a,
        ),
        _under_profile(
            profile_b,
            secrets_b,
            bedrock._retain_bedrock_lifecycle,
            owner_b,
        ),
    )

    async def request(home, secrets):
        region = await _under_profile(home, secrets, bedrock.resolve_bedrock_region)
        return await _under_profile(
            home,
            secrets,
            bedrock.call_converse,
            region,
            "amazon.nova-micro-v1:0",
            [{"role": "user", "content": "hello"}],
        )

    response_a, response_b = await asyncio.gather(
        request(profile_a, secrets_a),
        request(profile_b, secrets_b),
    )
    response_a_again = await request(profile_a, secrets_a)

    assert response_a.choices[0].message.content == "PROFILE-A"
    assert response_a_again.choices[0].message.content == "PROFILE-A"
    assert response_b.choices[0].message.content == "PROFILE-B"
    assert len(fake_sessions) == 2
    calls = [session.create_calls[0] for session in fake_sessions]
    assert {
        (
            call["region_name"],
            call["aws_access_key_id"],
            call["aws_secret_access_key"],
            call.get("aws_session_token"),
            call["config"].signature_version,
        )
        for call in calls
    } == {
        ("eu-west-1", "PROFILE-A", "SECRET-A", "SESSION-A", "v4"),
        ("ap-southeast-2", "PROFILE-B", "SECRET-B", None, "v4"),
    }
    assert all(session.credential_calls == 0 for session in fake_sessions)

    await asyncio.gather(
        _under_profile(
            profile_a,
            secrets_a,
            bedrock._release_bedrock_lifecycle,
            owner_a,
        ),
        _under_profile(
            profile_b,
            secrets_b,
            bedrock._release_bedrock_lifecycle,
            owner_b,
        ),
    )


@pytest.mark.asyncio
async def test_concurrent_scoped_bearer_tokens_use_distinct_token_providers(
    tmp_path,
    monkeypatch,
    fake_sessions,
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "FOREIGN-BEARER")
    set_multiplex_active(True)

    async def discover(label):
        return await _under_profile(
            tmp_path / label,
            {
                "AWS_BEARER_TOKEN_BEDROCK": f"BEARER-{label}",
                "AWS_REGION": "us-east-1",
            },
            bedrock.discover_bedrock_models,
            "us-east-1",
        )

    models_a, models_b = await asyncio.gather(discover("A"), discover("B"))

    assert [model["id"] for model in models_a] == ["BEARER-A"]
    assert [model["id"] for model in models_b] == ["BEARER-B"]
    assert len(fake_sessions) == 2
    assert {
        (
            session.create_calls[0]["config"].signature_version,
            session.create_calls[0]["aws_access_key_id"],
            session.create_calls[0]["aws_secret_access_key"],
        )
        for session in fake_sessions
    } == {("bearer", "hermes-bedrock-bearer", "hermes-bedrock-bearer")}
    assert all(session.credential_calls == 0 for session in fake_sessions)


@pytest.mark.asyncio
async def test_multiplex_empty_scope_does_not_fall_back_to_foreign_process_chain(
    tmp_path,
    monkeypatch,
    fake_sessions,
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FOREIGN-ACCESS")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "FOREIGN-SECRET")
    monkeypatch.setenv("AWS_REGION", "foreign-region-1")
    set_multiplex_active(True)

    assert (
        await _under_profile(
            tmp_path,
            {},
            bedrock.resolve_aws_auth_env_var,
        )
        is None
    )
    assert (
        await _under_profile(
            tmp_path,
            {},
            bedrock.resolve_bedrock_region,
        )
        == "us-east-1"
    )
    with pytest.raises(RuntimeError, match="explicit profile-scoped AWS"):
        await _under_profile(
            tmp_path,
            {},
            bedrock.call_converse,
            "us-east-1",
            "amazon.nova-micro-v1:0",
            [{"role": "user", "content": "hello"}],
        )
    assert fake_sessions == []


@pytest.mark.asyncio
async def test_multiplex_unscoped_and_unsafe_chains_fail_closed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AWS_PROFILE", "foreign-profile")
    set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError):
        await bedrock.resolve_aws_auth_env_var()
    with pytest.raises(UnscopedSecretError):
        await bedrock.resolve_bedrock_region()
    with pytest.raises(RuntimeError, match="AWS_PROFILE"):
        await _under_profile(
            tmp_path,
            {"AWS_PROFILE": "profile-a"},
            bedrock.resolve_aws_auth_env_var,
        )
    with pytest.raises(RuntimeError, match="AWS_CONTAINER_CREDENTIALS"):
        await _under_profile(
            tmp_path,
            {"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials"},
            bedrock.has_aws_credentials,
        )
    with pytest.raises(RuntimeError, match="configured together"):
        await _under_profile(
            tmp_path,
            {"AWS_ACCESS_KEY_ID": "PARTIAL"},
            bedrock.call_converse,
            "us-east-1",
            "amazon.nova-micro-v1:0",
            [{"role": "user", "content": "hello"}],
        )


@pytest.mark.asyncio
async def test_single_profile_keeps_default_aws_chain_and_public_env_behavior(
    tmp_path,
    monkeypatch,
    fake_sessions,
):
    monkeypatch.setenv("AWS_PROFILE", "developer")
    monkeypatch.setenv("AWS_REGION", "ca-central-1")

    assert await bedrock.resolve_aws_auth_env_var() == "AWS_PROFILE"
    assert await bedrock.resolve_bedrock_region() == "ca-central-1"
    response = await _under_profile(
        tmp_path,
        None,
        bedrock.call_converse,
        "ca-central-1",
        "amazon.nova-micro-v1:0",
        [{"role": "user", "content": "hello"}],
    )

    assert response.choices[0].message.content == "DEFAULT-CHAIN"
    assert len(fake_sessions) == 1
    assert fake_sessions[0].credential_calls == 1
    assert fake_sessions[0].create_calls == [
        {"service": "bedrock-runtime", "region_name": "ca-central-1"}
    ]


@pytest.mark.asyncio
async def test_scoped_request_repeated_cancellation_closes_owned_client(
    tmp_path,
    monkeypatch,
    fake_sessions,
):
    request_started = asyncio.Event()
    allow_request = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    original_converse = _FakeClient.converse
    original_exit = _FakeManager.__aexit__

    async def delayed_converse(client, **kwargs):
        request_started.set()
        await allow_request.wait()
        return await original_converse(client, **kwargs)

    async def delayed_exit(manager, *exc):
        close_started.set()
        await allow_close.wait()
        return await original_exit(manager, *exc)

    monkeypatch.setattr(_FakeClient, "converse", delayed_converse)
    monkeypatch.setattr(_FakeManager, "__aexit__", delayed_exit)
    set_multiplex_active(True)
    task = asyncio.create_task(
        _under_profile(
            tmp_path,
            {
                "AWS_ACCESS_KEY_ID": "PROFILE-A",
                "AWS_SECRET_ACCESS_KEY": "SECRET-A",
            },
            bedrock.call_converse,
            "us-east-1",
            "amazon.nova-micro-v1:0",
            [{"role": "user", "content": "hello"}],
        )
    )
    await request_started.wait()
    task.cancel("first")
    await close_started.wait()
    task.cancel("second")
    allow_request.set()
    allow_close.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert cancelled.value.args == ("first",)
    assert len(fake_sessions) == 1
    assert fake_sessions[0].credential_calls == 0
    assert [client.closed for client in fake_sessions[0].clients] == [1]
    state = await _under_profile(
        tmp_path,
        {
            "AWS_ACCESS_KEY_ID": "PROFILE-A",
            "AWS_SECRET_ACCESS_KEY": "SECRET-A",
        },
        bedrock._activate_bedrock_scope,
    )
    assert not state.clients
    assert state.lock is None
    assert state.lock_users == 0
    assert not [
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
        and pending.get_name().startswith("bedrock-")
        and not pending.done()
    ]
