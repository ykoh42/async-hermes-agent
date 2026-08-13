"""Profile, credential, loop, and lifecycle isolation for Bedrock clients."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import os
import weakref
from dataclasses import dataclass, field

import pytest

from agent import bedrock_adapter as bedrock
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


_credential_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bedrock_test_credential", default="default"
)


@dataclass
class _FakeClient:
    home: str
    credential: str
    service: str
    close_started: asyncio.Event | None = None
    allow_close: asyncio.Event | None = None
    closed: int = 0
    foundation_calls: int = 0
    discovery_started: asyncio.Event | None = None
    allow_discovery: asyncio.Event | None = None
    request_started: asyncio.Event | None = None
    allow_request: asyncio.Event | None = None
    enter_started: asyncio.Event | None = None
    allow_enter: asyncio.Event | None = None

    async def converse(self, **_kwargs):
        if self.request_started is not None:
            self.request_started.set()
        if self.allow_request is not None:
            await self.allow_request.wait()
        return {
            "output": {"message": {"content": [{"text": self.home}]}},
            "stopReason": "end_turn",
            "usage": {},
        }

    async def list_foundation_models(self):
        self.foundation_calls += 1
        if self.discovery_started is not None:
            self.discovery_started.set()
        if self.allow_discovery is not None:
            await self.allow_discovery.wait()
        label = f"{self.home}:{self.credential}"
        return {
            "modelSummaries": [
                {
                    "modelId": label,
                    "modelName": label,
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
        if self.client.enter_started is not None:
            self.client.enter_started.set()
        if self.client.allow_enter is not None:
            await self.client.allow_enter.wait()
        return self.client

    async def __aexit__(self, *_exc):
        if self.client.close_started is not None:
            self.client.close_started.set()
        if self.client.allow_close is not None:
            await self.client.allow_close.wait()
        self.client.closed += 1


@dataclass
class _FakeSession:
    home: str
    credential: str
    built: list[_FakeClient]
    client_options: dict = field(default_factory=dict)

    async def get_credentials(self):
        return None

    def create_client(self, service, *, region_name):
        del region_name
        client = _FakeClient(
            self.home,
            self.credential,
            service,
            **self.client_options,
        )
        self.built.append(client)
        return _FakeManager(client)


class _Owner:
    pass


async def _under_profile(home, operation, *args, credential="default", **kwargs):
    home_token = set_hermes_home_override(home)
    credential_token = _credential_scope.set(credential)
    try:
        return await operation(*args, **kwargs)
    finally:
        _credential_scope.reset(credential_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def fake_sdk(monkeypatch):
    built: list[_FakeClient] = []
    options: dict = {"_identity_calls": 0}

    def get_session():
        return _FakeSession(
            str(get_hermes_home()),
            _credential_scope.get(),
            built,
            {
                key: value
                for key, value in options.items()
                if not key.startswith("_")
            },
        )

    async def identity(session, _environment_identity=None):
        options["_identity_calls"] += 1
        return (("test-credential", session.credential),)

    monkeypatch.setattr(bedrock, "_aiobotocore_get_session", get_session)
    monkeypatch.setattr(bedrock, "_aiobotocore_import_error", None)
    monkeypatch.setattr(
        bedrock,
        "_aws_environment_identity",
        lambda: (("test-environment", _credential_scope.get()),),
    )
    monkeypatch.setattr(bedrock, "_aws_credential_identity", identity)
    return built, options


@pytest.mark.asyncio
async def test_credential_identity_never_retains_secret_text(monkeypatch):
    class Frozen:
        access_key = "RESOLVED-ACCESS-SECRET"

    class Credentials:
        async def get_frozen_credentials(self):
            return Frozen()

    class Session:
        async def get_credentials(self):
            return Credentials()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV-ACCESS-SECRET")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV-KEY-SECRET")
    identity = await bedrock._aws_credential_identity(Session())
    rendered = repr(identity)
    assert "ENV-ACCESS-SECRET" not in rendered
    assert "ENV-KEY-SECRET" not in rendered
    assert "RESOLVED-ACCESS-SECRET" not in rendered


@pytest.mark.asyncio
async def test_cancelled_lock_waiter_releases_loop_bound_state(tmp_path):
    state = await _under_profile(tmp_path / "profile", bedrock._activate_bedrock_scope)
    holder = bedrock._BedrockStateLock(state)
    await holder.__aenter__()

    async def wait_for_lock():
        async with bedrock._BedrockStateLock(state):
            pass

    waiter = asyncio.create_task(wait_for_lock())
    await asyncio.sleep(0)
    assert state.lock_users == 2
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert state.lock_users == 1
    await holder.__aexit__(None, None, None)
    assert state.lock_users == 0
    assert state.lock is None


@pytest.mark.asyncio
async def test_concurrent_profiles_and_credentials_never_share_clients(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    owner_a = _Owner()
    owner_b = _Owner()
    await asyncio.gather(
        _under_profile(profile_a, bedrock._retain_bedrock_lifecycle, owner_a),
        _under_profile(profile_b, bedrock._retain_bedrock_lifecycle, owner_b),
    )

    async def acquire(home, credential):
        async def operation():
            async with bedrock._get_bedrock_runtime_client("us-east-1") as client:
                return client

        return await _under_profile(home, operation, credential=credential)

    client_a, client_a_sibling, client_b = await asyncio.gather(
        acquire(profile_a, "credential-a"),
        acquire(profile_a, "credential-a"),
        acquire(profile_b, "credential-a"),
    )
    identity_calls = options["_identity_calls"]
    client_a_again = await acquire(profile_a, "credential-a")
    client_other_credential = await acquire(profile_a, "credential-b")

    assert client_a is client_a_sibling is client_a_again
    assert options["_identity_calls"] == identity_calls + 1
    assert client_a is not client_b
    assert client_a is not client_other_credential
    assert len(built) == 3
    await _under_profile(profile_a, bedrock._release_bedrock_lifecycle, owner_a)
    await _under_profile(profile_b, bedrock._release_bedrock_lifecycle, owner_b)
    assert all(client.closed == 1 for client in built)


@pytest.mark.asyncio
async def test_sibling_owner_release_uses_retained_profile_and_closes_once(
    tmp_path, fake_sdk
):
    built, _options = fake_sdk
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    first = _Owner()
    sibling = _Owner()
    await _under_profile(profile_a, bedrock._retain_bedrock_lifecycle, first)
    await _under_profile(profile_a, bedrock._retain_bedrock_lifecycle, sibling)

    async def acquire():
        async with bedrock._get_bedrock_runtime_client("us-east-1") as client:
            return client

    client = await _under_profile(profile_a, acquire)
    await _under_profile(profile_b, bedrock._release_bedrock_lifecycle, first)
    assert client.closed == 0
    await _under_profile(profile_b, bedrock._release_bedrock_lifecycle, sibling)
    assert client.closed == 1
    assert len(built) == 1


@pytest.mark.asyncio
async def test_canonical_profile_alias_reuses_one_client(tmp_path, fake_sdk):
    built, _options = fake_sdk
    profile = tmp_path / "profile"
    alias = tmp_path / "alias"
    profile.mkdir()
    os.symlink(profile, alias, target_is_directory=True)
    owner = _Owner()
    await _under_profile(profile, bedrock._retain_bedrock_lifecycle, owner)

    async def acquire():
        async with bedrock._get_bedrock_runtime_client("us-east-1") as client:
            return client

    first = await _under_profile(profile, acquire)
    second = await _under_profile(alias, acquire)
    assert first is second
    assert len(built) == 1
    await _under_profile(alias, bedrock._release_bedrock_lifecycle, owner)
    assert first.closed == 1


@pytest.mark.asyncio
async def test_final_release_finishes_client_close_through_repeated_cancel(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    profile = tmp_path / "profile"
    owner = _Owner()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    options.update(close_started=close_started, allow_close=allow_close)
    await _under_profile(profile, bedrock._retain_bedrock_lifecycle, owner)

    async def acquire():
        async with bedrock._get_bedrock_runtime_client("us-east-1"):
            pass

    await _under_profile(profile, acquire)
    release = asyncio.create_task(
        _under_profile(profile, bedrock._release_bedrock_lifecycle, owner)
    )
    await close_started.wait()
    release.cancel()
    await asyncio.sleep(0)
    release.cancel()
    await asyncio.sleep(0)
    assert not release.done()
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert built[0].closed == 1


@pytest.mark.asyncio
async def test_request_cancellation_closes_unleased_client_through_repeated_cancel(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    request_started = asyncio.Event()
    allow_request = asyncio.Event()
    options.update(request_started=request_started, allow_request=allow_request)
    profile = tmp_path / "profile"
    task = asyncio.create_task(
        _under_profile(
            profile,
            bedrock.call_converse,
            "us-east-1",
            "test-model",
            [{"role": "user", "content": "hi"}],
        )
    )
    await request_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow_request.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert built[0].closed == 1


@pytest.mark.asyncio
async def test_client_creation_cancellation_rolls_back_partial_manager(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    enter_started = asyncio.Event()
    allow_enter = asyncio.Event()
    options.update(enter_started=enter_started, allow_enter=allow_enter)
    profile = tmp_path / "profile"

    async def acquire():
        async with bedrock._get_bedrock_runtime_client("us-east-1"):
            pass

    task = asyncio.create_task(_under_profile(profile, acquire))
    await enter_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    allow_enter.set()
    assert built[0].closed == 1
    state = await _under_profile(profile, bedrock._activate_bedrock_scope)
    assert not state.clients
    assert state.lock is None
    assert state.lock_users == 0


@pytest.mark.asyncio
async def test_reset_closes_only_active_profile_clients(tmp_path, fake_sdk):
    built, _options = fake_sdk
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    owner_a = _Owner()
    owner_b = _Owner()
    await _under_profile(profile_a, bedrock._retain_bedrock_lifecycle, owner_a)
    await _under_profile(profile_b, bedrock._retain_bedrock_lifecycle, owner_b)

    async def acquire():
        async with bedrock._get_bedrock_runtime_client("us-east-1") as client:
            return client

    client_a = await _under_profile(profile_a, acquire)
    client_b = await _under_profile(profile_b, acquire)
    await _under_profile(profile_a, bedrock.reset_client_cache)
    assert client_a.closed == 1
    assert client_b.closed == 0
    assert await _under_profile(profile_b, acquire) is client_b
    await _under_profile(profile_a, bedrock._release_bedrock_lifecycle, owner_a)
    await _under_profile(profile_b, bedrock._release_bedrock_lifecycle, owner_b)
    assert client_b.closed == 1
    assert len(built) == 2


@pytest.mark.asyncio
async def test_discovery_cache_is_profile_credential_and_filter_scoped(
    tmp_path, fake_sdk
):
    built, _options = fake_sdk
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    models_a, models_a_sibling, models_b = await asyncio.gather(
        _under_profile(
            profile_a,
            bedrock.discover_bedrock_models,
            "us-east-1",
            credential="credential-a",
        ),
        _under_profile(
            profile_a,
            bedrock.discover_bedrock_models,
            "us-east-1",
            credential="credential-a",
        ),
        _under_profile(
            profile_b,
            bedrock.discover_bedrock_models,
            "us-east-1",
            credential="credential-a",
        ),
    )
    assert models_a_sibling == models_a
    cached_a = await _under_profile(
        profile_a,
        bedrock.discover_bedrock_models,
        "us-east-1",
        credential="credential-a",
    )
    other_credential = await _under_profile(
        profile_a,
        bedrock.discover_bedrock_models,
        "us-east-1",
        credential="credential-b",
    )
    filtered = await _under_profile(
        profile_a,
        bedrock.discover_bedrock_models,
        "us-east-1",
        ["test"],
        credential="credential-a",
    )
    assert models_a == cached_a
    assert models_a != models_b
    assert models_a != other_credential
    assert filtered == models_a
    assert len(built) == 4


@pytest.mark.asyncio
async def test_cancelled_discovery_closes_client_and_leaves_no_task(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    started = asyncio.Event()
    allow = asyncio.Event()
    options.update(discovery_started=started, allow_discovery=allow)
    profile = tmp_path / "profile"
    task = asyncio.create_task(
        _under_profile(
            profile,
            bedrock.discover_bedrock_models,
            "us-east-1",
        )
    )
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert built[0].closed == 1
    state = await _under_profile(profile, bedrock._activate_bedrock_scope)
    assert not state.discovery_tasks
    assert not state.discovery_waiters


@pytest.mark.asyncio
async def test_cancelled_discovery_waiter_does_not_cancel_sibling(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    started = asyncio.Event()
    allow = asyncio.Event()
    options.update(discovery_started=started, allow_discovery=allow)
    profile = tmp_path / "profile"

    def discover():
        return asyncio.create_task(
            _under_profile(
                profile,
                bedrock.discover_bedrock_models,
                "us-east-1",
            )
        )

    first = discover()
    sibling = discover()
    await started.wait()
    state = await _under_profile(profile, bedrock._activate_bedrock_scope)
    # The discovery-start event only proves that the shared operation began.
    # Profile canonicalization is independent async file I/O, so wait until
    # both callers have registered before exercising the waiter invariant.
    async with asyncio.timeout(5):
        while sum(state.discovery_waiters.values()) < 2:
            await asyncio.sleep(0)
    first.cancel()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    models = await sibling
    assert len(models) == 1
    assert len(built) == 1
    assert built[0].closed == 1
    assert not state.discovery_tasks
    assert not state.discovery_waiters


@pytest.mark.asyncio
async def test_discovery_reset_cancels_inflight_work_without_retry(
    tmp_path, fake_sdk
):
    built, options = fake_sdk
    started = asyncio.Event()
    allow = asyncio.Event()
    options.update(discovery_started=started, allow_discovery=allow)
    profile = tmp_path / "profile"
    discovery = asyncio.create_task(
        _under_profile(
            profile,
            bedrock.discover_bedrock_models,
            "us-east-1",
        )
    )
    await started.wait()
    await _under_profile(profile, bedrock.reset_discovery_cache)
    with pytest.raises(asyncio.CancelledError):
        await discovery
    allow.set()
    assert len(built) == 1
    assert built[0].closed == 1
    state = await _under_profile(profile, bedrock._activate_bedrock_scope)
    assert not state.discovery_tasks
    assert not state.discovery_waiters


def test_sequential_event_loops_close_clients_and_are_collectable(
    tmp_path, fake_sdk
):
    built, _options = fake_sdk
    loop_refs = []

    async def cycle(home):
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        owner = _Owner()
        await _under_profile(home, bedrock._retain_bedrock_lifecycle, owner)

        async def acquire():
            async with bedrock._get_bedrock_runtime_client("us-east-1"):
                pass

        await _under_profile(home, acquire)
        await _under_profile(home, bedrock._release_bedrock_lifecycle, owner)

    asyncio.run(cycle(tmp_path / "one"))
    asyncio.run(cycle(tmp_path / "two"))
    gc.collect()
    assert [reference() for reference in loop_refs] == [None, None]
    assert [client.closed for client in built] == [1, 1]
