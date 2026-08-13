"""Credential and event-loop isolation for retained model catalog caches."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import hermes_cli.models as models
from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _Response:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _Client:
    def __init__(self, get_result: Any):
        self._get_result = get_result

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False

    async def get(self, _url: str):
        if isinstance(self._get_result, BaseException):
            raise self._get_result
        if callable(self._get_result):
            return await self._get_result()
        return _Response(self._get_result)


@pytest.fixture(autouse=True)
def _clear_deepinfra_catalog_caches(monkeypatch):
    models._deepinfra_catalog_cache.clear()
    models._deepinfra_catalog_neg_cache.clear()
    monkeypatch.setenv("DEEPINFRA_BASE_URL", "https://catalog.deepinfra.test/v1")
    yield
    models._deepinfra_catalog_cache.clear()
    models._deepinfra_catalog_neg_cache.clear()


@pytest.mark.asyncio
async def test_deepinfra_catalog_cache_isolates_credentials_at_same_endpoint(
    monkeypatch,
    tmp_path,
):
    calls: list[str] = []
    both_started = asyncio.Event()
    starts = 0

    async def create_client(*, headers, **_kwargs):
        nonlocal starts
        authorization = headers.get("Authorization", "")
        calls.append(authorization)
        starts += 1
        if starts == 2:
            both_started.set()
        await both_started.wait()
        account = authorization.removeprefix("Bearer ")
        return _Client({"data": [{"id": f"model-for-{account}"}]})

    async def fetch_from(profile, api_key):
        home_token = set_hermes_home_override(profile)
        secret_token = set_secret_scope({
            "DEEPINFRA_API_KEY": api_key,
            "DEEPINFRA_BASE_URL": "https://catalog.deepinfra.test/v1",
        })
        try:
            assert models.deepinfra_base_url() == "https://catalog.deepinfra.test/v1"
            return await models._fetch_deepinfra_catalog()
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    monkeypatch.setattr(models, "_create_httpx_client", create_client)
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    account_a, account_b = await asyncio.gather(
        fetch_from(profile_a, "account-a-secret"),
        fetch_from(profile_b, "account-b-secret"),
    )
    assert account_a == [{"id": "model-for-account-a-secret"}]
    assert account_b == [{"id": "model-for-account-b-secret"}]
    assert await fetch_from(profile_a, "account-a-secret") == [
        {"id": "model-for-account-a-secret"}
    ]
    assert calls == ["Bearer account-a-secret", "Bearer account-b-secret"]
    assert len(models._deepinfra_catalog_cache) == 2
    assert all(
        "account-a-secret" not in repr(key) and "account-b-secret" not in repr(key)
        for key in models._deepinfra_catalog_cache
    )


@pytest.mark.asyncio
async def test_deepinfra_negative_cache_and_force_refresh_are_per_credential(
    monkeypatch,
):
    calls: list[str] = []
    account_a_fails = True

    async def create_client(*, headers, **_kwargs):
        nonlocal account_a_fails
        authorization = headers.get("Authorization", "")
        calls.append(authorization)
        if authorization == "Bearer account-a" and account_a_fails:
            return _Client(RuntimeError("account A catalog unavailable"))
        account = authorization.removeprefix("Bearer ")
        return _Client({"data": [{"id": f"model-for-{account}"}]})

    async def fetch(api_key, *, force_refresh=False):
        token = set_secret_scope({
            "DEEPINFRA_API_KEY": api_key,
            "DEEPINFRA_BASE_URL": "https://catalog.deepinfra.test/v1",
        })
        try:
            return await models._fetch_deepinfra_catalog(
                force_refresh=force_refresh
            )
        finally:
            reset_secret_scope(token)

    monkeypatch.setattr(models, "_create_httpx_client", create_client)
    assert await fetch("account-a") is None
    assert await fetch("account-a") is None

    assert await fetch("account-b") == [{"id": "model-for-account-b"}]

    account_a_fails = False
    assert await fetch("account-a", force_refresh=True) == [
        {"id": "model-for-account-a"}
    ]
    assert calls == [
        "Bearer account-a",
        "Bearer account-b",
        "Bearer account-a",
    ]
    assert models._deepinfra_catalog_neg_cache == {}


@pytest.mark.asyncio
async def test_deepinfra_negative_cache_retries_after_ttl(monkeypatch):
    now = 100.0
    calls = 0

    async def create_client(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Client(RuntimeError("temporary catalog failure"))
        return _Client({"data": [{"id": "recovered-model"}]})

    async def fetch():
        token = set_secret_scope({
            "DEEPINFRA_API_KEY": "ttl-account",
            "DEEPINFRA_BASE_URL": "https://catalog.deepinfra.test/v1",
        })
        try:
            return await models._fetch_deepinfra_catalog()
        finally:
            reset_secret_scope(token)

    monkeypatch.setattr(models, "_create_httpx_client", create_client)
    monkeypatch.setattr(models.time, "monotonic", lambda: now)

    assert await fetch() is None
    now += models._DEEPINFRA_CATALOG_NEG_TTL - 0.01
    assert await fetch() is None
    assert calls == 1

    now += 0.02
    assert await fetch() == [{"id": "recovered-model"}]
    assert calls == 2


@pytest.mark.asyncio
async def test_deepinfra_catalog_cancellation_does_not_seed_negative_cache(
    monkeypatch,
):
    started = asyncio.Event()

    async def pending_get():
        started.set()
        await asyncio.Future()

    async def create_client(**_kwargs):
        return _Client(pending_get)

    monkeypatch.setattr(models, "_create_httpx_client", create_client)
    token = set_secret_scope({
        "DEEPINFRA_API_KEY": "cancelled-account",
        "DEEPINFRA_BASE_URL": "https://catalog.deepinfra.test/v1",
    })
    try:
        task = asyncio.create_task(models._fetch_deepinfra_catalog())
    finally:
        reset_secret_scope(token)
    await started.wait()

    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert models._deepinfra_catalog_cache == {}
    assert models._deepinfra_catalog_neg_cache == {}


def test_deepinfra_cached_catalog_is_loop_neutral(monkeypatch):
    calls = 0

    async def create_client(**_kwargs):
        nonlocal calls
        calls += 1
        return _Client({"data": [{"id": "loop-neutral-model"}]})

    async def fetch():
        token = set_secret_scope({
            "DEEPINFRA_API_KEY": "same-account",
            "DEEPINFRA_BASE_URL": "https://catalog.deepinfra.test/v1",
        })
        try:
            return await models._fetch_deepinfra_catalog()
        finally:
            reset_secret_scope(token)

    monkeypatch.setattr(models, "_create_httpx_client", create_client)

    assert asyncio.run(fetch()) == [
        {"id": "loop-neutral-model"}
    ]
    assert asyncio.run(fetch()) == [
        {"id": "loop-neutral-model"}
    ]
    assert calls == 1
