"""Profile, loop, and cancellation isolation for Copilot token exchange."""

from __future__ import annotations

import asyncio
import gc
import json
import stat
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes_cli import copilot_auth
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@contextmanager
def _profile(home):
    Path(home).mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _response(token: str) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", copilot_auth._TOKEN_EXCHANGE_URL),
        json={"token": token, "expires_at": time.time() + 1800},
    )


def _client(*, side_effect=None, response=None):
    client = MagicMock()
    client.get = AsyncMock(side_effect=side_effect, return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture(autouse=True)
def _clear_caches():
    copilot_auth._jwt_cache.clear()
    copilot_auth._exchange_failure_cache.clear()
    yield
    copilot_auth._jwt_cache.clear()
    copilot_auth._exchange_failure_cache.clear()


@pytest.mark.asyncio
async def test_concurrent_same_profile_exchange_runs_once(tmp_path):
    client = _client(response=_response("api-once"))
    with _profile(tmp_path / "profile"), patch(
        "hermes_cli.copilot_auth._create_httpx_client",
        new=AsyncMock(return_value=client),
    ):
        first, second = await asyncio.gather(
            copilot_auth.exchange_copilot_token("gho_same"),
            copilot_auth.exchange_copilot_token("gho_same"),
        )

    assert first == second
    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_token_is_isolated_across_concurrent_profiles(tmp_path):
    clients = {
        "profile-a": _client(response=_response("api-profile-a")),
        "profile-b": _client(response=_response("api-profile-b")),
    }

    async def create_client(**_kwargs):
        return clients[get_hermes_home().name]

    async def exchange(home):
        with _profile(home):
            return await copilot_auth.exchange_copilot_token("gho_shared")

    with patch(
        "hermes_cli.copilot_auth._create_httpx_client",
        new=AsyncMock(side_effect=create_client),
    ):
        result_a, result_b = await asyncio.gather(
            exchange(tmp_path / "profile-a"),
            exchange(tmp_path / "profile-b"),
        )

    assert result_a[0] == "api-profile-a"
    assert result_b[0] == "api-profile-b"
    for profile, expected in (
        ("profile-a", "api-profile-a"),
        ("profile-b", "api-profile-b"),
    ):
        store = json.loads(
            (tmp_path / profile / copilot_auth._JWT_DISK_FILENAME).read_text()
        )
        assert {entry["api_token"] for entry in store.values()} == {expected}
        assert "gho_shared" not in repr(store)


@pytest.mark.asyncio
async def test_negative_cache_does_not_cross_profiles(tmp_path, monkeypatch):
    rejected = httpx.HTTPStatusError(
        "rejected",
        request=httpx.Request("GET", copilot_auth._TOKEN_EXCHANGE_URL),
        response=httpx.Response(403),
    )
    clients = [
        _client(side_effect=rejected),
        _client(response=_response("api-profile-b")),
    ]
    monkeypatch.setattr(copilot_auth.asyncio, "sleep", AsyncMock())
    with patch(
        "hermes_cli.copilot_auth._create_httpx_client",
        new=AsyncMock(side_effect=clients),
    ):
        with _profile(tmp_path / "profile-a"):
            with pytest.raises(ValueError):
                await copilot_auth.exchange_copilot_token("gho_shared")
        with _profile(tmp_path / "profile-b"):
            result = await copilot_auth.exchange_copilot_token("gho_shared")

    assert result[0] == "api-profile-b"


@pytest.mark.asyncio
async def test_symlink_profile_aliases_share_exchange_and_disk_lock(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_get(*_args, **_kwargs):
        started.set()
        await release.wait()
        return _response("api-alias")

    client = _client(side_effect=delayed_get)

    async def exchange(home):
        with _profile(home):
            return await copilot_auth.exchange_copilot_token("gho_alias")

    with patch(
        "hermes_cli.copilot_auth._create_httpx_client",
        new=AsyncMock(return_value=client),
    ):
        owner = asyncio.create_task(exchange(profile))
        await started.wait()
        sibling = asyncio.create_task(exchange(alias))
        await asyncio.sleep(0)
        release.set()
        assert await owner == await sibling

    client.get.assert_awaited_once()


def test_exchange_state_does_not_retain_closed_event_loop(tmp_path):
    async def run_exchange():
        with _profile(tmp_path / "profile"):
            client = _client(response=_response("api-loop"))
            with patch(
                "hermes_cli.copilot_auth._create_httpx_client",
                new=AsyncMock(return_value=client),
            ):
                return await asyncio.gather(
                    copilot_auth.exchange_copilot_token("gho_loop"),
                    copilot_auth.exchange_copilot_token("gho_loop"),
                )

    loop = asyncio.new_event_loop()
    loop_ref = weakref.ref(loop)
    try:
        first = loop.run_until_complete(run_exchange())
    finally:
        loop.close()
        del loop
    gc.collect()

    second = asyncio.run(run_exchange())
    assert first == second
    assert loop_ref() is None


@pytest.mark.asyncio
async def test_concurrent_disk_save_and_evict_preserve_unrelated_entry(tmp_path):
    with _profile(tmp_path / "profile"):
        expires = time.time() + 1800
        stale = copilot_auth._token_fingerprint("gho_stale")
        retained = copilot_auth._token_fingerprint("gho_retained")
        await copilot_auth._save_jwt_to_disk(stale, "api-stale", expires, None)
        await asyncio.gather(
            copilot_auth.evict_cached_exchanged_token("gho_stale"),
            copilot_auth._save_jwt_to_disk(
                retained, "api-retained", expires, None
            ),
        )
        store = await copilot_auth._read_jwt_store(
            tmp_path / "profile" / copilot_auth._JWT_DISK_FILENAME
        )

    assert store is not None
    assert stale not in store
    assert store[retained]["api_token"] == "api-retained"


@pytest.mark.asyncio
async def test_cancelled_owner_releases_exchange_and_leaves_no_secret_temp(tmp_path):
    started = asyncio.Event()
    blocker = asyncio.Event()

    async def blocked_get(*_args, **_kwargs):
        started.set()
        await blocker.wait()

    cancelled_client = _client(side_effect=blocked_get)
    success_client = _client(response=_response("api-retry"))
    home = tmp_path / "profile"
    with _profile(home), patch(
        "hermes_cli.copilot_auth._create_httpx_client",
        new=AsyncMock(side_effect=[cancelled_client, success_client]),
    ):
        task = asyncio.create_task(
            copilot_auth.exchange_copilot_token("gho_cancelled")
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await copilot_auth.exchange_copilot_token("gho_cancelled")

    assert result[0] == "api-retry"
    assert not list(home.glob(f"{copilot_auth._JWT_DISK_FILENAME}*.tmp"))
    cancelled_client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_cancellation_during_disk_replace_cleans_secret_temp(
    tmp_path, monkeypatch
):
    home = tmp_path / "profile"
    started = asyncio.Event()
    real_replace = copilot_auth.aiofiles.os.replace
    replace_calls = 0

    async def replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            started.set()
            await asyncio.Event().wait()
        return await real_replace(source, destination)

    monkeypatch.setattr(copilot_auth.aiofiles.os, "replace", replace)
    with _profile(home):
        task = asyncio.create_task(
            copilot_auth._save_jwt_to_disk(
                "fingerprint", "api-secret", time.time() + 1800, None
            )
        )
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not list(home.glob(f"{copilot_auth._JWT_DISK_FILENAME}*.tmp"))
        await copilot_auth._save_jwt_to_disk(
            "fingerprint", "api-retry", time.time() + 1800, None
        )
        store = await copilot_auth._read_jwt_store(
            home / copilot_auth._JWT_DISK_FILENAME
        )

    assert store is not None
    assert store["fingerprint"]["api_token"] == "api-retry"


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_owner_exchange(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_get(*_args, **_kwargs):
        started.set()
        await release.wait()
        return _response("api-owner")

    client = _client(side_effect=delayed_get)
    with _profile(tmp_path / "profile"), patch(
        "hermes_cli.copilot_auth._create_httpx_client",
        new=AsyncMock(return_value=client),
    ):
        owner = asyncio.create_task(
            copilot_auth.exchange_copilot_token("gho_waiter")
        )
        await started.wait()
        waiter = asyncio.create_task(
            copilot_auth.exchange_copilot_token("gho_waiter")
        )
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        result = await owner

    assert result[0] == "api-owner"
    client.get.assert_awaited_once()
    mode = stat.S_IMODE(
        (tmp_path / "profile" / copilot_auth._JWT_DISK_FILENAME).stat().st_mode
    )
    assert mode == 0o600
