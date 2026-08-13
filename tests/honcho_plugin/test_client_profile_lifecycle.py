"""Profile and lifecycle isolation for the retained Honcho client."""

from __future__ import annotations

import asyncio
import gc
import json
import queue
import threading
import types
import weakref
from unittest.mock import AsyncMock, patch

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import client as client_module
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
    resolve_config_path,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_source", ["default", "global"])
async def test_multiplex_profiles_ignore_foreign_legacy_honcho_config(
    tmp_path,
    monkeypatch,
    legacy_source,
):
    default_home = tmp_path / "default-home"
    global_path = tmp_path / "global" / "config.json"
    default_path = default_home / "honcho.json"
    poison_path = default_path if legacy_source == "default" else global_path
    poison_path.parent.mkdir(parents=True)
    poison_path.write_text(
        json.dumps(
            {
                "apiKey": "foreign-file-key",
                "workspace": "foreign-workspace",
                "aiPeer": "foreign-peer",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        client_module,
        "get_default_hermes_root",
        AsyncMock(return_value=default_home),
    )
    monkeypatch.setattr(
        client_module,
        "resolve_global_config_path",
        lambda: global_path,
    )
    monkeypatch.setenv("HONCHO_API_KEY", "foreign-process-key")

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def resolve(name: str):
        profile_home = tmp_path / f"profile-{name}"
        home_token = set_hermes_home_override(profile_home)
        secret_token = set_secret_scope(
            {
                "HERMES_HONCHO_HOST": f"host-{name}",
                "HONCHO_API_KEY": f"key-{name}",
            }
        )
        try:
            return (
                await resolve_config_path(),
                await HonchoClientConfig.from_global_config(),
            )
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    try:
        (path_a, config_a), (path_b, config_b) = await asyncio.gather(
            resolve("alpha"),
            resolve("beta"),
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert path_a == tmp_path / "profile-alpha" / "honcho.json"
    assert path_b == tmp_path / "profile-beta" / "honcho.json"
    assert (config_a.api_key, config_a.workspace_id, config_a.ai_peer) == (
        "key-alpha",
        "hermes",
        "host-alpha",
    )
    assert (config_b.api_key, config_b.workspace_id, config_b.ai_peer) == (
        "key-beta",
        "hermes",
        "host-beta",
    )


@pytest.mark.asyncio
async def test_multiplex_missing_local_config_preserves_scoped_empty_and_fails_closed(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "profile"
    global_path = tmp_path / "global" / "config.json"
    global_path.parent.mkdir(parents=True)
    global_path.write_text(
        json.dumps({"apiKey": "foreign-file-key"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        client_module,
        "resolve_global_config_path",
        lambda: global_path,
    )
    monkeypatch.setenv("HONCHO_API_KEY", "foreign-process-key")

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(profile_home)
    try:
        secret_token = set_secret_scope(
            {"HERMES_HONCHO_HOST": "host-empty", "HONCHO_API_KEY": ""}
        )
        try:
            config = await HonchoClientConfig.from_global_config()
        finally:
            reset_secret_scope(secret_token)

        with pytest.raises(UnscopedSecretError):
            await HonchoClientConfig.from_global_config(host="host-unscoped")
    finally:
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    assert config.host == "host-empty"
    assert config.api_key == ""
    assert config.enabled is False


@pytest.mark.asyncio
async def test_single_profile_keeps_global_fallback_and_explicit_path_parity(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "profile"
    default_home = tmp_path / "default"
    global_path = tmp_path / "global" / "config.json"
    global_path.parent.mkdir(parents=True)
    global_path.write_text(
        json.dumps({"apiKey": "global-key", "workspace": "global-workspace"}),
        encoding="utf-8",
    )
    explicit_path = profile_home / "explicit.json"
    explicit_path.parent.mkdir(parents=True)
    explicit_path.write_text(
        json.dumps({"apiKey": "explicit-key", "workspace": "explicit-workspace"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        client_module,
        "get_default_hermes_root",
        AsyncMock(return_value=default_home),
    )
    monkeypatch.setattr(
        client_module,
        "resolve_global_config_path",
        lambda: global_path,
    )

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(False)
    home_token = set_hermes_home_override(profile_home)
    try:
        assert await resolve_config_path() == global_path
        global_config = await HonchoClientConfig.from_global_config(host="hermes")

        set_multiplex_active(True)
        secret_token = set_secret_scope({})
        try:
            explicit_config = await HonchoClientConfig.from_global_config(
                host="hermes",
                config_path=explicit_path,
            )
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    assert (global_config.api_key, global_config.workspace_id) == (
        "global-key",
        "global-workspace",
    )
    assert (explicit_config.api_key, explicit_config.workspace_id) == (
        "explicit-key",
        "explicit-workspace",
    )


@pytest.mark.asyncio
async def test_multiplex_oauth_refresh_never_reads_or_rewrites_global_config(
    tmp_path,
    monkeypatch,
):
    from plugins.memory.honcho import oauth

    profile_home = tmp_path / "profile"
    global_path = tmp_path / "global" / "config.json"
    global_path.parent.mkdir(parents=True)
    global_path.write_text(
        json.dumps(
            {
                "hosts": {
                    "host-alpha": {
                        "apiKey": "hch-at-foreign",
                        "oauth": {
                            "refreshToken": "hch-rt-foreign",
                            "expiresAt": 0,
                            "clientId": "foreign-client",
                            "tokenEndpoint": "https://foreign.invalid/token",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    original_global = global_path.read_bytes()
    monkeypatch.setattr(
        client_module,
        "resolve_global_config_path",
        lambda: global_path,
    )
    post_form = AsyncMock(
        side_effect=AssertionError("foreign OAuth grant must not be refreshed")
    )
    monkeypatch.setattr(oauth, "_http_post_form", post_form)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope(
        {"HERMES_HONCHO_HOST": "host-alpha", "HONCHO_API_KEY": "scoped-key"}
    )
    try:
        config = await HonchoClientConfig.from_global_config()
        await client_module._apply_fresh_oauth_token(config)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    assert config.api_key == "scoped-key"
    assert global_path.read_bytes() == original_global
    post_form.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiplex_oauth_refresh_reads_and_rewrites_only_profile_config(
    tmp_path,
    monkeypatch,
):
    from plugins.memory.honcho import oauth

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    profile_path = profile_home / "honcho.json"
    profile_path.write_text(
        json.dumps(
            {
                "hosts": {
                    "host-alpha": {
                        "apiKey": "hch-at-old",
                        "oauth": {
                            "refreshToken": "hch-rt-old",
                            "expiresAt": 0,
                            "clientId": "profile-client",
                            "tokenEndpoint": "https://profile.example/token",
                            "scope": "write",
                            "tokenType": "Bearer",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def refresh(_url, data, _timeout):
        assert data["refresh_token"] == "hch-rt-old"
        return {
            "access_token": "hch-at-new",
            "refresh_token": "hch-rt-new",
            "expires_in": 3600,
            "scope": "write",
            "token_type": "Bearer",
        }

    monkeypatch.setattr(oauth, "_http_post_form", refresh)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope({})
    config = HonchoClientConfig(host="host-alpha", api_key="hch-at-old")
    try:
        await client_module._apply_fresh_oauth_token(config)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    saved_host = json.loads(profile_path.read_text(encoding="utf-8"))["hosts"][
        "host-alpha"
    ]
    assert config.api_key == "hch-at-new"
    assert saved_host["apiKey"] == "hch-at-new"
    assert saved_host["oauth"]["refreshToken"] == "hch-rt-new"


@pytest.mark.asyncio
async def test_oauth_refresh_cancellation_propagates_from_profile_local_path(
    tmp_path,
    monkeypatch,
):
    from plugins.memory.honcho import oauth

    profile_home = tmp_path / "profile"
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope({})
    ensure_fresh_token = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(oauth, "ensure_fresh_token", ensure_fresh_token)
    try:
        with pytest.raises(asyncio.CancelledError):
            await client_module._apply_fresh_oauth_token(
                HonchoClientConfig(host="hermes", api_key="hch-at-old")
            )
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    ensure_fresh_token.assert_awaited_once_with(
        profile_home / "honcho.json",
        "hermes",
    )


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_all_honcho_environment_settings(
    monkeypatch,
):
    for key, value in {
        "HERMES_HONCHO_HOST": "process-host",
        "HONCHO_API_KEY": "process-key",
        "HONCHO_BASE_URL": "https://process.invalid",
        "HONCHO_TIMEOUT": "999",
        "HONCHO_ENVIRONMENT": "process-environment",
    }.items():
        monkeypatch.setenv(key, value)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def resolve(name: str, timeout: int):
        token = set_secret_scope(
            {
                "HERMES_HONCHO_HOST": f"host-{name}",
                "HONCHO_API_KEY": f"key-{name}",
                "HONCHO_BASE_URL": f"https://{name}.example",
                "HONCHO_TIMEOUT": str(timeout),
                "HONCHO_ENVIRONMENT": f"environment-{name}",
            }
        )
        try:
            config = await HonchoClientConfig.from_env()
            return await client_module.resolve_active_host(), config
        finally:
            reset_secret_scope(token)

    try:
        (host_a, config_a), (host_b, config_b) = await asyncio.gather(
            resolve("alpha", 111), resolve("beta", 222)
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert (
        host_a,
        config_a.host,
        config_a.api_key,
        config_a.base_url,
        config_a.timeout,
        config_a.environment,
    ) == (
        "host-alpha",
        "host-alpha",
        "key-alpha",
        "https://alpha.example",
        111.0,
        "environment-alpha",
    )
    assert (
        host_b,
        config_b.host,
        config_b.api_key,
        config_b.base_url,
        config_b.timeout,
        config_b.environment,
    ) == (
        "host-beta",
        "host-beta",
        "key-beta",
        "https://beta.example",
        222.0,
        "environment-beta",
    )


async def _profile_client(home, api_key: str):
    token = set_hermes_home_override(home)
    try:
        config_path = home / "honcho.json"
        config = await HonchoClientConfig.from_global_config(
            host="hermes",
            config_path=config_path,
        )
        assert config.api_key == api_key
        return await get_honcho_client()
    finally:
        reset_hermes_home_override(token)


async def _reset_profile(home) -> None:
    token = set_hermes_home_override(home)
    try:
        await reset_honcho_client()
    finally:
        reset_hermes_home_override(token)


def _write_profile(home, api_key: str) -> None:
    home.mkdir(parents=True)
    (home / "honcho.json").write_text(
        json.dumps({"apiKey": api_key, "enabled": True}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_concurrent_profiles_do_not_share_credentials_or_clients(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_profile(home_a, "key-a")
    _write_profile(home_b, "key-b")

    try:
        client_a, client_b = await asyncio.gather(
            _profile_client(home_a, "key-a"),
            _profile_client(home_b, "key-b"),
        )
        assert client_a is not client_b
        assert client_a._async_http.api_key == "key-a"
        assert client_b._async_http.api_key == "key-b"
        assert await _profile_client(home_a, "key-a") is client_a
        assert await _profile_client(home_b, "key-b") is client_b
    finally:
        await _reset_profile(home_a)
        await _reset_profile(home_b)


@pytest.mark.asyncio
async def test_symlink_alias_uses_same_profile_client(tmp_path):
    home = tmp_path / "profile"
    alias = tmp_path / "profile-alias"
    _write_profile(home, "key")
    alias.symlink_to(home, target_is_directory=True)

    try:
        first = await _profile_client(home, "key")
        second = await _profile_client(alias, "key")
        assert second is first
    finally:
        await _reset_profile(home)


def test_sequential_event_loops_do_not_reuse_or_retain_clients(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home, "key")
    loop_refs: list[weakref.ReferenceType[asyncio.AbstractEventLoop]] = []
    clients = []

    async def run_once():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        client = await _profile_client(home, "key")
        clients.append(client)
        await _reset_profile(home)

    asyncio.run(run_once())
    asyncio.run(run_once())
    gc.collect()

    assert clients[0] is not clients[1]
    assert loop_refs[0]() is None


def test_concurrent_profiles_on_different_event_loops_are_isolated(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_profile(home_a, "key-a")
    _write_profile(home_b, "key-b")
    ready_a = threading.Event()
    ready_b = threading.Event()
    release = threading.Event()
    results: queue.Queue[tuple[object | None, str | None, BaseException | None]] = (
        queue.Queue()
    )

    async def run_profile(home, key, ready):
        ready.set()
        while not release.is_set():
            await asyncio.sleep(0)
        client = await _profile_client(home, key)
        try:
            results.put((client, client._async_http.api_key, None))
        finally:
            await _reset_profile(home)

    def run(home, key, ready) -> None:
        try:
            asyncio.run(run_profile(home, key, ready))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            results.put((None, None, exc))

    thread_a = threading.Thread(target=run, args=(home_a, "key-a", ready_a))
    thread_b = threading.Thread(target=run, args=(home_b, "key-b", ready_b))
    thread_a.start()
    thread_b.start()
    assert ready_a.wait(timeout=2)
    assert ready_b.wait(timeout=2)
    release.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)

    first = results.get_nowait()
    second = results.get_nowait()
    assert first[2] is second[2] is None
    assert first[0] is not second[0]
    assert {first[1], second[1]} == {"key-a", "key-b"}


@pytest.mark.asyncio
async def test_sibling_provider_shutdown_releases_only_last_consumer(tmp_path):
    home = tmp_path / "profile"
    other_home = tmp_path / "other-profile"
    first = HonchoMemoryProvider()
    second = HonchoMemoryProvider()
    config = HonchoClientConfig(enabled=False)

    token = set_hermes_home_override(home)
    try:
        with patch.object(
            HonchoClientConfig,
            "from_global_config",
            new=AsyncMock(return_value=config),
        ):
            await first.initialize("first")
            await second.initialize("second")
        state = await client_module._activate_honcho_client_state()
        transport = types.SimpleNamespace(close=AsyncMock())
        fake_client = types.SimpleNamespace(_async_http=transport)

        async def build():
            return fake_client

        await state.slot.get(build)
    finally:
        reset_hermes_home_override(token)

    # A caller may close an owned provider while another request profile is active.
    token = set_hermes_home_override(other_home)
    try:
        await first.shutdown()
        assert state.slot.peek() is fake_client
        transport.close.assert_not_awaited()

        await second.shutdown()
        assert state.slot.peek() is None
        transport.close.assert_awaited_once()
        await second.shutdown()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_initialize_failure_remains_fail_open_and_shutdown_releases_owner(
    tmp_path,
):
    provider = HonchoMemoryProvider()
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        with patch.object(
            HonchoClientConfig,
            "from_global_config",
            new=AsyncMock(side_effect=RuntimeError("bad config")),
        ):
            await provider.initialize("session")

        assert provider._manager is None
        assert provider in client_module._honcho_client_owners
        await provider.shutdown()
        assert provider not in client_module._honcho_client_owners
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_reset_finishes_transport_close_before_repeated_cancellation(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    close_calls = 0

    class Transport:
        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            started.set()
            await release.wait()

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        state = await client_module._activate_honcho_client_state()

        async def build():
            return types.SimpleNamespace(_async_http=Transport())

        await state.slot.get(build)
        reset_task = asyncio.create_task(reset_honcho_client())
        await started.wait()
        reset_task.cancel()
        await asyncio.sleep(0)
        reset_task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await reset_task
        assert close_calls == 1
        assert state.slot.peek() is None
        assert not {
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name() == "honcho-client-reset"
        }
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_cancelled_provider_shutdown_closes_manager_and_client(tmp_path):
    manager_started = asyncio.Event()
    release_manager = asyncio.Event()
    manager_calls = 0

    class Manager:
        async def shutdown(self) -> None:
            nonlocal manager_calls
            manager_calls += 1
            manager_started.set()
            await release_manager.wait()

    provider = HonchoMemoryProvider()
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        await client_module._retain_honcho_client_lifecycle(provider)
        state = await client_module._activate_honcho_client_state()
        transport = types.SimpleNamespace(close=AsyncMock())

        async def build():
            return types.SimpleNamespace(_async_http=transport)

        await state.slot.get(build)
        provider._manager = Manager()

        shutdown_task = asyncio.create_task(provider.shutdown())
        await manager_started.wait()
        shutdown_task.cancel()
        await asyncio.sleep(0)
        shutdown_task.cancel()
        release_manager.set()

        with pytest.raises(asyncio.CancelledError):
            await shutdown_task
        assert manager_calls == 1
        transport.close.assert_awaited_once()
        assert state.slot.peek() is None
        assert provider not in client_module._honcho_client_owners
        assert not {
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("honcho-")
        }
    finally:
        reset_hermes_home_override(token)
