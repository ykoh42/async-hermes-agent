"""Regression tests for credential-pool OAuth refresh write-through to root.

Companion to ``tests/hermes_cli/test_xai_oauth_writethrough.py``. That file
covers the *non-pool* xAI refresh path (``_save_xai_oauth_tokens``). These
cover the **credential-pool** refresh path
(``CredentialPool._sync_device_code_entry_to_auth_store``): when a profile
that has no own ``providers.<id>`` block refreshes — via the pool — a rotating
OAuth grant it resolved from the global-root fallback, the rotated chain must
be written back to the global root too. Otherwise root keeps a revoked refresh
token and every other profile reading root's stale grant dies with
``refresh_token_reused`` / ``invalid_grant`` once its access token expires
(issue #48415, the Codex/xAI analog of #43589).

The tests drive the real ``_sync_device_code_entry_to_auth_store`` against
real on-disk auth stores (profile + root under ``tmp_path``) rather than
mocking the save boundary, so they exercise the actual atomic write path.
"""

import json
import asyncio
from unittest.mock import AsyncMock

import pytest

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)
from hermes_cli import auth as A


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(provider: str, *, id: str, access_token: str, refresh_token: str):
    return PooledCredential(
        provider=provider,
        id=id,
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token=access_token,
        refresh_token=refresh_token,
    )


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    The pytest seat belt in ``_write_through_provider_state_to_global_root``
    only refuses the *real* user's ``$HOME/.hermes/auth.json``; a tmp_path
    root is allowed, so point HOME away from the tmp root to keep the guard
    from tripping on these fixtures.
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    async def auth_file_path():
        return profile_path

    async def global_auth_file_path():
        return root_path

    monkeypatch.setattr(A, "_auth_file_path", auth_file_path)
    monkeypatch.setattr(A, "_global_auth_file_path", global_auth_file_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path








@pytest.mark.asyncio
async def test_global_write_through_preserves_concurrent_root_update(
    profile_and_root, monkeypatch
):
    """A stale profile write-through must not erase a concurrent root login."""
    _profile_path, root_path = profile_and_root
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {"access_token": "old-xai", "refresh_token": "old-r"}
                }
            },
            "credential_pool": {
                "anthropic": [{"id": "anthropic-existing"}],
                "openrouter": [{"id": "openrouter-existing"}],
            },
        },
    )

    helper_loaded = asyncio.Event()
    allow_helper_save = asyncio.Event()
    writer_started = asyncio.Event()
    real_auth_load = A._load_auth_store

    async def paused_helper_load(path=None):
        store = await real_auth_load(path)
        if asyncio.current_task().get_name() == "profile-write-through":
            helper_loaded.set()
            await allow_helper_save.wait()
        return store

    monkeypatch.setattr(A, "_load_auth_store", paused_helper_load)

    async def profile_write_through():
        await CP._write_through_provider_state_to_global_root(
            "xai-oauth",
            {"tokens": {"access_token": "new-xai", "refresh_token": "new-r"}},
        )

    async def concurrent_codex_login():
        writer_started.set()
        async with A._auth_store_transaction(root_path):
            store = await A._load_auth_store(root_path)
            A._store_provider_state(
                store,
                "openai-codex",
                {"tokens": {"access_token": "codex-a", "refresh_token": "codex-r"}},
                set_active=False,
            )
            pool = store.setdefault("credential_pool", {})
            pool["openai-codex"] = [{"id": "codex-login"}]
            await A._save_auth_store(store, target_path=root_path)

    helper = asyncio.create_task(
        profile_write_through(),
        name="profile-write-through",
    )
    await asyncio.wait_for(helper_loaded.wait(), timeout=5)

    writer = asyncio.create_task(concurrent_codex_login(), name="concurrent-login")
    await asyncio.wait_for(writer_started.wait(), timeout=5)
    await asyncio.sleep(0)
    assert not writer.done()
    allow_helper_save.set()
    await asyncio.wait_for(asyncio.gather(helper, writer), timeout=5)

    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "new-r"
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "codex-r"
    assert root["credential_pool"]["openai-codex"] == [{"id": "codex-login"}]
    assert root["credential_pool"]["anthropic"] == [{"id": "anthropic-existing"}]
    assert root["credential_pool"]["openrouter"] == [{"id": "openrouter-existing"}]


@pytest.mark.asyncio
async def test_codex_pool_refresh_uses_native_async_transport(monkeypatch, tmp_path):
    provider = "openai-codex"

    entry = _entry(
        provider,
        id="codex-1",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    pool = CredentialPool(provider, [entry])

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    await A._save_auth_store(
        {
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                }
            }
        }
    )
    refresh = AsyncMock(
        return_value={
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "last_refresh": "2026-08-06T00:00:00Z",
        }
    )
    monkeypatch.setattr(A, "refresh_codex_oauth_pure", refresh)

    updated = await pool._refresh_entry(entry, force=True)

    assert updated is not None
    assert updated.access_token == "fresh-access"
    assert updated.refresh_token == "fresh-refresh"
    refresh.assert_awaited_once_with(
        "stale-access",
        "stale-refresh",
        timeout_seconds=20,
    )
    persisted = await A._load_auth_store()
    assert persisted["providers"][provider]["tokens"] == {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
    }
    assert persisted["credential_pool"][provider][0]["access_token"] == "fresh-access"


@pytest.mark.asyncio
async def test_codex_pool_refresh_updates_global_owner(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    provider = "openai-codex"
    entry = _entry(
        provider,
        id="codex-root",
        access_token="root-access",
        refresh_token="root-refresh",
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
            "credential_pool": {provider: [entry.to_dict()]},
        },
    )
    refresh = AsyncMock(
        return_value={
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2026-08-06T01:00:00Z",
        }
    )
    monkeypatch.setattr(A, "refresh_codex_oauth_pure", refresh)

    updated = await CredentialPool(provider, [entry])._refresh_entry(
        entry,
        force=True,
    )

    assert updated is not None
    root = _read_store(root_path)
    assert root["providers"][provider]["tokens"]["refresh_token"] == "rotated-refresh"
    assert root["credential_pool"][provider][0]["access_token"] == "rotated-access"
    assert not profile_path.exists()


@pytest.mark.asyncio
async def test_codex_terminal_refresh_failure_quarantines_singleton(
    monkeypatch,
    tmp_path,
):
    provider = "openai-codex"
    entry = _entry(
        provider,
        id="codex-terminal",
        access_token="dead-access",
        refresh_token="dead-refresh",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    await A._save_auth_store(
        {
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": entry.access_token,
                        "refresh_token": entry.refresh_token,
                    }
                }
            },
            "credential_pool": {provider: [entry.to_dict()]},
        }
    )
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        AsyncMock(
            side_effect=A.AuthError(
                "expired",
                provider=provider,
                code="invalid_grant",
                relogin_required=True,
            )
        ),
    )
    pool = CredentialPool(provider, [entry])

    assert await pool._refresh_entry(entry, force=True) is None

    persisted = await A._load_auth_store()
    assert persisted["providers"][provider]["tokens"] == {}
    assert persisted["providers"][provider]["last_auth_error"]["code"] == "invalid_grant"
    assert persisted["credential_pool"][provider] == []
    assert pool.entries() == []


@pytest.mark.asyncio
async def test_codex_transient_refresh_failure_keeps_credential(monkeypatch, tmp_path):
    provider = "openai-codex"
    entry = _entry(
        provider,
        id="codex-limited",
        access_token="limited-access",
        refresh_token="limited-refresh",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    await A._save_auth_store(
        {
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": entry.access_token,
                        "refresh_token": entry.refresh_token,
                    }
                }
            },
            "credential_pool": {provider: [entry.to_dict()]},
        }
    )
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        AsyncMock(
            side_effect=A.AuthError(
                "limited",
                provider=provider,
                code=A.CODEX_RATE_LIMITED_CODE,
                relogin_required=False,
            )
        ),
    )
    pool = CredentialPool(provider, [entry])

    assert await pool._refresh_entry(entry, force=True) is None

    persisted_entry = (await A._load_auth_store())["credential_pool"][provider][0]
    assert persisted_entry["access_token"] == "limited-access"
    assert persisted_entry["last_status"] == "exhausted"
    assert pool.entries()[0].last_status == "exhausted"
