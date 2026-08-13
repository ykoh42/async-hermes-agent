from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import auth


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(outer_token)
        secret_scope.set_multiplex_active(previous_multiplex)


async def _resolve_in_scope(values: dict[str, str]) -> dict:
    token = secret_scope.set_secret_scope(values)
    try:
        await asyncio.sleep(0)
        return await auth.resolve_qwen_runtime_credentials()
    finally:
        secret_scope.reset_secret_scope(token)


def test_qwen_profile_sources_participate_in_child_env_scrubbing() -> None:
    from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

    provider = auth.PROVIDER_REGISTRY["qwen-oauth"]
    assert provider.api_key_env_vars == ("QWEN_API_KEY",)
    assert provider.base_url_env_var == "HERMES_QWEN_BASE_URL"
    assert {"QWEN_API_KEY", "HERMES_QWEN_BASE_URL"}.issubset(
        _HERMES_PROVIDER_ENV_BLOCKLIST
    )


@pytest.mark.asyncio
async def test_qwen_runtime_uses_only_each_profile_scoped_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("QWEN_API_KEY", "foreign-process-qwen")
    monkeypatch.setenv("HERMES_QWEN_BASE_URL", "https://foreign.example/v1")
    global_reader = AsyncMock(side_effect=AssertionError("must not read ~/.qwen"))
    monkeypatch.setattr(auth, "_read_qwen_cli_tokens", global_reader)

    profile_a, profile_b = await asyncio.gather(
        _resolve_in_scope(
            {
                "QWEN_API_KEY": "profile-a-qwen",
                "HERMES_QWEN_BASE_URL": "https://a.example/v1/",
            }
        ),
        _resolve_in_scope(
            {
                "QWEN_API_KEY": "profile-b-qwen",
                "HERMES_QWEN_BASE_URL": "https://b.example/v1",
            }
        ),
    )

    assert (profile_a["api_key"], profile_a["base_url"]) == (
        "profile-a-qwen",
        "https://a.example/v1",
    )
    assert (profile_b["api_key"], profile_b["base_url"]) == (
        "profile-b-qwen",
        "https://b.example/v1",
    )
    assert profile_a["source"] == profile_b["source"] == "env:QWEN_API_KEY"
    assert profile_a["auth_file"] == profile_b["auth_file"] == ""
    global_reader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{}, {"QWEN_API_KEY": ""}])
async def test_qwen_runtime_missing_or_empty_profile_key_does_not_borrow_process(
    monkeypatch: pytest.MonkeyPatch,
    scope: dict[str, str],
) -> None:
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("QWEN_API_KEY", "foreign-process-qwen")
    global_reader = AsyncMock(side_effect=AssertionError("must not read ~/.qwen"))
    monkeypatch.setattr(auth, "_read_qwen_cli_tokens", global_reader)

    token = secret_scope.set_secret_scope(scope)
    try:
        with pytest.raises(auth.AuthError) as exc_info:
            await auth.resolve_qwen_runtime_credentials()
    finally:
        secret_scope.reset_secret_scope(token)

    assert exc_info.value.provider == "qwen-oauth"
    assert exc_info.value.code == "qwen_profile_credentials_required"
    global_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_qwen_runtime_unscoped_multiplex_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("QWEN_API_KEY", "foreign-process-qwen")

    with pytest.raises(secret_scope.UnscopedSecretError, match="QWEN_API_KEY"):
        await auth.resolve_qwen_runtime_credentials()


@pytest.mark.asyncio
async def test_qwen_global_store_helpers_fail_before_io_or_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"QWEN_API_KEY": "profile-key"})
    path_lookup = Mock(side_effect=AssertionError("must not inspect ~/.qwen"))
    client_factory = AsyncMock(side_effect=AssertionError("must not refresh"))
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", path_lookup)
    monkeypatch.setattr(auth, "_create_httpx_client", client_factory)
    try:
        for operation in (
            auth._read_qwen_cli_tokens(),
            auth._save_qwen_cli_tokens({"access_token": "profile-key"}),
            auth._refresh_qwen_cli_tokens(
                {"access_token": "global", "refresh_token": "global-refresh"}
            ),
        ):
            with pytest.raises(auth.AuthError) as exc_info:
                await operation
            assert exc_info.value.code == "qwen_global_credentials_unsafe"
    finally:
        secret_scope.reset_secret_scope(token)

    path_lookup.assert_not_called()
    client_factory.assert_not_awaited()


@pytest.mark.asyncio
async def test_qwen_single_profile_keeps_cli_file_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(False)
    auth_path = tmp_path / ".qwen" / "oauth_creds.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text(
        json.dumps(
            {
                "access_token": "single-profile-cli-token",
                "refresh_token": "refresh",
                "expiry_date": int((time.time() + 3600) * 1000),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: auth_path)
    monkeypatch.setenv("QWEN_API_KEY", "single-profile-env-token")

    resolved = await auth.resolve_qwen_runtime_credentials()

    assert resolved["api_key"] == "single-profile-cli-token"
    assert resolved["source"] == "qwen-cli"
    assert resolved["auth_file"] == str(auth_path)


@pytest.mark.asyncio
async def test_qwen_refresh_cancellation_closes_owned_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(False)
    request_started = asyncio.Event()
    client_closed = asyncio.Event()

    class BlockingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc) -> None:
            client_closed.set()

        async def post(self, *_args, **_kwargs):
            request_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        auth,
        "_create_httpx_client",
        AsyncMock(return_value=BlockingClient()),
    )

    task = asyncio.create_task(
        auth._refresh_qwen_cli_tokens(
            {"access_token": "old", "refresh_token": "refresh"}
        )
    )
    await asyncio.wait_for(request_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client_closed.is_set()


@pytest.mark.asyncio
async def test_qwen_repeatedly_cancelled_atomic_save_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(False)
    auth_path = tmp_path / ".qwen" / "oauth_creds.json"
    replace_started = asyncio.Event()
    remove_started = asyncio.Event()
    release_remove = asyncio.Event()
    real_remove = auth.aiofiles.os.remove

    async def blocked_replace(_source: Path, _target: Path) -> None:
        replace_started.set()
        await asyncio.Event().wait()

    async def blocked_remove(path: Path) -> None:
        remove_started.set()
        await release_remove.wait()
        await real_remove(path)

    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: auth_path)
    monkeypatch.setattr(auth.aiofiles.os, "replace", blocked_replace)
    monkeypatch.setattr(auth.aiofiles.os, "remove", blocked_remove)

    task = asyncio.create_task(
        auth._save_qwen_cli_tokens({"access_token": "never-persisted"})
    )
    await asyncio.wait_for(replace_started.wait(), timeout=1)
    task.cancel()
    await asyncio.wait_for(remove_started.wait(), timeout=1)
    task.cancel()
    release_remove.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not auth_path.exists()
    assert list(auth_path.parent.glob("oauth_creds.json.tmp.*")) == []


@pytest.mark.asyncio
async def test_qwen_pool_seeding_isolates_concurrent_profile_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.credential_pool import load_pool

    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("QWEN_API_KEY", "foreign-process-qwen")

    async def load(label: str):
        home_token = set_hermes_home_override(tmp_path / f"profile-{label}")
        scope_token = secret_scope.set_secret_scope(
            {
                "QWEN_API_KEY": f"profile-{label}-qwen",
                "HERMES_QWEN_BASE_URL": f"https://{label}.example/v1",
            }
        )
        try:
            pool = await load_pool("qwen-oauth")
            entry = pool.entries()[0]
            return entry.access_token, entry.base_url, entry.source
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    assert await asyncio.gather(load("a"), load("b")) == [
        ("profile-a-qwen", "https://a.example/v1", "env:QWEN_API_KEY"),
        ("profile-b-qwen", "https://b.example/v1", "env:QWEN_API_KEY"),
    ]


@pytest.mark.asyncio
async def test_qwen_runtime_provider_isolates_concurrent_profile_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config, runtime_provider

    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("QWEN_API_KEY", "foreign-process-qwen")
    monkeypatch.setattr(
        runtime_provider,
        "resolve_requested_provider",
        AsyncMock(return_value="qwen-oauth"),
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_provider",
        AsyncMock(return_value="qwen-oauth"),
    )
    monkeypatch.setattr(config, "load_config_readonly", AsyncMock(return_value={}))

    async def resolve(label: str):
        token = secret_scope.set_secret_scope(
            {
                "QWEN_API_KEY": f"profile-{label}-qwen",
                "HERMES_QWEN_BASE_URL": f"https://{label}.example/v1",
            }
        )
        try:
            return await runtime_provider.resolve_runtime_provider(
                requested="qwen-oauth"
            )
        finally:
            secret_scope.reset_secret_scope(token)

    profile_a, profile_b = await asyncio.gather(resolve("a"), resolve("b"))

    assert (profile_a["api_key"], profile_a["base_url"], profile_a["source"]) == (
        "profile-a-qwen",
        "https://a.example/v1",
        "env:QWEN_API_KEY",
    )
    assert (profile_b["api_key"], profile_b["base_url"], profile_b["source"]) == (
        "profile-b-qwen",
        "https://b.example/v1",
        "env:QWEN_API_KEY",
    )
