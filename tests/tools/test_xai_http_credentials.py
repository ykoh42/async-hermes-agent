import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _set_xai_oauth_unavailable(monkeypatch):
    from agent import credential_pool

    pool = MagicMock()
    pool.select = AsyncMock(return_value=None)
    pool.try_refresh_matching = AsyncMock(return_value=None)
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        AsyncMock(return_value=pool),
    )


@pytest.mark.asyncio
async def test_has_xai_credentials_fails_closed_without_profile_scope(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from tools.xai_http import has_xai_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "foreign-process-key")
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            await has_xai_credentials()
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
async def test_has_xai_credentials_isolates_concurrent_profile_auth_stores(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.xai_http import has_xai_credentials

    process_home = tmp_path / "process"
    process_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.setenv("XAI_API_KEY", "foreign-process-key")

    async def probe(home, *, oauth_token):
        home.mkdir()
        if oauth_token:
            (home / "auth.json").write_text(
                json.dumps(
                    {
                        "providers": {
                            "xai-oauth": {
                                "tokens": {"access_token": oauth_token}
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
        home_token = set_hermes_home_override(home)
        scope_token = secret_scope.set_secret_scope({"XAI_API_KEY": ""})
        try:
            await asyncio.sleep(0)
            return await has_xai_credentials()
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        assert await asyncio.gather(
            probe(tmp_path / "a", oauth_token="oauth-a"),
            probe(tmp_path / "b", oauth_token=""),
        ) == [True, False]
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
async def test_xai_credentials_fail_closed_without_profile_scope(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from hermes_cli.config import invalidate_env_cache
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "foreign-xai-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://foreign.example/v1")
    _set_xai_oauth_unavailable(monkeypatch)
    invalidate_env_cache()
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            await resolve_xai_http_credentials(force_refresh=True)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        invalidate_env_cache()


@pytest.mark.asyncio
async def test_xai_credentials_preserve_explicit_empty_profile_values(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "foreign-process-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://foreign-process.x.ai/v1")
    _set_xai_oauth_unavailable(monkeypatch)
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(
        {
            "HERMES_XAI_BASE_URL": "",
            "XAI_API_KEY": "",
            "XAI_BASE_URL": "",
        }
    )
    secret_scope.set_multiplex_active(True)
    try:
        assert await resolve_xai_http_credentials(force_refresh=True) == {
            "provider": "xai",
            "api_key": "",
            "base_url": "https://api.x.ai/v1",
        }
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
async def test_oauth_profile_scope_error_is_not_masked_by_api_key_fallback(
    monkeypatch,
):
    from agent import credential_pool, secret_scope
    from hermes_cli import config
    from tools.xai_http import resolve_xai_http_credentials

    fallback = AsyncMock(return_value="would-mask-profile-error")
    monkeypatch.setattr(config, "get_env_value_prefer_dotenv", fallback)
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        AsyncMock(
            side_effect=secret_scope.UnscopedSecretError("missing profile scope")
        ),
    )
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope({})
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(
            secret_scope.UnscopedSecretError,
            match="missing profile scope",
        ):
            await resolve_xai_http_credentials()
        fallback.assert_not_awaited()
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
@pytest.mark.parametrize("force_refresh", [False, True])
async def test_xai_oauth_pool_cancellation_never_enters_api_key_fallback(
    force_refresh, monkeypatch
):
    from agent import credential_pool
    from hermes_cli import config
    from tools.xai_http import resolve_xai_http_credentials

    pool = MagicMock()
    pool.select = AsyncMock(
        side_effect=asyncio.CancelledError() if not force_refresh else None
    )
    pool.try_refresh_matching = AsyncMock(
        side_effect=asyncio.CancelledError() if force_refresh else None
    )
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        AsyncMock(return_value=pool),
    )
    fallback = AsyncMock(return_value="would-mask-cancellation")
    monkeypatch.setattr(config, "get_env_value_prefer_dotenv", fallback)

    with pytest.raises(asyncio.CancelledError):
        await resolve_xai_http_credentials(force_refresh=force_refresh)
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_xai_env_lookup_cancellation_propagates(monkeypatch):
    from hermes_cli import config
    from tools.xai_http import resolve_xai_http_credentials

    _set_xai_oauth_unavailable(monkeypatch)
    reader = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(config, "get_env_value_prefer_dotenv", reader)

    with pytest.raises(asyncio.CancelledError):
        await resolve_xai_http_credentials()
    reader.assert_awaited_once_with("HERMES_XAI_BASE_URL")


@pytest.mark.asyncio
async def test_concurrent_api_key_profiles_do_not_share_credentials(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.xai_http import resolve_xai_http_credentials

    process_home = tmp_path / "process"
    process_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.setenv("XAI_API_KEY", "foreign-process-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://foreign-process.x.ai/v1")
    _set_xai_oauth_unavailable(monkeypatch)

    async def resolve(home, api_key, base_url):
        home.mkdir()
        home_token = set_hermes_home_override(home)
        scope_token = secret_scope.set_secret_scope(
            {"XAI_API_KEY": api_key, "XAI_BASE_URL": base_url}
        )
        try:
            await asyncio.sleep(0)
            return await resolve_xai_http_credentials(force_refresh=True)
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        profile_a, profile_b = await asyncio.gather(
            resolve(tmp_path / "a", "profile-a-key", "https://a.x.ai/v1"),
            resolve(tmp_path / "b", "profile-b-key", "https://b.x.ai/v1"),
        )
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)

    assert profile_a == {
        "provider": "xai",
        "api_key": "profile-a-key",
        "base_url": "https://a.x.ai/v1",
    }
    assert profile_b == {
        "provider": "xai",
        "api_key": "profile-b-key",
        "base_url": "https://b.x.ai/v1",
    }


@pytest.mark.asyncio
async def test_concurrent_oauth_profiles_keep_auth_stores_and_overrides_isolated(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.xai_http import resolve_xai_http_credentials

    process_home = tmp_path / "process"
    process_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    async def resolve(home, name):
        home.mkdir()
        (home / "auth.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {
                        "xai-oauth": {
                            "tokens": {
                                "access_token": f"oauth-{name}",
                                "refresh_token": f"refresh-{name}",
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        home_token = set_hermes_home_override(home)
        scope_token = secret_scope.set_secret_scope(
            {
                "HERMES_XAI_BASE_URL": f"https://{name}.x.ai/v1",
                "XAI_API_KEY": f"fallback-{name}",
            }
        )
        try:
            await asyncio.sleep(0)
            return await resolve_xai_http_credentials()
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        profile_a, profile_b = await asyncio.gather(
            resolve(tmp_path / "a", "a"),
            resolve(tmp_path / "b", "b"),
        )
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)

    assert profile_a == {
        "provider": "xai-oauth",
        "api_key": "oauth-a",
        "base_url": "https://a.x.ai/v1",
    }
    assert profile_b == {
        "provider": "xai-oauth",
        "api_key": "oauth-b",
        "base_url": "https://b.x.ai/v1",
    }


@pytest.mark.asyncio
async def test_xai_credentials_do_not_fall_back_to_environ_when_scope_has_no_key(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from hermes_cli.config import invalidate_env_cache
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "foreign-xai-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://foreign.example/v1")
    _set_xai_oauth_unavailable(monkeypatch)
    invalidate_env_cache()
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope({})
    secret_scope.set_multiplex_active(True)
    try:
        credentials = await resolve_xai_http_credentials(force_refresh=True)
        assert credentials == {
            "provider": "xai",
            "api_key": "",
            "base_url": "https://api.x.ai/v1",
        }
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        invalidate_env_cache()
