"""Profile-scoped environment lookup for web provider adapters."""

from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from agent.web_search_provider import get_provider_env
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_web_provider_missing_profile_key_does_not_borrow_process_env(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("PROFILE_WEB_API_KEY", "process-profile-key")
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(tmp_path / "profile-b")
    secret_token = secret_scope.set_secret_scope({})
    try:
        assert await get_provider_env("PROFILE_WEB_API_KEY") == ""
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_web_provider_reads_concurrent_profiles_own_dotenv(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / ".env").write_text(
        "PROFILE_WEB_API_KEY=profile-a-key\n",
        encoding="utf-8",
    )
    (profile_b / ".env").write_text(
        "PROFILE_WEB_API_KEY=profile-b-key\n",
        encoding="utf-8",
    )
    secret_scope.set_multiplex_active(True)

    async def lookup(profile):
        home_token = set_hermes_home_override(profile)
        secret_token = secret_scope.set_secret_scope({})
        try:
            return await get_provider_env("PROFILE_WEB_API_KEY")
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    assert await asyncio.gather(lookup(profile_a), lookup(profile_b)) == [
        "profile-a-key",
        "profile-b-key",
    ]


@pytest.mark.asyncio
async def test_web_provider_unscoped_multiplex_lookup_fails_closed(monkeypatch):
    monkeypatch.setenv("PROFILE_WEB_API_KEY", "process-profile-key")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError):
        await get_provider_env("PROFILE_WEB_API_KEY")
