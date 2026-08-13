from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from hermes_cli import copilot_auth


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(outer_token)
        secret_scope.set_multiplex_active(previous_multiplex)


async def _in_scope(secrets: dict[str, str]):
    token = secret_scope.set_secret_scope(secrets)
    try:
        await asyncio.sleep(0)
        return await copilot_auth.resolve_copilot_token()
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_copilot_tokens_are_isolated_between_concurrent_profiles(
    monkeypatch,
):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_foreign-process")
    monkeypatch.setenv("GH_TOKEN", "gho_foreign-process-gh")
    monkeypatch.setenv("GITHUB_TOKEN", "gho_foreign-process-github")
    secret_scope.set_multiplex_active(True)

    profile_a, profile_b = await asyncio.gather(
        _in_scope({"COPILOT_GITHUB_TOKEN": "gho_profile_a"}),
        _in_scope({"GH_TOKEN": "github_pat_profile_b"}),
    )

    assert profile_a == ("gho_profile_a", "COPILOT_GITHUB_TOKEN")
    assert profile_b == ("github_pat_profile_b", "GH_TOKEN")


@pytest.mark.asyncio
async def test_copilot_token_fails_closed_without_profile_scope(monkeypatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_foreign-process")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="COPILOT_GITHUB_TOKEN",
    ):
        await copilot_auth.resolve_copilot_token()


@pytest.mark.asyncio
async def test_explicit_empty_profile_tokens_skip_global_gh_fallback(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "gho_foreign-process")
    cli_lookup = AsyncMock(return_value="gho_foreign-cli")
    monkeypatch.setattr(copilot_auth, "_try_gh_cli_token", cli_lookup)
    secret_scope.set_multiplex_active(True)

    assert await _in_scope(
        {
            "COPILOT_GITHUB_TOKEN": "",
            "GH_TOKEN": "",
            "GITHUB_TOKEN": "",
        }
    ) == ("", "")
    cli_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_gh_fallback_skips_process_account_in_multiplex(
    monkeypatch,
):
    candidates = AsyncMock(return_value=["gh"])
    monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", candidates)
    secret_scope.set_multiplex_active(True)

    assert await copilot_auth._try_gh_cli_token() is None
    candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_profile_gh_fallback_return_shape_is_unchanged(monkeypatch):
    for env_var in copilot_auth.COPILOT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        copilot_auth,
        "_try_gh_cli_token",
        AsyncMock(return_value="gho_single_profile"),
    )

    assert await copilot_auth.resolve_copilot_token() == (
        "gho_single_profile",
        "gh auth token",
    )


@pytest.mark.asyncio
async def test_copilot_token_resolution_propagates_cancellation(monkeypatch):
    for env_var in copilot_auth.COPILOT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    entered = asyncio.Event()

    async def wait_for_cli():
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(copilot_auth, "_try_gh_cli_token", wait_for_cli)
    task = asyncio.create_task(copilot_auth.resolve_copilot_token())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
