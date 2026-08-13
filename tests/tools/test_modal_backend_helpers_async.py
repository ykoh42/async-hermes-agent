"""Modal backend selection parity with async credential discovery."""

from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from tools import tool_backend_helpers as helpers


@pytest.fixture(autouse=True)
def _restore_modal_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(False)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_direct_modal_credentials_from_environment(monkeypatch):
    # Whitespace remains truthy exactly like the retained os.getenv checks.
    monkeypatch.setenv("MODAL_TOKEN_ID", " ")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "\t")
    assert await helpers.has_direct_modal_credentials() is True


@pytest.mark.asyncio
async def test_direct_modal_credentials_are_isolated_between_profiles(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    secret_scope.set_multiplex_active(True)

    async def unexpected_config(_path):  # noqa: ASYNC124 - must not be called
        pytest.fail("multiplexed selection must not probe OS-user ~/.modal.toml")

    monkeypatch.setattr(helpers.aiofiles.os.path, "exists", unexpected_config)

    async def configured() -> bool:
        token = secret_scope.set_secret_scope(
            {
                "MODAL_TOKEN_ID": "profile-a-id",
                "MODAL_TOKEN_SECRET": "profile-a-secret",
            }
        )
        try:
            return await helpers.has_direct_modal_credentials()
        finally:
            secret_scope.reset_secret_scope(token)

    async def incomplete() -> bool:
        token = secret_scope.set_secret_scope(
            {"MODAL_TOKEN_ID": "profile-b-id"}
        )
        try:
            return await helpers.has_direct_modal_credentials()
        finally:
            secret_scope.reset_secret_scope(token)

    assert await asyncio.gather(configured(), incomplete()) == [True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"MODAL_TOKEN_ID": "", "MODAL_TOKEN_SECRET": "profile-secret"},
        {"MODAL_TOKEN_ID": "profile-id", "MODAL_TOKEN_SECRET": ""},
    ],
)
async def test_direct_modal_credentials_skip_global_config_for_incomplete_scope(
    monkeypatch,
    scope,
):
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(scope)

    async def unexpected_config(_path):  # noqa: ASYNC124 - must not be called
        pytest.fail("multiplexed selection must not probe OS-user ~/.modal.toml")

    monkeypatch.setattr(helpers.aiofiles.os.path, "exists", unexpected_config)
    try:
        assert await helpers.has_direct_modal_credentials() is False
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_direct_modal_credentials_fail_closed_without_profile_scope(
    monkeypatch,
):
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError, match="MODAL_TOKEN_ID"):
        await helpers.has_direct_modal_credentials()


@pytest.mark.asyncio
async def test_direct_modal_credentials_preserve_modal_toml_fallback(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    observed_paths = []

    async def has_config(path):  # noqa: ASYNC124 - coroutine-shaped test double
        observed_paths.append(path)
        return True

    monkeypatch.setattr(helpers.aiofiles.os.path, "exists", has_config)

    assert await helpers.has_direct_modal_credentials() is True
    assert observed_paths[0].endswith("/.modal.toml")


@pytest.mark.asyncio
async def test_direct_modal_config_probe_preserves_cancellation(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    async def cancelled(_path):  # noqa: ASYNC124 - cancellation test double
        raise asyncio.CancelledError

    monkeypatch.setattr(helpers.aiofiles.os.path, "exists", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await helpers.has_direct_modal_credentials()


def test_modal_selection_prefers_managed_in_auto_mode():
    state = helpers.resolve_modal_backend_state(
        "auto",
        has_direct=True,
        managed_ready=True,
        managed_enabled=True,
    )
    assert state["selected_backend"] == "managed"


def test_managed_mode_is_blocked_without_entitlement():
    state = helpers.resolve_modal_backend_state(
        "managed",
        has_direct=True,
        managed_ready=True,
        managed_enabled=False,
    )
    assert state["selected_backend"] is None
    assert state["managed_mode_blocked"] is True
