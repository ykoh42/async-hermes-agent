from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from agent.codex_runtime import _resolve_codex_app_server_home
from agent.transports.codex_app_server_session import CodexAppServerSession


@pytest.fixture(autouse=True)
def _restore_secret_scope(monkeypatch: pytest.MonkeyPatch):
    token = secret_scope.set_secret_scope(None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    yield
    secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_codex_home_is_isolated_across_concurrent_profiles(monkeypatch):
    monkeypatch.setenv("CODEX_HOME", "/foreign/process/codex")

    async def resolve(label: str) -> str | None:
        token = secret_scope.set_secret_scope(
            {"CODEX_HOME": f"/profiles/{label}/codex"}
        )
        try:
            await asyncio.sleep(0)
            return _resolve_codex_app_server_home()
        finally:
            secret_scope.reset_secret_scope(token)

    assert await asyncio.gather(resolve("a"), resolve("b")) == [
        "/profiles/a/codex",
        "/profiles/b/codex",
    ]


@pytest.mark.parametrize("scope", [{}, {"CODEX_HOME": ""}])
def test_multiplex_codex_home_does_not_fall_back_to_os_user_state(
    monkeypatch,
    scope,
) -> None:
    monkeypatch.setenv("CODEX_HOME", "/foreign/process/codex")
    token = secret_scope.set_secret_scope(scope)
    try:
        with pytest.raises(RuntimeError, match="profile-scoped CODEX_HOME"):
            _resolve_codex_app_server_home()
    finally:
        secret_scope.reset_secret_scope(token)


def test_unscoped_multiplex_codex_home_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", "/foreign/process/codex")
    with pytest.raises(secret_scope.UnscopedSecretError):
        _resolve_codex_app_server_home()


def test_single_profile_codex_home_parity(monkeypatch) -> None:
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("CODEX_HOME", "/legacy/codex")
    assert _resolve_codex_app_server_home() == "/legacy/codex"
    monkeypatch.delenv("CODEX_HOME")
    assert _resolve_codex_app_server_home() is None


def test_codex_permission_profile_uses_active_profile_setting(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "strict")
    token = secret_scope.set_secret_scope(
        {"HERMES_TERMINAL_SECURITY_MODE": "unrestricted"}
    )
    try:
        session = CodexAppServerSession()
    finally:
        secret_scope.reset_secret_scope(token)

    assert session._permission_profile == "full-access"
