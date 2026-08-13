from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import anthropic_adapter, secret_scope


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(previous_multiplex)


async def _resolve_in_scope(values: dict[str, str]) -> str | None:
    token = secret_scope.set_secret_scope(values)
    try:
        return await anthropic_adapter.resolve_anthropic_token()
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_multiplex_resolution_skips_global_claude_code_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    global_reader = AsyncMock(
        return_value={
            "accessToken": "global-claude-code-token",
            "refreshToken": "global-refresh-token",
            "expiresAt": 0,
        }
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "read_claude_code_credentials",
        global_reader,
    )

    class EmptyPool:
        def entries(self):
            return []

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(return_value=EmptyPool()),
    )

    resolved = await asyncio.gather(
        _resolve_in_scope({"ANTHROPIC_TOKEN": "profile-a-token"}),
        _resolve_in_scope({"ANTHROPIC_TOKEN": "profile-b-token"}),
    )

    assert resolved == ["profile-a-token", "profile-b-token"]
    global_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiplex_global_readers_do_not_touch_keychain_or_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    spawn = AsyncMock(side_effect=AssertionError("must not spawn security"))
    exists = AsyncMock(side_effect=AssertionError("must not inspect ~/.claude"))
    monkeypatch.setattr(anthropic_adapter.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(anthropic_adapter.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(anthropic_adapter.aiofiles.os.path, "exists", exists)

    assert await anthropic_adapter._read_claude_code_credentials_from_keychain() is None
    assert await anthropic_adapter._read_claude_code_credentials_from_file() is None
    assert await anthropic_adapter.read_claude_code_credentials() is None
    spawn.assert_not_awaited()
    exists.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiplex_refresh_write_and_interactive_setup_fail_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    refresh = AsyncMock(side_effect=AssertionError("must not refresh global grant"))
    spawn = AsyncMock(side_effect=AssertionError("must not start Claude Code"))
    monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", refresh)
    monkeypatch.setattr(anthropic_adapter.asyncio, "create_subprocess_exec", spawn)

    with pytest.raises(RuntimeError, match="global OAuth credential"):
        await anthropic_adapter._refresh_oauth_token(
            {"accessToken": "global", "refreshToken": "global-refresh"}
        )
    with pytest.raises(RuntimeError, match="global credential file"):
        await anthropic_adapter._write_claude_code_credentials(
            "access", "refresh", 1
        )
    with pytest.raises(RuntimeError, match="global interactive OAuth setup"):
        await anthropic_adapter.run_oauth_setup_token()

    refresh.assert_not_awaited()
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_profile_keeps_global_claude_code_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(False)
    global_reader = AsyncMock(
        return_value={
            "accessToken": "global-claude-code-token",
            "refreshToken": "global-refresh-token",
            "expiresAt": 0,
        }
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "read_claude_code_credentials",
        global_reader,
    )
    monkeypatch.setenv("ANTHROPIC_TOKEN", "process-env-token")

    assert (
        await anthropic_adapter.resolve_anthropic_token()
        == "global-claude-code-token"
    )
    global_reader.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_multiplex_pool_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(side_effect=asyncio.CancelledError("cancel-pool")),
    )
    token = secret_scope.set_secret_scope({})
    try:
        with pytest.raises(asyncio.CancelledError, match="cancel-pool"):
            await anthropic_adapter.resolve_anthropic_token()
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_multiplex_pool_oauth_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_scope.set_multiplex_active(True)
    pool = SimpleNamespace(
        entries=lambda: [
            SimpleNamespace(
                auth_type="oauth",
                access_token="profile-owned-pool-token",
            )
        ]
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(return_value=pool),
    )
    token = secret_scope.set_secret_scope({})
    try:
        assert (
            await anthropic_adapter.resolve_anthropic_token()
            == "profile-owned-pool-token"
        )
    finally:
        secret_scope.reset_secret_scope(token)
