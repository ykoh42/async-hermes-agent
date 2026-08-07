from unittest.mock import AsyncMock, patch

import pytest

from agent import secret_scope
from tools import managed_tool_gateway


@pytest.fixture(autouse=True)
def _reset_secret_scope():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_resolve_managed_tool_gateway_derives_vendor_origin():
    with patch.object(
        managed_tool_gateway,
        "managed_nous_tools_enabled",
        new=AsyncMock(return_value=True),
    ):
        result = await managed_tool_gateway.resolve_managed_tool_gateway(
            "firecrawl",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=AsyncMock(return_value="nous-token"),
        )

    assert result is not None
    assert result.gateway_origin == "https://firecrawl-gateway.example.com"
    assert result.nous_user_token == "nous-token"
    assert result.managed_mode is True


@pytest.mark.asyncio
async def test_scoped_token_miss_does_not_borrow_dotenv(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "other-profile-token")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"UNRELATED": "value"})
    try:
        assert await managed_tool_gateway._read_user_token_override() is None
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_read_nous_access_token_refreshes_with_upstream_skew():
    with (
        patch.object(
            managed_tool_gateway,
            "_read_user_token_override",
            new=AsyncMock(return_value=None),
        ),
        patch.object(
            managed_tool_gateway,
            "peek_nous_access_token",
            new=AsyncMock(return_value="stale-token"),
        ),
        patch(
            "hermes_cli.auth.resolve_nous_access_token",
            new=AsyncMock(return_value="fresh-token"),
        ) as resolve_token,
    ):
        assert await managed_tool_gateway.read_nous_access_token() == "fresh-token"

    resolve_token.assert_awaited_once_with(refresh_skew_seconds=120)


@pytest.mark.asyncio
async def test_read_nous_access_token_preserves_cached_token_on_refresh_failure():
    with (
        patch.object(
            managed_tool_gateway,
            "_read_user_token_override",
            new=AsyncMock(return_value=None),
        ),
        patch.object(
            managed_tool_gateway,
            "peek_nous_access_token",
            new=AsyncMock(return_value="cached-token"),
        ),
        patch(
            "hermes_cli.auth.resolve_nous_access_token",
            new=AsyncMock(side_effect=RuntimeError("portal unavailable")),
        ),
    ):
        assert await managed_tool_gateway.read_nous_access_token() == "cached-token"
