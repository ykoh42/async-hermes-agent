"""FAL_KEY env var normalization (whitespace-only treated as unset)."""

from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from tools.tool_backend_helpers import fal_key_is_configured


@pytest.fixture(autouse=True)
def _reset_secret_scope():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_fal_key_whitespace_is_unset(monkeypatch):
    # Whitespace-only FAL_KEY must NOT register as configured, and the managed
    # gateway fallback must be disabled for this assertion to be meaningful.
    monkeypatch.setenv("FAL_KEY", "   ")

    from tools import image_generation_tool

    monkeypatch.setattr(
        image_generation_tool,
        "_resolve_managed_fal_gateway",
        AsyncMock(return_value=None),
    )

    assert await image_generation_tool.check_fal_api_key() is False


@pytest.mark.asyncio
async def test_scoped_fal_key_miss_does_not_borrow_dotenv(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "other-profile-key")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"UNRELATED": "value"})
    try:
        assert await fal_key_is_configured() is False
    finally:
        secret_scope.reset_secret_scope(token)


# ---------------------------------------------------------------------------
# Actionable setup message when no FAL backend is reachable.
# Regression for the silent-drop UX gap described in issue #2543.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_generate_tool_returns_actionable_error_when_no_backend(monkeypatch):
    """End-to-end: handler must surface the actionable message, not a bare string."""
    import json

    from tools import image_generation_tool

    monkeypatch.setattr(
        image_generation_tool, "fal_key_is_configured", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        image_generation_tool,
        "_resolve_managed_fal_gateway",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        image_generation_tool,
        "managed_nous_tools_enabled",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        image_generation_tool,
        "nous_tool_gateway_unavailable_message",
        AsyncMock(return_value=""),
    )

    result = json.loads(
        await image_generation_tool.image_generate_tool(prompt="a cat")
    )

    assert result["success"] is False
    assert "https://fal.ai" in result["error"]
    assert "FAL_KEY" in result["error"]
