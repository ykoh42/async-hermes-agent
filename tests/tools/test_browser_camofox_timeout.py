"""Tests for browser_camofox._get_command_timeout — config-driven timeout."""
from unittest.mock import AsyncMock, patch

import pytest


class TestCamofoxCommandTimeout:
    """Verify that the Camofox HTTP backend reads browser.command_timeout."""

    pytestmark = pytest.mark.asyncio

    async def test_default_is_30(self):
        """When config has no browser.command_timeout, default to 30s."""
        from tools.browser_camofox import _get_command_timeout

        # Clear cache
        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(return_value={}),
        ):
            assert await _get_command_timeout() == 30


    async def test_config_read_error_falls_back(self):
        """If config read raises, fall back to 30s."""
        from tools.browser_camofox import _get_command_timeout

        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(side_effect=Exception("no config")),
        ):
            assert await _get_command_timeout() == 30
