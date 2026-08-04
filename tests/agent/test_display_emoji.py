"""Tests for get_tool_emoji in agent/display.py — skin + registry integration."""

from unittest.mock import patch as mock_patch, MagicMock

from agent.display import get_tool_emoji


class TestGetToolEmoji:
    """Verify the skin → registry → fallback resolution chain."""





    def test_custom_default(self):
        """Custom default is returned when nothing matches."""
        with mock_patch("agent.display._get_skin", return_value=None):
            mock_reg = MagicMock()
            mock_reg.get_emoji.return_value = ""
            import sys
            mock_module = MagicMock()
            mock_module.registry = mock_reg
            with mock_patch.dict(sys.modules, {"tools.registry": mock_module}):
                result = get_tool_emoji("x", default="⚙️")
                assert result == "⚙️"

    def test_registry_emoji_is_used_without_ui_skin_overrides(self):
        """The async harness keeps display metadata in the tool registry."""
        skin = MagicMock()
        skin.tool_emojis = {"terminal": "⚔"}
        mock_reg = MagicMock()
        mock_reg.get_emoji.return_value = "🔍"
        import sys
        mock_module = MagicMock()
        mock_module.registry = mock_reg
        with mock_patch("agent.display._get_skin", return_value=skin), \
             mock_patch.dict(sys.modules, {"tools.registry": mock_module}):
            assert get_tool_emoji("terminal") == "🔍"
            assert get_tool_emoji("web_search") == "🔍"
