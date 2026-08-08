"""Tests for _parse_env_var and _get_env_config env-var validation."""

import importlib
from unittest.mock import patch

import pytest

import sys
import tools.terminal_tool  # noqa: F401 -- ensure module is loaded
_tt_mod = sys.modules["tools.terminal_tool"]
from tools.terminal_tool import _parse_env_var


class TestParseEnvVar:
    """Unit tests for _parse_env_var."""

    # -- valid values work normally --

    def test_valid_int(self):
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "300"}):
            assert _parse_env_var("TERMINAL_TIMEOUT", "180") == 300


    # -- invalid int raises ValueError with env var name --


class TestImportTimeEnvParsing:
    """Module-level env parsing should never make terminal_tool unimportable."""

    def test_invalid_foreground_timeout_falls_back_to_default(self):
        try:
            with patch.dict("os.environ", {"TERMINAL_MAX_FOREGROUND_TIMEOUT": "5m"}, clear=False):
                mod = importlib.reload(_tt_mod)
                assert mod.FOREGROUND_MAX_TIMEOUT == 600
        finally:
            importlib.reload(_tt_mod)

    def test_invalid_disk_warning_threshold_falls_back_to_default(self):
        try:
            with patch.dict("os.environ", {"TERMINAL_DISK_WARNING_GB": "huge"}, clear=False):
                mod = importlib.reload(_tt_mod)
                assert mod.DISK_USAGE_WARNING_THRESHOLD_GB == 500.0
        finally:
            importlib.reload(_tt_mod)
