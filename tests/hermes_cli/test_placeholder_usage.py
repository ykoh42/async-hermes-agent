"""Tests for CLI placeholder text in config/setup output."""

import os
from unittest.mock import patch

from hermes_cli.config import show_config


def test_show_config_marks_placeholders(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        show_config()

    out = capsys.readouterr().out
    assert "hermes config set <key> <value>" in out
