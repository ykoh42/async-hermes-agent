"""Tests for ``export `` prefix handling in the hand-rolled .env parsers.

Bash-compatible .env files commonly prefix lines with ``export `` (users
copy-paste from shell profiles, cloud provider docs, tutorials). The relevant
parsers — ``hermes_cli.config.load_env``,
``hermes_cli.main._has_any_provider_configured``, and the native async skill
environment reader — split on ``line.partition("=")`` and must
strip the ``export `` prefix first, otherwise ``export API_KEY=sk-...`` is
stored under the wrong key ``"export API_KEY"`` and the real key is lost
(setup wizard re-triggers, providers undetected, skill env passthrough drops
the var). See PR #6659.

These assert the behavior contract (prefix stripped → canonical key resolves),
not the literal parser source.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

def _write_env(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_config_load_env_strips_export_prefix(tmp_path):
    from hermes_cli.config import invalidate_env_cache, load_env

    env_path = tmp_path / ".env"
    _write_env(
        env_path,
        'export OPENAI_API_KEY=sk-export-123\n'
        'export OPENROUTER_API_KEY="sk-or-456"\n'
        'ANTHROPIC_API_KEY=sk-plain-789\n',
    )
    invalidate_env_cache()
    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            env = load_env()
    finally:
        invalidate_env_cache()

    # Canonical keys resolve, export-prefixed wrong keys never appear.
    assert env["OPENAI_API_KEY"] == "sk-export-123"
    assert env["OPENROUTER_API_KEY"] == "sk-or-456"
    assert env["ANTHROPIC_API_KEY"] == "sk-plain-789"
    assert "export OPENAI_API_KEY" not in env


@pytest.mark.asyncio
async def test_skills_tool_load_env_strips_export_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(
        "export SOME_SKILL_KEY=skillval\nPLAIN=plainval\n", encoding="utf-8"
    )

    # The async skill reader resolves get_hermes_home()/.env at call time.
    import importlib

    import tools.skills_tool as skills_tool

    importlib.reload(skills_tool)
    with patch.object(skills_tool, "get_hermes_home", return_value=tmp_path):
        env = await skills_tool.load_env()

    assert env["SOME_SKILL_KEY"] == "skillval"
    assert env["PLAIN"] == "plainval"
    assert "export SOME_SKILL_KEY" not in env
