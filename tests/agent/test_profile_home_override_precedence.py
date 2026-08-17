"""Profile-home precedence for prompt assembly."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _DB:
    def __init__(self, home: Path):
        self.db_path = home / "state.db"


def _agent_for(home: Path):
    return SimpleNamespace(_session_db=_DB(home))


def test_bound_override_wins_over_shared_db_home(tmp_path):
    from agent import system_prompt

    root = tmp_path / "root"
    bot_home = root / "profiles" / "mybot"
    bot_home.mkdir(parents=True)
    agent = _agent_for(root)
    token = set_hermes_home_override(bot_home)
    try:
        assert system_prompt._agent_home(agent) == bot_home
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_db_home_wins_without_bound_override(tmp_path, monkeypatch):
    from agent import system_prompt

    root = tmp_path / "root"
    bot_home = root / "profiles" / "mybot"
    bot_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert system_prompt._agent_home(_agent_for(bot_home)) == bot_home
    assert await system_prompt._profile_name_for_home(bot_home) == "mybot"


@pytest.mark.asyncio
async def test_soul_and_skills_use_agent_home(tmp_path, monkeypatch):
    from agent import prompt_builder

    default_home = tmp_path / "default"
    bot_home = default_home / "profiles" / "bot"
    bot_home.mkdir(parents=True)
    (default_home / "SOUL.md").write_text("DEFAULT", encoding="utf-8")
    (bot_home / "SOUL.md").write_text("BOT", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    assert await prompt_builder.load_soul_md(home_override=bot_home) == "BOT"
