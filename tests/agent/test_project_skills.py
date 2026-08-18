"""Project-local skill discovery and per-repository trust gate."""

import json
from pathlib import Path

import pytest

import agent.skill_utils as su
from tools.skills_tool import _find_all_skills, skill_view


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills" / "local-skill").mkdir(parents=True)
    (home / "skills" / "local-skill" / "SKILL.md").write_text(
        "---\nname: repo-skill\ndescription: local\n---\nlocal\n"
    )
    config = home / "config.yaml"
    config.write_text("skills:\n  external_dirs: []\n")
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    hs = repo / ".hermes" / "skills" / "repo-skill"
    hs.mkdir(parents=True)
    (hs / "SKILL.md").write_text(
        "---\nname: repo-skill\ndescription: from repo\n---\nrepo\n"
    )
    ag = repo / ".agents" / "skills" / "conv-skill"
    ag.mkdir(parents=True)
    (ag / "SKILL.md").write_text(
        "---\nname: conv-skill\ndescription: convention\n---\nbody\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(repo)
    su._external_dirs_cache_clear()
    yield {"home": home, "repo": repo, "config": config}
    su._external_dirs_cache_clear()


def _trust(config: Path, repo: Path) -> None:
    config.write_text(
        f"skills:\n  external_dirs: []\n  trusted_project_dirs: ['{repo}']\n"
    )
    su._external_dirs_cache_clear()


@pytest.mark.asyncio
async def test_untrusted_project_is_not_loaded(project_env):
    assert await su.get_project_skills_dirs() == []
    assert await su.get_untrusted_project_skills_root() is not None


@pytest.mark.asyncio
async def test_trusted_project_has_precedence(project_env):
    _trust(project_env["config"], project_env["repo"])
    dirs = await su.get_project_skills_dirs()
    assert (project_env["repo"] / ".hermes" / "skills").resolve() in dirs
    assert (project_env["repo"] / ".agents" / "skills").resolve() in dirs
    assert await su.find_project_root() == project_env["repo"].resolve()

    listed = await _find_all_skills()
    names = [item["name"] for item in listed]
    assert names.count("repo-skill") == 1
    assert "conv-skill" in names
    result = await skill_view("repo-skill")
    payload = json.loads(result)
    assert payload["success"] is True
    assert payload["content"].endswith("repo\n")


@pytest.mark.asyncio
async def test_project_discovery_can_be_disabled(project_env):
    project_env["config"].write_text(
        "skills:\n  project_discovery: false\n"
        f"  trusted_project_dirs: ['{project_env['repo']}']\n"
    )
    su._external_dirs_cache_clear()
    assert await su.get_project_skills_dirs() == []
    assert await su.get_untrusted_project_skills_root() is None


@pytest.mark.asyncio
async def test_git_file_is_a_project_marker(tmp_path, monkeypatch):
    repo = tmp_path / "worktree"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere\n")
    monkeypatch.chdir(repo)
    assert await su.find_project_root() == repo.resolve()
