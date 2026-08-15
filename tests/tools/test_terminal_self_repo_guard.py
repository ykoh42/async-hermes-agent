"""terminal_tool wiring tests for the self-repo git mutation guard."""

import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.self_repo_guard as self_repo_guard


pytestmark = pytest.mark.asyncio


def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "hermes-agent"
    (root / ".git").mkdir(parents=True)
    return root.resolve()


async def _run(command, config, monkeypatch, repo_root, session_cwds=None, **kwargs):
    from tools.terminal_tool import terminal_tool

    monkeypatch.setattr(self_repo_guard, "get_running_source_root", lambda: repo_root)
    mock_env = MagicMock()
    mock_env.execute = AsyncMock(return_value={"output": "ok", "returncode": 0})
    mock_env.cwd = config["cwd"]

    async def get_env_config():
        return config

    with ExitStack() as stack:
        stack.enter_context(patch("tools.terminal_tool._get_env_config", get_env_config))
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(
            patch("tools.terminal_tool._active_environments", {"default": mock_env})
        )
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(
            patch("tools.terminal_tool._session_cwds", session_cwds or {})
        )
        stack.enter_context(
            patch(
                "tools.terminal_tool.check_all_command_guards",
                new=AsyncMock(return_value={"approved": True}),
            )
        )
        result = json.loads(await terminal_tool(command=command, **kwargs))
    return result, mock_env


async def test_blocks_checkout_in_source_repo(repo, monkeypatch):
    config = _make_env_config(cwd=str(repo))
    result, env = await _run("git checkout pr-51020", config, monkeypatch, repo)
    assert result["status"] == "blocked"
    assert "mix module versions" in result["error"]
    assert str(repo) in result["error"]
    env.execute.assert_not_awaited()


async def test_force_cannot_bypass(repo, monkeypatch):
    config = _make_env_config(cwd=str(repo))
    result, env = await _run(
        "git reset --hard origin/main", config, monkeypatch, repo, force=True
    )
    assert result["status"] == "blocked"
    env.execute.assert_not_awaited()


async def test_workdir_targeting_repo_is_blocked(repo, monkeypatch, tmp_path):
    config = _make_env_config(cwd=str(tmp_path))
    result, env = await _run("git pull", config, monkeypatch, repo, workdir=str(repo))
    assert result["status"] == "blocked"
    env.execute.assert_not_awaited()


async def test_session_cwd_targeting_repo_is_blocked(repo, monkeypatch, tmp_path):
    config = _make_env_config(cwd=str(tmp_path))
    result, env = await _run(
        "git checkout main",
        config,
        monkeypatch,
        repo,
        session_cwds={"session-1": str(repo)},
        task_id="session-1",
    )
    assert result["status"] == "blocked"
    env.execute.assert_not_awaited()


async def test_readonly_git_passes_through(repo, monkeypatch):
    config = _make_env_config(cwd=str(repo))
    result, env = await _run("git status", config, monkeypatch, repo)
    assert result.get("status") != "blocked"
    env.execute.assert_awaited_once()


async def test_mutation_outside_repo_passes_through(repo, monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    config = _make_env_config(cwd=str(other))
    result, env = await _run("git checkout main", config, monkeypatch, repo)
    assert result.get("status") != "blocked"
    env.execute.assert_awaited_once()


async def test_packaged_install_passes_through(repo, monkeypatch):
    config = _make_env_config(cwd=str(repo))
    result, env = await _run("git checkout main", config, monkeypatch, None)
    assert result.get("status") != "blocked"
    env.execute.assert_awaited_once()
