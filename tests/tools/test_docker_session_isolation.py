"""Retained upstream Docker session-isolation contract.

The upstream tests are synchronous because they exercise lexical state
resolution.  The native-async terminal implementation keeps those helpers
synchronous and performs filesystem/network work only at awaited boundaries.
"""

from __future__ import annotations

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_state():
    before_overrides = dict(terminal_tool._task_env_overrides)
    before_aliases = dict(terminal_tool._container_aliases)
    before_cwds = dict(terminal_tool._session_cwds)
    terminal_tool._task_env_overrides.clear()
    terminal_tool._container_aliases.clear()
    terminal_tool._session_cwds.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before_overrides)
    terminal_tool._container_aliases.clear()
    terminal_tool._container_aliases.update(before_aliases)
    terminal_tool._session_cwds.clear()
    terminal_tool._session_cwds.update(before_cwds)


def _enable_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")


def _disable_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")


def test_persistent_docker_keeps_shared_default(monkeypatch):
    _disable_isolation(monkeypatch)
    assert terminal_tool._resolve_container_task_id("session-a") == "default"


def test_nonpersistent_docker_keys_by_session(monkeypatch):
    _enable_isolation(monkeypatch)
    assert terminal_tool._resolve_container_task_id("session-a") == "session-a"
    assert terminal_tool._resolve_container_task_id("session-b") == "session-b"


def test_backend_override_wins_over_session_mode(monkeypatch):
    _enable_isolation(monkeypatch)
    terminal_tool.register_task_env_overrides(
        "rollout", {"docker_image": "custom:latest"}
    )
    assert terminal_tool._resolve_container_task_id("rollout") == "rollout"


def test_cwd_only_override_stays_shared(monkeypatch, tmp_path):
    _enable_isolation(monkeypatch)
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(tmp_path), "cwd_source": "session"}
    )
    assert terminal_tool._resolve_container_task_id("session") == "session"


def test_subagent_alias_resolves_to_parent(monkeypatch):
    _enable_isolation(monkeypatch)
    terminal_tool.register_container_alias("child", "session-a")
    assert terminal_tool._resolve_container_task_id("child") == "session-a"


def test_alias_chain_and_cycle_terminate(monkeypatch):
    _enable_isolation(monkeypatch)
    terminal_tool.register_container_alias("child", "session-a")
    terminal_tool.register_container_alias("grandchild", "child")
    assert terminal_tool._resolve_container_task_id("grandchild") == "session-a"
    terminal_tool.register_container_alias("x", "y")
    terminal_tool.register_container_alias("y", "x")
    assert terminal_tool._resolve_container_task_id("x") in {"x", "y"}


def test_isolation_rejects_process_global_mount(monkeypatch, tmp_path):
    _enable_isolation(monkeypatch)
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": str(tmp_path),
    }
    assert terminal_tool._resolve_task_host_cwd(config, "fresh") is None


def test_isolation_accepts_session_attached_mount(monkeypatch, tmp_path):
    _enable_isolation(monkeypatch)
    workspace = tmp_path / "attached"
    workspace.mkdir()
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(workspace), "cwd_source": "session"}
    )
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": "/previous/workspace",
    }
    assert terminal_tool._resolve_task_host_cwd(config, "session") == str(workspace)


def test_isolation_rejects_process_tagged_override(monkeypatch, tmp_path):
    _enable_isolation(monkeypatch)
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(tmp_path), "cwd_source": "process"}
    )
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": str(tmp_path),
    }
    assert terminal_tool._resolve_task_host_cwd(config, "session") is None


def test_shared_mode_keeps_legacy_host_mount(monkeypatch, tmp_path):
    _disable_isolation(monkeypatch)
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": str(tmp_path),
    }
    assert terminal_tool._resolve_task_host_cwd(config, "session") == str(tmp_path)


def test_container_command_discards_host_recorded_cwd(monkeypatch):
    terminal_tool.record_session_cwd("session", "/Users/me/project")
    assert terminal_tool._resolve_command_cwd(
        workdir=None,
        default_cwd="/workspace",
        session_key="session",
        env_type="docker",
    ) == "/workspace"


def test_container_command_keeps_container_recorded_cwd(monkeypatch):
    terminal_tool.record_session_cwd("session", "/workspace/subdir")
    assert terminal_tool._resolve_command_cwd(
        workdir=None,
        default_cwd="/workspace",
        session_key="session",
        env_type="docker",
    ) == "/workspace/subdir"


def test_explicit_workdir_wins_on_container():
    terminal_tool.record_session_cwd("session", "/workspace/a")
    assert terminal_tool._resolve_command_cwd(
        workdir="/workspace/b",
        default_cwd="/workspace",
        session_key="session",
        env_type="docker",
    ) == "/workspace/b"
