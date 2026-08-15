"""Native-async tests for terminal_tool.ensure_task_env."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tools.terminal_tool as tt


pytestmark = pytest.mark.asyncio


def _config(env_type: str) -> dict:
    return {
        "env_type": env_type,
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "lifetime_seconds": 300,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": 22,
        "ssh_key": "",
    }


def _clear(task_id: str) -> None:
    tt._active_environments.pop(task_id, None)
    tt._last_activity.pop(task_id, None)


async def test_local_backend_is_noop(monkeypatch):
    async def get_config():
        return _config("local")

    monkeypatch.setattr(tt, "_get_env_config", get_config)
    with patch.object(tt, "_create_environment") as create:
        assert await tt.ensure_task_env("t-local") is None
    create.assert_not_called()


async def test_non_local_creates_and_reuses(monkeypatch):
    task_id = tt._resolve_container_task_id("t-ssh")
    _clear(task_id)
    fake = SimpleNamespace(
        cwd="/tmp",
        _ensure_initialized=AsyncMock(),
    )

    async def get_config():
        return _config("ssh")

    monkeypatch.setattr(tt, "_get_env_config", get_config)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda *_args: None)
    try:
        with patch.object(tt, "_create_environment", return_value=fake) as create:
            assert await tt.ensure_task_env("t-ssh") is fake
            create.assert_called_once()
            assert tt.get_active_env("t-ssh") is fake

            assert await tt.ensure_task_env("t-ssh") is fake
            create.assert_called_once()
    finally:
        _clear(task_id)


async def test_creation_failure_returns_none_and_caches_nothing(monkeypatch):
    task_id = tt._resolve_container_task_id("t-ssh-fail")
    _clear(task_id)

    async def get_config():
        return _config("ssh")

    monkeypatch.setattr(tt, "_get_env_config", get_config)
    try:
        with patch.object(tt, "_create_environment", side_effect=RuntimeError("boom")):
            assert await tt.ensure_task_env("t-ssh-fail") is None
        assert tt.get_active_env("t-ssh-fail") is None
    finally:
        _clear(task_id)
