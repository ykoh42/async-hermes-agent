"""Behavioral contract for the retained native-async environment surface."""

from __future__ import annotations

import asyncio
import inspect
import json
import time

import pytest


def test_base_environment_public_import_and_coroutine_lifecycle():
    from tools.environments import BaseEnvironment

    assert inspect.iscoroutinefunction(BaseEnvironment.execute)
    assert inspect.iscoroutinefunction(BaseEnvironment.cleanup)
    assert inspect.iscoroutinefunction(BaseEnvironment.stop)


def test_pure_terminal_state_helpers_remain_synchronous():
    import tools.terminal_tool as terminal

    for name in (
        "record_session_cwd",
        "get_session_cwd",
        "register_task_env_overrides",
        "clear_task_env_overrides",
        "resolve_task_overrides",
        "get_active_env",
        "is_persistent_env",
    ):
        assert not inspect.iscoroutinefunction(getattr(terminal, name)), name


def test_environment_factory_keeps_upstream_parameter_contract():
    import tools.terminal_tool as terminal

    signature = inspect.signature(terminal._create_environment)
    assert list(signature.parameters) == [
        "env_type",
        "image",
        "cwd",
        "timeout",
        "ssh_config",
        "container_config",
        "local_config",
        "task_id",
        "host_cwd",
    ]
    assert signature.parameters["task_id"].default == "default"


def test_singularity_factory_preserves_upstream_resource_mapping():
    import tools.terminal_tool as terminal

    environment = terminal._create_environment(
        "singularity",
        "docker://python:3.11",
        "/root",
        180,
        container_config={
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 8192,
            "container_persistent": True,
        },
        task_id="rollout",
    )

    assert environment.image == "docker://python:3.11"
    assert environment._cpu == 2
    assert environment._memory == 4096
    assert environment._disk == 8192
    assert environment._persistent is True
    assert environment._task_id == "rollout"


def test_ssh_factory_preserves_upstream_configuration_mapping():
    import tools.terminal_tool as terminal

    environment = terminal._create_environment(
        "ssh",
        "",
        "~",
        180,
        ssh_config={
            "host": "example.com",
            "user": "hermes",
            "port": 2200,
            "key": "/tmp/id_ed25519",
        },
    )

    assert environment.host == "example.com"
    assert environment.user == "hermes"
    assert environment.port == 2200
    assert environment.key_path == "/tmp/id_ed25519"


def test_vercel_factory_preserves_upstream_resource_mapping():
    import tools.terminal_tool as terminal

    environment = terminal._create_environment(
        "vercel_sandbox",
        "",
        "/vercel/sandbox",
        180,
        container_config={
            "vercel_runtime": "node22",
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 51200,
            "container_persistent": True,
        },
        task_id="rollout",
    )

    assert environment._runtime == "node22"
    assert environment._cpu == 2
    assert environment._memory == 4096
    assert environment._persistent is True
    assert environment._task_id == "rollout"


def test_daytona_factory_preserves_upstream_resource_mapping():
    import tools.terminal_tool as terminal

    environment = terminal._create_environment(
        "daytona",
        "nikolaik/python-nodejs:python3.11-nodejs20",
        "/root",
        180,
        container_config={
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 8192,
            "container_persistent": True,
        },
        task_id="rollout",
    )

    assert environment._image == "nikolaik/python-nodejs:python3.11-nodejs20"
    assert environment._cpu == 2
    assert environment._memory == 4096
    assert environment._disk == 8192
    assert environment._persistent is True
    assert environment._task_id == "rollout"


@pytest.mark.asyncio
async def test_terminal_env_is_not_silently_downgraded_to_local(monkeypatch):
    import tools.terminal_tool as terminal

    await terminal.cleanup_all_environments()
    monkeypatch.setenv("TERMINAL_ENV", "not-a-backend")
    config = await terminal._get_env_config()
    assert config["env_type"] == "not-a-backend"
    result = json.loads(
        await terminal.terminal_tool(
            "printf should-not-run",
            task_id="unsupported-environment",
        )
    )
    assert result["status"] == "error"
    assert "Unknown environment type" in result["error"]
    assert "should-not-run" not in result["output"]


@pytest.mark.asyncio
async def test_cleanup_vm_awaits_backend_cleanup_before_reraising_cancellation():
    import tools.terminal_tool as terminal

    task_id = "cleanup-owned-environment"
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    class Environment:
        cwd = "/root"

        async def cleanup(self, *, force_remove: bool = False) -> None:
            assert force_remove is True
            started.set()
            await release.wait()
            completed.set()

    terminal.register_task_env_overrides(task_id, {"env_type": "docker"})
    with terminal._env_lock:
        terminal._active_environments[task_id] = Environment()
    cleanup = asyncio.create_task(terminal.cleanup_vm(task_id, force_remove=True))
    await started.wait()
    cleanup.cancel()
    await asyncio.sleep(0)
    cleanup.cancel()
    assert cleanup.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert completed.is_set()
    terminal.clear_task_env_overrides(task_id)


@pytest.mark.asyncio
async def test_idle_reaper_awaits_stale_environment_cleanup(monkeypatch):
    import tools.terminal_tool as terminal
    from tools.process_registry import process_registry

    task_id = "stale-environment"
    cleaned = asyncio.Event()

    class Environment:
        cwd = "/root"

        async def cleanup(self) -> None:
            cleaned.set()

    async def inactive(_task_id):
        return False

    monkeypatch.setattr(process_registry, "has_active_processes", inactive)
    terminal.register_task_env_overrides(task_id, {"env_type": "docker"})
    with terminal._env_lock:
        terminal._active_environments[task_id] = Environment()
        terminal._last_activity[task_id] = time.time() - 10

    assert await terminal._cleanup_inactive_envs(lifetime_seconds=1) == 1
    assert cleaned.is_set()
    with terminal._env_lock:
        assert task_id not in terminal._active_environments
    terminal.clear_task_env_overrides(task_id)


@pytest.mark.asyncio
async def test_idle_reaper_preserves_environment_with_background_process(monkeypatch):
    import tools.terminal_tool as terminal
    from tools.process_registry import process_registry

    task_id = "active-environment"

    class Environment:
        cwd = "/root"

        async def cleanup(self) -> None:
            raise AssertionError("active environment must not be cleaned")

    async def active(_task_id):
        return True

    monkeypatch.setattr(process_registry, "has_active_processes", active)
    terminal.register_task_env_overrides(task_id, {"env_type": "docker"})
    with terminal._env_lock:
        terminal._active_environments[task_id] = Environment()
        terminal._last_activity[task_id] = time.time() - 10

    assert await terminal._cleanup_inactive_envs(lifetime_seconds=1) == 0
    with terminal._env_lock:
        assert task_id in terminal._active_environments
        terminal._active_environments.pop(task_id)
        terminal._last_activity.pop(task_id, None)
    terminal.clear_task_env_overrides(task_id)


@pytest.mark.asyncio
async def test_remote_backend_name_reaches_command_guard(monkeypatch):
    import tools.terminal_tool as terminal

    seen: list[str] = []

    class Environment:
        cwd = "/root"

    async def get_environment(_task_id):
        return Environment()

    async def guard(_command, backend, **_kwargs):
        seen.append(backend)
        return {"approved": False, "message": "blocked for test"}

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setattr(terminal, "check_all_command_guards", guard)
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)

    result = json.loads(
        await terminal.terminal_tool("true", task_id="remote-guard")
    )
    assert result["status"] == "blocked"
    assert seen == ["ssh"]


@pytest.mark.asyncio
async def test_remote_background_uses_environment_process_registry(monkeypatch):
    import tools.terminal_tool as terminal
    from tools.process_registry import ProcessSession, process_registry

    class Environment:
        cwd = "/root"
        env = {}

    environment = Environment()
    captured: dict = {}

    async def config():
        return {"env_type": "ssh"}

    async def get_environment(_task_id):
        return environment

    async def approve(*_args, **_kwargs):
        return {"approved": True}

    async def spawn_via_env(**kwargs):
        captured.update(kwargs)
        return ProcessSession(
            "proc_remote",
            kwargs["command"],
            task_id=kwargs["task_id"],
            session_key=kwargs["session_key"],
            pid=4242,
            env_ref=kwargs["env"],
            cwd=kwargs["cwd"],
            started_at=1.0,
            pid_scope="sandbox",
        )

    monkeypatch.setattr(terminal, "_get_env_config", config)
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)
    monkeypatch.setattr(terminal, "check_all_command_guards", approve)
    monkeypatch.setattr(process_registry, "spawn_via_env", spawn_via_env)

    result = json.loads(
        await terminal.terminal_tool(
            "python server.py",
            background=True,
            task_id="remote-task",
            session_id="session-a",
        )
    )

    assert result["session_id"] == "proc_remote"
    assert result["pid"] == 4242
    assert captured["env"] is environment
    assert captured["command"] == "python server.py"
    assert captured["cwd"] == "/root"
