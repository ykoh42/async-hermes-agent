"""Profile-safe subprocess environments for LSP servers and installers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.lsp import install as install_module
from agent.lsp.client import LSPClient
from agent.secret_scope import (
    UnscopedSecretError,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.env_passthrough import clear_env_passthrough, register_env_passthrough


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


async def _run_lsp_profile(home: Path, label: str) -> dict[str, str | None]:
    workspace = home / "workspace"
    workspace.mkdir(parents=True)
    capture = home / "lsp-env.json"
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"PROFILE_SCOPED_TOKEN": f"scoped-{label}"})
    client = LSPClient(
        server_id=f"mock-{label}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env={
            "MOCK_LSP_SCRIPT": "clean",
            "MOCK_LSP_ENV_CAPTURE": str(capture),
            "LSP_PROFILE_LABEL": label,
            "PROFILE_SCOPED_TOKEN": "foreign-process-value",
            "PATH": f"/{label}/bin",
            "HERMES_HOME": "/foreign/home",
            "OPENAI_API_KEY": f"openai-{label}",
            "ANTHROPIC_API_KEY": f"anthropic-{label}",
            "AUXILIARY_VISION_API_KEY": f"internal-{label}",
        },
        cwd=str(workspace),
    )
    try:
        await client.start()
        await client.shutdown()
        return json.loads(capture.read_text(encoding="utf-8"))
    finally:
        if client.is_running:
            await client.shutdown()
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_sequential_and_concurrent_mock_lsp_servers_get_active_profile_env(
    tmp_path,
    monkeypatch,
    concurrent,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "foreign-anthropic")
    monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "foreign-internal")
    monkeypatch.setenv("PROFILE_SCOPED_TOKEN", "foreign-scoped")
    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)
    try:
        if concurrent:
            env_a, env_b = await asyncio.gather(
                _run_lsp_profile(tmp_path / "profile-a", "a"),
                _run_lsp_profile(tmp_path / "profile-b", "b"),
            )
        else:
            env_a = await _run_lsp_profile(tmp_path / "profile-a", "a")
            env_b = await _run_lsp_profile(tmp_path / "profile-b", "b")
    finally:
        set_multiplex_active(False)
        clear_env_passthrough()

    assert env_a == {
        "home": str(tmp_path / "profile-a"),
        "path": "/a/bin",
        "label": "a",
        "scoped": "scoped-a",
        "openai": None,
        "anthropic": None,
        "internal": None,
    }
    assert env_b == {
        "home": str(tmp_path / "profile-b"),
        "path": "/b/bin",
        "label": "b",
        "scoped": "scoped-b",
        "openai": None,
        "anthropic": None,
        "internal": None,
    }


@pytest.mark.asyncio
async def test_lsp_spawn_fails_closed_without_multiplex_secret_scope(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PROFILE_SCOPED_TOKEN", "foreign-scoped")
    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)
    client = LSPClient(
        server_id="missing-scope",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        env={"PROFILE_SCOPED_TOKEN": "foreign-overlay"},
    )
    try:
        with pytest.raises(UnscopedSecretError):
            await client.start()
    finally:
        set_multiplex_active(False)
        clear_env_passthrough()

    assert client.state == "error"
    assert client._proc is None


async def _run_fake_installer(home: Path, label: str) -> dict[str, str | None]:
    home.mkdir()
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"PROFILE_SCOPED_TOKEN": f"scoped-{label}"})
    code = (
        "import json, os, sys; "
        "sys.stderr.write(json.dumps({"
        "'home': os.environ.get('HERMES_HOME'), "
        "'path': os.environ.get('PATH'), "
        "'label': os.environ.get('INSTALL_PROFILE_LABEL'), "
        "'scoped': os.environ.get('PROFILE_SCOPED_TOKEN'), "
        "'openai': os.environ.get('OPENAI_API_KEY'), "
        "'internal': os.environ.get('AUXILIARY_VISION_API_KEY')}))"
    )
    try:
        returncode, stderr = await install_module._run_install(
            [sys.executable, "-c", code],
            timeout=10,
            env={
                "INSTALL_PROFILE_LABEL": label,
                "PROFILE_SCOPED_TOKEN": "foreign-process-value",
                "PATH": f"/{label}/installer-bin",
                "HERMES_HOME": "/foreign/home",
                "OPENAI_API_KEY": f"openai-{label}",
                "AUXILIARY_VISION_API_KEY": f"internal-{label}",
            },
        )
        assert returncode == 0
        return json.loads(stderr)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_sequential_and_concurrent_installers_use_sanitized_profile_env(
    tmp_path,
    monkeypatch,
    concurrent,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai")
    monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "foreign-internal")
    monkeypatch.setenv("PROFILE_SCOPED_TOKEN", "foreign-scoped")
    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)
    try:
        if concurrent:
            env_a, env_b = await asyncio.gather(
                _run_fake_installer(tmp_path / "profile-a", "a"),
                _run_fake_installer(tmp_path / "profile-b", "b"),
            )
        else:
            env_a = await _run_fake_installer(tmp_path / "profile-a", "a")
            env_b = await _run_fake_installer(tmp_path / "profile-b", "b")
    finally:
        set_multiplex_active(False)
        clear_env_passthrough()

    assert env_a == {
        "home": str(tmp_path / "profile-a"),
        "path": "/a/installer-bin",
        "label": "a",
        "scoped": "scoped-a",
        "openai": None,
        "internal": None,
    }
    assert env_b == {
        "home": str(tmp_path / "profile-b"),
        "path": "/b/installer-bin",
        "label": "b",
        "scoped": "scoped-b",
        "openai": None,
        "internal": None,
    }


@pytest.mark.asyncio
async def test_installer_fails_closed_without_multiplex_secret_scope(
    monkeypatch,
):
    monkeypatch.setenv("PROFILE_SCOPED_TOKEN", "foreign-scoped")
    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)
    create = AsyncMock()
    monkeypatch.setattr(
        install_module.asyncio,
        "create_subprocess_exec",
        create,
    )
    try:
        with pytest.raises(UnscopedSecretError):
            await install_module._run_install(
                [sys.executable, "-c", "raise SystemExit(0)"],
                timeout=10,
                env={"PROFILE_SCOPED_TOKEN": "foreign-overlay"},
            )
    finally:
        set_multiplex_active(False)
        clear_env_passthrough()

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_npm_go_and_pip_all_use_common_installer_runner(
    tmp_path,
    monkeypatch,
):
    calls = []

    async def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return 1, "synthetic failure"

    async def fake_existing(name):
        return "/usr/bin/npm" if name == "npm" else None

    monkeypatch.setattr(install_module, "_run_install", fake_run)
    monkeypatch.setattr(install_module, "_existing_binary", fake_existing)
    monkeypatch.setattr(
        install_module,
        "_which",
        AsyncMock(return_value="/usr/bin/go"),
    )
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        await install_module._install_npm("fake-npm", "fake-npm")
        await install_module._install_go("example.invalid/gopls", "gopls")
        await install_module._install_pip("fake-pip", "fake-pip")
    finally:
        reset_hermes_home_override(token)

    assert [call[0][0] for call in calls] == [
        "/usr/bin/npm",
        "/usr/bin/go",
        sys.executable,
    ]
    assert "env" not in calls[0][1]
    assert calls[1][1]["env"] == {
        "GOBIN": str(tmp_path / "profile" / "lsp" / "bin")
    }
    assert "env" not in calls[2][1]


@pytest.mark.asyncio
async def test_repeatedly_cancelled_installer_reaps_process_group(monkeypatch):
    captured = []
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def capture_process(*args, **kwargs):
        process = await create_subprocess_exec(*args, **kwargs)
        captured.append((process, kwargs))
        return process

    monkeypatch.setattr(
        install_module.asyncio,
        "create_subprocess_exec",
        capture_process,
    )
    install_task = asyncio.create_task(
        install_module._run_install(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout=30,
        )
    )
    while not captured:
        await asyncio.sleep(0)

    install_task.cancel()
    await asyncio.sleep(0)
    install_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await install_task

    process, kwargs = captured[0]
    assert process.returncode is not None
    assert kwargs["start_new_session"] is (os.name == "posix")
    assert not {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("Task-")
        and task.get_coro().__qualname__.startswith("_finish_process")
    }
