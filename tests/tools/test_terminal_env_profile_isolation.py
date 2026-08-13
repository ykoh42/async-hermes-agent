from __future__ import annotations

import asyncio
import json

import pytest

from agent import secret_scope
from tools import terminal_tool
from tools.environments import base as base_environment
from tools.environments import singularity


@pytest.fixture(autouse=True)
def _restore_secret_scope(monkeypatch: pytest.MonkeyPatch):
    token = secret_scope.set_secret_scope(None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    yield
    secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_terminal_env_settings_are_profile_scoped(monkeypatch) -> None:
    async def empty_config():
        return {}

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", empty_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", empty_config)
    monkeypatch.setenv("TERMINAL_ENV", "foreign-process-backend")
    monkeypatch.setenv("TERMINAL_SSH_KEY", "/foreign/process/key")

    async def resolve(label: str):
        token = secret_scope.set_secret_scope(
            {
                "TERMINAL_ENV": "ssh",
                "TERMINAL_SSH_HOST": f"{label}.example",
                "TERMINAL_SSH_USER": label,
                "TERMINAL_SSH_KEY": f"/profiles/{label}/id_ed25519",
                "TERMINAL_SSH_PERSISTENT": "true" if label == "a" else "",
                "TERMINAL_TIMEOUT": "41" if label == "a" else "42",
            }
        )
        try:
            config = await terminal_tool._get_env_config()
            await asyncio.sleep(0)
            return config
        finally:
            secret_scope.reset_secret_scope(token)

    profile_a, profile_b = await asyncio.gather(resolve("a"), resolve("b"))

    assert (profile_a["env_type"], profile_a["ssh_host"], profile_a["ssh_key"]) == (
        "ssh",
        "a.example",
        "/profiles/a/id_ed25519",
    )
    assert (profile_b["env_type"], profile_b["ssh_host"], profile_b["ssh_key"]) == (
        "ssh",
        "b.example",
        "/profiles/b/id_ed25519",
    )
    assert (profile_a["ssh_persistent"], profile_a["timeout"]) == (True, 41)
    assert (profile_b["ssh_persistent"], profile_b["timeout"]) == (False, 42)


@pytest.mark.asyncio
async def test_terminal_config_precedence_preserves_explicit_empty(
    monkeypatch,
) -> None:
    async def raw_config():
        return {"terminal": {"timeout": "51", "ssh_key": ""}}

    async def merged_config():
        return {
            "terminal": {
                "backend": "local",
                "timeout": 71,
                "lifetime_seconds": 333,
                "ssh_host": "merged.example",
                "ssh_key": "/merged/key",
            }
        }

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", raw_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", merged_config)

    token = secret_scope.set_secret_scope(
        {
            "TERMINAL_ENV": "ssh",
            "TERMINAL_TIMEOUT": "61",
            "TERMINAL_SSH_HOST": "scoped.example",
            "TERMINAL_SSH_KEY": "/scoped/key",
        }
    )
    try:
        config = await terminal_tool._get_env_config()
    finally:
        secret_scope.reset_secret_scope(token)

    assert config["env_type"] == "ssh"
    assert config["timeout"] == 51
    assert config["lifetime_seconds"] == 333
    assert config["ssh_host"] == "scoped.example"
    assert config["ssh_key"] == ""


@pytest.mark.asyncio
async def test_deleted_cwd_fallback_does_not_read_foreign_terminal_cwd(
    monkeypatch,
) -> None:
    async def empty_config():
        return {}

    async def deleted_cwd():
        raise FileNotFoundError

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", empty_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", empty_config)
    monkeypatch.setattr(terminal_tool.aiofiles.os, "getcwd", deleted_cwd)
    monkeypatch.setenv("TERMINAL_CWD", "/foreign/profile/workspace")
    monkeypatch.setenv("HOME", "/safe/profile-home")

    token = secret_scope.set_secret_scope({})
    try:
        config = await terminal_tool._get_env_config()
    finally:
        secret_scope.reset_secret_scope(token)

    assert config["cwd"] == "/safe/profile-home"


@pytest.mark.asyncio
async def test_terminal_env_fails_closed_without_profile_scope(monkeypatch) -> None:
    async def empty_config():
        return {}

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", empty_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", empty_config)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_KEY", "/foreign/process/key")

    with pytest.raises(secret_scope.UnscopedSecretError):
        await terminal_tool._get_env_config()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["", "   "])
async def test_explicit_empty_backend_does_not_fail_open_to_local(
    monkeypatch,
    backend: str,
) -> None:
    async def empty_config():
        return {}

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", empty_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", empty_config)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    token = secret_scope.set_secret_scope({"TERMINAL_ENV": backend})
    try:
        config = await terminal_tool._get_env_config()
        result = json.loads(
            await terminal_tool.terminal_tool(
                "printf should-not-run",
                task_id=f"empty-backend-{backend!r}",
            )
        )
    finally:
        secret_scope.reset_secret_scope(token)

    assert config["env_type"] == backend
    assert result["status"] == "error"
    assert "Unknown environment type" in result["error"]
    assert result["output"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd", ["", "   "])
async def test_explicit_empty_cwd_is_not_replaced_by_process_cwd(
    monkeypatch,
    cwd: str,
) -> None:
    async def empty_config():
        return {}

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", empty_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", empty_config)
    monkeypatch.setenv("TERMINAL_CWD", "/foreign/profile/workspace")

    token = secret_scope.set_secret_scope(
        {"TERMINAL_ENV": "local", "TERMINAL_CWD": cwd}
    )
    try:
        config = await terminal_tool._get_env_config()
    finally:
        secret_scope.reset_secret_scope(token)

    assert config["cwd"] == cwd


@pytest.mark.asyncio
async def test_single_profile_terminal_env_precedence_is_preserved(
    monkeypatch,
) -> None:
    async def empty_config():
        return {}

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", empty_config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", empty_config)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_KEY", "/legacy/process/key")
    monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", "false")

    token = secret_scope.set_secret_scope({})
    try:
        config = await terminal_tool._get_env_config()
    finally:
        secret_scope.reset_secret_scope(token)

    assert config["env_type"] == "ssh"
    assert config["ssh_key"] == "/legacy/process/key"
    assert config["ssh_persistent"] is False


def test_only_import_time_terminal_policy_is_process_global() -> None:
    assert secret_scope._is_global_env("TERMINAL_MAX_FOREGROUND_TIMEOUT")
    assert secret_scope._is_global_env("TERMINAL_DISK_WARNING_GB")
    assert not secret_scope._is_global_env("TERMINAL_CWD")
    assert not secret_scope._is_global_env("TERMINAL_TIMEOUT")
    assert not secret_scope._is_global_env("TERMINAL_SSH_PERSISTENT")
    assert not secret_scope._is_global_env("HERMES_TERMINAL_SECURITY_MODE")


def test_retained_terminal_env_parser_uses_active_profile(monkeypatch) -> None:
    monkeypatch.setenv("TERMINAL_TIMEOUT", "999")
    token = secret_scope.set_secret_scope({"TERMINAL_TIMEOUT": "37"})
    try:
        value = terminal_tool._parse_env_var("TERMINAL_TIMEOUT", "180")
    finally:
        secret_scope.reset_secret_scope(token)

    assert value == 37


@pytest.mark.asyncio
async def test_terminal_storage_directories_are_profile_scoped(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "TERMINAL_SANDBOX_DIR", str(tmp_path / "foreign-sandboxes")
    )
    monkeypatch.setenv(
        "TERMINAL_SCRATCH_DIR", str(tmp_path / "foreign-scratch")
    )

    async def resolve(label: str):
        sandbox = tmp_path / label / "sandboxes"
        scratch = tmp_path / label / "scratch"
        token = secret_scope.set_secret_scope(
            {
                "TERMINAL_SANDBOX_DIR": str(sandbox),
                "TERMINAL_SCRATCH_DIR": str(scratch),
            }
        )
        try:
            return await asyncio.gather(
                base_environment.get_sandbox_dir(),
                singularity._get_scratch_dir(),
            )
        finally:
            secret_scope.reset_secret_scope(token)

    profile_a, profile_b = await asyncio.gather(resolve("a"), resolve("b"))
    assert profile_a == [
        tmp_path / "a" / "sandboxes",
        tmp_path / "a" / "scratch",
    ]
    assert profile_b == [
        tmp_path / "b" / "sandboxes",
        tmp_path / "b" / "scratch",
    ]


@pytest.mark.asyncio
async def test_terminal_storage_directories_fail_closed_unscoped(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "foreign"))
    monkeypatch.setenv("TERMINAL_SCRATCH_DIR", str(tmp_path / "foreign"))

    with pytest.raises(secret_scope.UnscopedSecretError):
        await base_environment.get_sandbox_dir()
    with pytest.raises(secret_scope.UnscopedSecretError):
        await singularity._get_scratch_dir()
