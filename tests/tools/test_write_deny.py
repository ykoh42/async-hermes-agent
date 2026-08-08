"""Async write-deny boundary tests."""

import os
from pathlib import Path

import pytest

from agent.file_safety import get_write_denied_error


pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "path",
    [
        "/etc/shadow",
        "~/.ssh/authorized_keys",
        "~/.ssh/id_ed25519",
        "~/.netrc",
        "~/.pgpass",
        "~/.npmrc",
        "~/.pypirc",
    ],
)
async def test_credentials_and_system_paths_are_denied(path):
    assert await get_write_denied_error(path) is not None


async def test_profile_mode_still_denies_root_env(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    root_env = root / ".env"
    root_env.write_text("secret")
    monkeypatch.setenv("HERMES_HOME", str(profile))

    import agent.file_safety as safety

    monkeypatch.setattr(safety, "_hermes_home_path", lambda: profile)
    async def get_root():
        return root

    monkeypatch.setattr(safety, "_hermes_root_path", get_root)
    assert await get_write_denied_error(str(root_env)) is not None


async def test_shell_profiles_and_temp_files_are_allowed(tmp_path):
    home = Path.home()
    for name in (".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"):
        assert await get_write_denied_error(str(home / name)) is None
    assert await get_write_denied_error(str(tmp_path / "safe.txt")) is None
