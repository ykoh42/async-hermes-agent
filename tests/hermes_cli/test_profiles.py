"""Tests for profile identity detection used by the agent runtime."""

from pathlib import Path

import pytest
from blockbuster import BlockBuster

from hermes_cli.profiles import get_active_profile_name


@pytest.mark.asyncio
async def test_default_profile(tmp_path: Path, monkeypatch) -> None:
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    assert await get_active_profile_name() == "default"


@pytest.mark.asyncio
async def test_named_profile(tmp_path: Path, monkeypatch) -> None:
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        profile_name = await get_active_profile_name()
    finally:
        blockbuster.deactivate()

    assert profile_name == "coder"


@pytest.mark.asyncio
async def test_custom_deployment_is_its_default_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_home = tmp_path / "deployment"
    custom_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setenv("HERMES_HOME", str(custom_home))

    assert await get_active_profile_name() == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["root", "profile"])
async def test_external_alias_into_native_tree_is_custom(
    tmp_path: Path,
    monkeypatch,
    target_kind: str,
) -> None:
    native_root = tmp_path / ".hermes"
    target = (
        native_root
        if target_kind == "root"
        else native_root / "profiles" / "coder"
    )
    target.mkdir(parents=True)
    alias = tmp_path / f"{target_kind}-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(alias))

    assert await get_active_profile_name() == "custom"


@pytest.mark.asyncio
async def test_native_alias_to_external_custom_root_is_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    native_root = tmp_path / ".hermes"
    external = tmp_path / "external"
    native_root.mkdir()
    external.mkdir()
    alias = native_root / "out"
    try:
        alias.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(alias))

    assert await get_active_profile_name() == "default"


@pytest.mark.asyncio
async def test_absolute_profile_does_not_require_a_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    deleted_cwd = tmp_path / "deleted-cwd"
    deleted_cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.chdir(deleted_cwd)
    deleted_cwd.rmdir()

    assert await get_active_profile_name() == "coder"
