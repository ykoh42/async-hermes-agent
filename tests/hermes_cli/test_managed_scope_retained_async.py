from __future__ import annotations

import ast
import asyncio
import gc
import os
import weakref
from pathlib import Path

import pytest
from blockbuster import BlockBuster


_LEGACY_IO_HELPERS = {
    "get_managed_dir",
    "load_managed_config",
    "load_managed_env",
    "apply_managed_overlay",
    "managed_config_keys",
    "is_key_managed",
    "is_env_managed",
}


def _reset_config_caches() -> None:
    import hermes_cli.config as config

    config._LOAD_CONFIG_CACHE.clear()
    config._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config._LAST_READONLY_CONFIG_SOURCES_BY_PATH.clear()
    config._ASYNC_CONFIG_LOCKS.clear()


def test_retained_runtime_does_not_call_legacy_sync_managed_helpers() -> None:
    """Runtime consumers must use the native readonly config/env paths.

    ``hermes_cli.config`` owns legacy synchronous mutation APIs, while
    ``hermes_logging`` owns an explicit synchronous setup API. Neither is an
    event-loop runtime caller, so exclude those owners from this retained-path
    guard rather than forcing their whole write/bootstrap graphs async here.
    """
    root = Path(__file__).resolve().parents[2]
    runtime_paths = [
        root / "run_agent.py",
        root / "hermes_state.py",
        root / "model_tools.py",
        *(root / "agent").rglob("*.py"),
        *(root / "tools").rglob("*.py"),
        *(root / "plugins").rglob("*.py"),
        root / "hermes_cli" / "nous_subscription.py",
        root / "hermes_cli" / "runtime_provider.py",
    ]
    violations: list[str] = []
    for path in runtime_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        managed_aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "hermes_cli.managed_scope"
            for alias in node.names
            if alias.name in _LEGACY_IO_HELPERS
        }
        module_aliases = {
            alias.asname or alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "hermes_cli.managed_scope"
        }
        module_aliases.update(
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "hermes_cli"
            for alias in node.names
            if alias.name == "managed_scope"
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in managed_aliases:
                violations.append(
                    f"{path.relative_to(root)}:{node.lineno}:{node.func.id}"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _LEGACY_IO_HELPERS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases | {"managed_scope"}
            ):
                violations.append(
                    f"{path.relative_to(root)}:{node.lineno}:{node.func.attr}"
                )
    assert violations == []


@pytest.mark.asyncio
async def test_readonly_loader_honors_managed_policy_without_legacy_io_helpers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import managed_scope
    from hermes_cli.config import load_config_readonly

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        "timezone: America/New_York\n"
        "approvals:\n"
        "  ask: [user-rule]\n",
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(
        "timezone: Asia/Tokyo\n"
        "approvals:\n"
        "  deny: [managed-rule]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_config_caches()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retained readonly loader called legacy sync managed I/O")

    for name in _LEGACY_IO_HELPERS:
        monkeypatch.setattr(managed_scope, name, forbidden)

    blocker = BlockBuster()
    blocker.activate()
    try:
        config = await load_config_readonly()
    finally:
        blocker.deactivate()

    assert config["timezone"] == "Asia/Tokyo"
    assert config["approvals"]["ask"] == ["user-rule"]
    assert config["approvals"]["deny"] == ["managed-rule"]


@pytest.mark.asyncio
async def test_managed_policy_edit_invalidates_readonly_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.config import load_config_readonly

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    managed_config = managed / "config.yaml"
    managed_config.write_text("timezone: Asia/Tokyo\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_config_caches()

    first = await load_config_readonly()
    prior_mtime = managed_config.stat().st_mtime_ns
    managed_config.write_text("timezone: Europe/Paris\n", encoding="utf-8")
    os.utime(managed_config, ns=(prior_mtime + 1_000_000, prior_mtime + 1_000_000))
    second = await load_config_readonly()

    assert first["timezone"] == "Asia/Tokyo"
    assert second["timezone"] == "Europe/Paris"
    assert second is not first


@pytest.mark.asyncio
async def test_cancelled_managed_stat_releases_readonly_lock_for_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiofiles.os

    from hermes_cli.config import load_config_readonly

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    managed_config = managed / "config.yaml"
    managed_config.write_text("timezone: Asia/Tokyo\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_config_caches()

    original_stat = aiofiles.os.stat
    stat_started = asyncio.Event()
    stat_gate = asyncio.Event()

    async def interrupted_stat(path, *args, **kwargs):
        if Path(path) == managed_config:
            stat_started.set()
            await stat_gate.wait()
        return await original_stat(path, *args, **kwargs)

    monkeypatch.setattr(aiofiles.os, "stat", interrupted_stat)
    task = asyncio.create_task(load_config_readonly())
    await stat_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    monkeypatch.setattr(aiofiles.os, "stat", original_stat)
    config = await asyncio.wait_for(load_config_readonly(), timeout=2)
    assert config["timezone"] == "Asia/Tokyo"


@pytest.mark.asyncio
async def test_concurrent_profiles_share_global_policy_without_user_cache_leak(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.config import load_config_readonly
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "approvals:\n  deny: [global-policy]\n",
        encoding="utf-8",
    )
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for index, home in enumerate(homes):
        home.mkdir()
        (home / "config.yaml").write_text(
            f"timezone: Profile/{index}\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_config_caches()

    async def load_profile(home: Path):
        token = set_hermes_home_override(home)
        try:
            await asyncio.sleep(0)
            return await load_config_readonly()
        finally:
            reset_hermes_home_override(token)

    first, second = await asyncio.gather(*(load_profile(home) for home in homes))

    assert first["timezone"] == "Profile/0"
    assert second["timezone"] == "Profile/1"
    assert first["approvals"]["deny"] == ["global-policy"]
    assert second["approvals"]["deny"] == ["global-policy"]


def test_readonly_managed_policy_is_profile_safe_across_event_loops(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.config import load_config_readonly

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "approvals:\n  deny: [global-policy]\n",
        encoding="utf-8",
    )
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for index, home in enumerate(homes):
        home.mkdir()
        (home / "config.yaml").write_text(
            f"timezone: Profile/{index}\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_config_caches()
    loop_refs: list[weakref.ReferenceType[asyncio.AbstractEventLoop]] = []
    results = []

    for home in homes:
        monkeypatch.setenv("HERMES_HOME", str(home))

        async def load_one():
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            return await load_config_readonly()

        results.append(asyncio.run(load_one()))

    gc.collect()
    assert [result["timezone"] for result in results] == ["Profile/0", "Profile/1"]
    assert all(
        result["approvals"]["deny"] == ["global-policy"] for result in results
    )
    assert all(loop_ref() is None for loop_ref in loop_refs)
