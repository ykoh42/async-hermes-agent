"""Profile-scoped persistent environment snapshot stores."""

from __future__ import annotations

import asyncio

import aiofiles.os
import pytest

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.environments import modal, singularity


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "filename"),
    [
        (modal, "modal_snapshots.json"),
        (singularity, "singularity_snapshots.json"),
    ],
)
async def test_snapshot_store_resolves_active_profile_at_each_call(
    module,
    filename,
    tmp_path,
):
    async def round_trip(profile, value):
        token = set_hermes_home_override(profile)
        try:
            await module._save_snapshots({"owner": value})
            await asyncio.sleep(0)
            return await module._load_snapshots()
        finally:
            reset_hermes_home_override(token)

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    result_a, result_b = await asyncio.gather(
        round_trip(profile_a, "alpha"),
        round_trip(profile_b, "beta"),
    )

    assert result_a == {"owner": "alpha"}
    assert result_b == {"owner": "beta"}
    assert await aiofiles.os.path.isfile(profile_a / filename)
    assert await aiofiles.os.path.isfile(profile_b / filename)


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [modal, singularity])
async def test_snapshot_store_constant_monkeypatch_remains_supported(
    module,
    monkeypatch,
    tmp_path,
):
    explicit = tmp_path / "explicit" / "snapshots.json"
    monkeypatch.setattr(module, "_SNAPSHOT_STORE", explicit)
    token = set_hermes_home_override(tmp_path / "other-profile")
    try:
        await module._save_snapshots({"explicit": True})
        assert await module._load_snapshots() == {"explicit": True}
    finally:
        reset_hermes_home_override(token)

    assert await aiofiles.os.path.isfile(explicit)


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [modal, singularity])
async def test_concurrent_snapshot_updates_preserve_both_tasks(
    module,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(module, "_SNAPSHOT_STORE", tmp_path / "snapshots.json")
    backing: dict[str, str] = {}
    active_loads = 0
    max_active_loads = 0

    async def load():
        nonlocal active_loads, max_active_loads
        snapshot = dict(backing)
        active_loads += 1
        max_active_loads = max(max_active_loads, active_loads)
        await asyncio.sleep(0)
        active_loads -= 1
        return snapshot

    async def save(data):
        await asyncio.sleep(0)
        backing.clear()
        backing.update(data)

    monkeypatch.setattr(module, "_load_snapshots", load)
    monkeypatch.setattr(module, "_save_snapshots", save)

    if module is modal:
        await asyncio.gather(
            module._store_direct_snapshot("task-a", "snapshot-a"),
            module._store_direct_snapshot("task-b", "snapshot-b"),
        )
        expected = {
            "direct:task-a": "snapshot-a",
            "direct:task-b": "snapshot-b",
        }
    else:
        await asyncio.gather(
            module._store_snapshot("task-a", "overlay-a"),
            module._store_snapshot("task-b", "overlay-b"),
        )
        expected = {"task-a": "overlay-a", "task-b": "overlay-b"}

    assert backing == expected
    assert max_active_loads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [modal, singularity])
async def test_canonical_profile_aliases_share_snapshot_update_lock(
    module,
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)
    backing: dict[str, str] = {}
    active_loads = 0
    max_active_loads = 0

    async def load():
        nonlocal active_loads, max_active_loads
        snapshot = dict(backing)
        active_loads += 1
        max_active_loads = max(max_active_loads, active_loads)
        await asyncio.sleep(0)
        active_loads -= 1
        return snapshot

    async def save(data):
        await asyncio.sleep(0)
        backing.clear()
        backing.update(data)

    monkeypatch.setattr(module, "_load_snapshots", load)
    monkeypatch.setattr(module, "_save_snapshots", save)

    async def update(active_profile, task_id):
        token = set_hermes_home_override(active_profile)
        try:
            if module is modal:
                await module._store_direct_snapshot(task_id, task_id)
            else:
                await module._store_snapshot(task_id, task_id)
        finally:
            reset_hermes_home_override(token)

    await asyncio.gather(update(profile, "task-a"), update(alias, "task-b"))
    assert len(backing) == 2
    assert max_active_loads == 1


@pytest.mark.parametrize("module", [modal, singularity])
def test_snapshot_update_lock_is_safe_across_sequential_event_loops(
    module,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(module, "_SNAPSHOT_STORE", tmp_path / "snapshots.json")
    backing: dict[str, str] = {}

    async def load():
        await asyncio.sleep(0)
        return dict(backing)

    async def save(data):
        await asyncio.sleep(0)
        backing.clear()
        backing.update(data)

    monkeypatch.setattr(module, "_load_snapshots", load)
    monkeypatch.setattr(module, "_save_snapshots", save)

    async def update(prefix):
        if module is modal:
            await asyncio.gather(
                module._store_direct_snapshot(f"{prefix}-a", "a"),
                module._store_direct_snapshot(f"{prefix}-b", "b"),
            )
        else:
            await asyncio.gather(
                module._store_snapshot(f"{prefix}-a", "a"),
                module._store_snapshot(f"{prefix}-b", "b"),
            )

    asyncio.run(update("first"))
    asyncio.run(update("second"))
    assert len(backing) == 4
