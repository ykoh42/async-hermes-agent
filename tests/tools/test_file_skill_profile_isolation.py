"""Profile isolation for file metadata and skill-view repeat state."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiofiles
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import file_state, file_tools, skills_tool


@pytest.fixture(autouse=True)
def clear_scoped_state():
    file_state.get_registry().clear()
    file_tools._read_tracker_states.clear()
    file_tools._patch_failure_tracker_states.clear()
    file_tools._max_read_chars_by_profile.clear()
    skills_tool._skill_view_tracker_states.clear()
    skills_tool._SKILLS_CACHE.clear()
    yield
    file_state.get_registry().clear()
    file_tools._read_tracker_states.clear()
    file_tools._patch_failure_tracker_states.clear()
    file_tools._max_read_chars_by_profile.clear()
    skills_tool._skill_view_tracker_states.clear()
    skills_tool._SKILLS_CACHE.clear()


@pytest.mark.asyncio
async def test_read_dedup_and_patch_failures_do_not_cross_profiles(
    tmp_path: Path,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    shared = tmp_path / "shared.txt"
    shared.write_text("shared content\n", encoding="utf-8")

    async def first_read(profile: Path) -> dict:
        token = set_hermes_home_override(profile)
        try:
            return json.loads(
                await file_tools.read_file_tool(str(shared), task_id="same-task")
            )
        finally:
            reset_hermes_home_override(token)

    read_a, read_b = await asyncio.gather(
        first_read(profile_a),
        first_read(profile_b),
    )
    assert "shared content" in read_a["content"]
    assert "shared content" in read_b["content"]
    assert read_a.get("dedup") is not True
    assert read_b.get("dedup") is not True

    token = set_hermes_home_override(profile_a)
    try:
        await file_tools._activate_file_tracker_scope()
        assert file_tools._record_patch_failure("same-task", str(shared)) == 1
        assert file_tools._record_patch_failure("same-task", str(shared)) == 2
        assert file_tools._record_patch_failure("same-task", str(shared)) == 3
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_b)
    try:
        await file_tools._activate_file_tracker_scope()
        assert file_tools._record_patch_failure("same-task", str(shared)) == 1
        file_tools.clear_file_ops_cache("same-task")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_a)
    try:
        await file_tools._activate_file_tracker_scope()
        assert file_tools._record_patch_failure("same-task", str(shared)) == 4
        assert str(shared) in file_state.known_reads("same-task")
        repeated = json.loads(
            await file_tools.read_file_tool(str(shared), task_id="same-task")
        )
        assert repeated["status"] == "unchanged"
        assert repeated["dedup"] is True
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_file_read_character_budget_is_profile_scoped(
    monkeypatch,
    tmp_path: Path,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    file_tools._max_read_chars_by_profile.clear()

    async def load_config():
        from hermes_constants import get_hermes_home

        return {
            "file_read_max_chars": 111 if get_hermes_home() == profile_a else 222
        }

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", load_config)

    async def budget(profile: Path) -> int:
        token = set_hermes_home_override(profile)
        try:
            return await file_tools._get_max_read_chars()
        finally:
            reset_hermes_home_override(token)

    assert await asyncio.gather(budget(profile_a), budget(profile_b)) == [111, 222]
    assert await asyncio.gather(budget(profile_a), budget(profile_b)) == [111, 222]


@pytest.mark.asyncio
async def test_file_state_same_task_write_in_other_profile_stays_external(
    tmp_path: Path,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    shared = tmp_path / "shared.txt"
    shared.write_text("before\n", encoding="utf-8")

    token = set_hermes_home_override(profile_a)
    try:
        await file_state.record_read("same-task", shared)
        assert str(shared) in file_state.known_reads("same-task")
    finally:
        reset_hermes_home_override(token)

    await asyncio.sleep(0.01)
    async with aiofiles.open(shared, "w", encoding="utf-8") as handle:
        await handle.write("profile B write\n")

    token = set_hermes_home_override(profile_b)
    try:
        await file_state.note_write("same-task", shared)
        assert await file_state.check_stale("same-task", shared) is None
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_a)
    try:
        warning = await file_state.check_stale("same-task", shared)
        assert warning is not None
        assert "external edit" in warning
        assert str(shared) in file_state.known_reads("same-task")
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_path_lock_still_coordinates_shared_physical_file_across_profiles(
    tmp_path: Path,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    shared = str(tmp_path / "shared.txt")
    acquired_a = asyncio.Event()
    release_a = asyncio.Event()
    acquired_b = asyncio.Event()

    async def hold_a() -> None:
        token = set_hermes_home_override(profile_a)
        try:
            async with file_state.lock_path(shared):
                acquired_a.set()
                await release_a.wait()
        finally:
            reset_hermes_home_override(token)

    async def enter_b() -> None:
        token = set_hermes_home_override(profile_b)
        try:
            async with file_state.lock_path(shared):
                acquired_b.set()
        finally:
            reset_hermes_home_override(token)

    holder = asyncio.create_task(hold_a())
    await acquired_a.wait()
    waiter = asyncio.create_task(enter_b())
    await asyncio.sleep(0)
    assert not acquired_b.is_set()
    release_a.set()
    await asyncio.gather(holder, waiter)
    assert acquired_b.is_set()


def _write_skill(home: Path, marker: str) -> None:
    skill = home / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Profile demo.\n---\n"
        f"# Demo\n\n{marker}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_skill_view_dedup_and_reset_are_profile_scoped(tmp_path: Path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_skill(profile_a, "instructions from A")
    _write_skill(profile_b, "instructions from B")

    async def view(profile: Path) -> dict:
        token = set_hermes_home_override(profile)
        try:
            return json.loads(
                await skills_tool._handle_skill_view(
                    {"name": "demo"},
                    task_id="same-task",
                )
            )
        finally:
            reset_hermes_home_override(token)

    first_a, first_b = await asyncio.gather(view(profile_a), view(profile_b))
    assert "instructions from A" in first_a["content"]
    assert "instructions from B" in first_b["content"]

    repeated_a, repeated_b = await asyncio.gather(view(profile_a), view(profile_b))
    assert repeated_a["status"] == "unchanged"
    assert repeated_b["status"] == "unchanged"

    token = set_hermes_home_override(profile_b)
    try:
        skills_tool.reset_skill_view_dedup("same-task")
    finally:
        reset_hermes_home_override(token)

    refreshed_b = await view(profile_b)
    still_cached_a = await view(profile_a)
    assert "instructions from B" in refreshed_b["content"]
    assert still_cached_a["status"] == "unchanged"


@pytest.mark.asyncio
async def test_canonical_profile_alias_shares_skill_dedup(tmp_path: Path):
    profile = tmp_path / "profile"
    _write_skill(profile, "canonical instructions")
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)

    token = set_hermes_home_override(profile)
    try:
        first = json.loads(
            await skills_tool._handle_skill_view(
                {"name": "demo"},
                task_id="same-task",
            )
        )
    finally:
        reset_hermes_home_override(token)
    assert "canonical instructions" in first["content"]

    token = set_hermes_home_override(alias)
    try:
        repeated = json.loads(
            await skills_tool._handle_skill_view(
                {"name": "demo"},
                task_id="same-task",
            )
        )
    finally:
        reset_hermes_home_override(token)
    assert repeated["status"] == "unchanged"
