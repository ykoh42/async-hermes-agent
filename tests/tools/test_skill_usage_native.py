"""Native-async skill usage/provenance sidecar parity tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
import tools.skill_usage as skill_usage


@pytest.mark.asyncio
async def test_concurrent_counter_updates_are_not_lost(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)

    await asyncio.gather(*(skill_usage.bump_view("review") for _ in range(40)))

    record = await skill_usage.get_record("review")
    assert record["view_count"] == 40
    assert record["last_viewed_at"]
    data = json.loads((tmp_path / ".usage.json").read_text(encoding="utf-8"))
    assert data["review"]["view_count"] == 40


@pytest.mark.asyncio
async def test_corrupt_sidecar_fails_closed_for_ownership(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
    (tmp_path / ".usage.json").write_text("not-json", encoding="utf-8")

    assert await skill_usage.is_curator_managed("review") is False


@pytest.mark.asyncio
async def test_background_marker_survives_reload(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)

    await skill_usage.mark_agent_created("review")

    assert await skill_usage.is_curator_managed("review") is True
    assert (await skill_usage.load_usage())["review"]["created_by"] == "agent"


@pytest.mark.asyncio
async def test_local_skill_lookup_uses_frontmatter_name(monkeypatch, tmp_path):
    skill_dir = tmp_path / "category" / "directory-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: canonical-name\ndescription: Canonical.\n---\n\n# Skill\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)

    assert await skill_usage._find_local_skill_dir("canonical-name") == skill_dir
    assert await skill_usage._find_local_skill_dir("directory-name") is None


@pytest.mark.asyncio
async def test_concurrent_profile_scopes_write_isolated_sidecars(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def write_profile(home, skill_name):
        token = set_hermes_home_override(home)
        try:
            await skill_usage.bump_view(skill_name)
        finally:
            reset_hermes_home_override(token)

    await asyncio.gather(
        write_profile(profile_a, "only-a"),
        write_profile(profile_b, "only-b"),
    )

    data_a = json.loads(
        (profile_a / "skills" / ".usage.json").read_text(encoding="utf-8")
    )
    data_b = json.loads(
        (profile_b / "skills" / ".usage.json").read_text(encoding="utf-8")
    )
    assert set(data_a) == {"only-a"}
    assert set(data_b) == {"only-b"}


@pytest.mark.asyncio
async def test_cross_process_counter_updates_are_not_lost(monkeypatch, tmp_path):
    source_root = Path(__file__).resolve().parents[2]
    script = """
import asyncio
from tools.skill_usage import bump_view

async def main():
    for _ in range(25):
        await bump_view("shared")

asyncio.run(main())
"""
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(tmp_path)
    environment["PYTHONPATH"] = str(source_root)
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            cwd=source_root,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(2)
    ]
    outputs = await asyncio.gather(*(process.communicate() for process in processes))
    assert [process.returncode for process in processes] == [0, 0], outputs

    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path / "skills")
    assert (await skill_usage.get_record("shared"))["view_count"] == 50


@pytest.mark.asyncio
async def test_usage_lock_cleanup_survives_repeated_cancellation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    original_close = skill_usage._os_close

    async def controlled_close(fd):
        close_started.set()
        await release_close.wait()
        await original_close(fd)

    monkeypatch.setattr(skill_usage, "_os_close", controlled_close)
    entered = asyncio.Event()
    hold_lock = asyncio.Event()

    async def acquire_and_wait():
        async with skill_usage._usage_file_lock():
            entered.set()
            await hold_lock.wait()

    task = asyncio.create_task(acquire_and_wait())
    await entered.wait()
    task.cancel()
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_archive_finishes_state_persistence_before_reraising_cancellation(
    monkeypatch,
    tmp_path,
):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review.\n---\n\n# Review\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)
    await skill_usage.mark_agent_created("review")

    save_started = asyncio.Event()
    release_save = asyncio.Event()
    original_save = skill_usage.save_usage

    async def controlled_save(data):
        save_started.set()
        await release_save.wait()
        await original_save(data)

    monkeypatch.setattr(skill_usage, "save_usage", controlled_save)
    task = asyncio.create_task(skill_usage.archive_skill("review"))
    await save_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    release_save.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not skill_dir.exists()
    assert (skills_dir / ".archive" / "review" / "SKILL.md").exists()
    assert (await skill_usage.get_record("review"))["state"] == skill_usage.STATE_ARCHIVED


@pytest.mark.asyncio
async def test_archive_preserves_upstream_result_and_collision_name(
    monkeypatch,
    tmp_path,
):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review.\n---\n\n# Review\n",
        encoding="utf-8",
    )
    archived = skills_dir / ".archive" / "review"
    archived.mkdir(parents=True)
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)

    ok, message = await skill_usage.archive_skill("review")

    assert ok is True
    assert message.startswith(f"archived to {archived}-")
    suffix = message.rsplit("-", 1)[-1]
    assert len(suffix) == 14
    assert suffix.isdigit()
