"""Native-async skill usage/provenance sidecar parity tests."""

from __future__ import annotations

import asyncio
import errno
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
async def test_save_usage_keeps_upstream_best_effort_error_contract(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)

    async def fail_makedirs(_path, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(skill_usage.aiofiles.os, "makedirs", fail_makedirs)
    assert await skill_usage.save_usage({"review": {"use_count": 1}}) is None


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


@pytest.mark.asyncio
async def test_archive_create_directory_failure_preserves_error_tuple(
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
    original_makedirs = skill_usage.aiofiles.os.makedirs

    async def fail_archive_directory(path, **kwargs):
        if Path(path) == skills_dir / ".archive":
            raise OSError("read-only filesystem")
        await original_makedirs(path, **kwargs)

    monkeypatch.setattr(
        skill_usage.aiofiles.os,
        "makedirs",
        fail_archive_directory,
    )
    ok, message = await skill_usage.archive_skill("review")

    assert ok is False
    assert message == "failed to create archive dir: read-only filesystem"
    assert skill_dir.exists()


@pytest.mark.asyncio
async def test_archive_cross_device_fallback_preserves_tree_and_symlink(
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
    references = skill_dir / "references"
    references.mkdir()
    (references / "data.txt").write_text("payload", encoding="utf-8")
    try:
        (references / "alias.txt").symlink_to("data.txt")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)
    original_replace = skill_usage.aiofiles.os.replace

    async def cross_device_once(source, destination):
        if Path(source) == skill_dir:
            raise OSError(errno.EXDEV, "cross-device link")
        await original_replace(source, destination)

    monkeypatch.setattr(skill_usage.aiofiles.os, "replace", cross_device_once)
    ok, _message = await skill_usage.archive_skill("review")

    archived = skills_dir / ".archive" / "review"
    assert ok is True
    assert not skill_dir.exists()
    assert (archived / "references" / "data.txt").read_text() == "payload"
    assert (archived / "references" / "alias.txt").is_symlink()
    assert os.readlink(archived / "references" / "alias.txt") == "data.txt"


@pytest.mark.asyncio
async def test_restore_cross_device_fallback_preserves_upstream_success(
    monkeypatch,
    tmp_path,
):
    skills_dir = tmp_path / "skills"
    archived = skills_dir / ".archive" / "review"
    archived.mkdir(parents=True)
    (archived / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review.\n---\n\n# Review\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)
    original_replace = skill_usage.aiofiles.os.replace

    async def cross_device_once(source, destination):
        if Path(source) == archived:
            raise OSError(errno.EXDEV, "cross-device link")
        await original_replace(source, destination)

    monkeypatch.setattr(skill_usage.aiofiles.os, "replace", cross_device_once)
    ok, message = await skill_usage.restore_skill("review")

    assert ok is True
    assert message == f"restored to {skills_dir / 'review'}"
    assert not archived.exists()
    assert (skills_dir / "review" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_suppression_roundtrip_is_durable(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
    await skill_usage.add_suppressed_name("bundled-one")
    await skill_usage.add_suppressed_name("bundled-two")
    assert await skill_usage.read_suppressed_names() == {
        "bundled-one",
        "bundled-two",
    }
    await skill_usage.remove_suppressed_name("bundled-one")
    assert await skill_usage.read_suppressed_names() == {"bundled-two"}


@pytest.mark.asyncio
async def test_builtin_archive_and_restore_update_suppression(
    monkeypatch, tmp_path
):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "bundled-one"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: bundled-one\ndescription: Bundled.\n---\n\n# Skill\n",
        encoding="utf-8",
    )
    (skills_dir / ".bundled_manifest").write_text(
        "bundled-one:abc\n", encoding="utf-8"
    )
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)

    async def pruning_enabled():
        return True

    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", pruning_enabled)
    archived, _ = await skill_usage.archive_skill("bundled-one")
    assert archived is True
    assert await skill_usage.read_suppressed_names() == {"bundled-one"}
    restored, _ = await skill_usage.restore_skill("bundled-one")
    assert restored is True
    assert await skill_usage.read_suppressed_names() == set()
    record = await skill_usage.get_record("bundled-one")
    assert record["state"] == skill_usage.STATE_ACTIVE
    assert record["archived_at"] is None


@pytest.mark.asyncio
async def test_sync_state_is_opt_in_and_curation_gated(monkeypatch, tmp_path):
    skills_dir = tmp_path / "skills"
    local = skills_dir / "local"
    local.mkdir(parents=True)
    (local / "SKILL.md").write_text(
        "---\nname: local\ndescription: Local.\n---\n\n# Local\n",
        encoding="utf-8",
    )
    hub = skills_dir / "hubbed"
    hub.mkdir(parents=True)
    (hub / "SKILL.md").write_text(
        "---\nname: hubbed\ndescription: Hub.\n---\n\n# Hub\n",
        encoding="utf-8",
    )
    lock_dir = skills_dir / ".hub"
    lock_dir.mkdir()
    (lock_dir / "lock.json").write_text(
        json.dumps({"installed": {"hubbed": {}}}), encoding="utf-8"
    )
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)
    assert await skill_usage.is_sync_enabled("local") is False
    await skill_usage.set_sync("local", True)
    await skill_usage.set_sync("hubbed", True)
    assert await skill_usage.is_sync_enabled("local") is True
    assert await skill_usage.is_sync_enabled("hubbed") is False


@pytest.mark.asyncio
async def test_reports_preserve_activity_and_provenance(monkeypatch, tmp_path):
    skills_dir = tmp_path / "skills"
    for name in ("managed", "unmanaged"):
        directory = skills_dir / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test.\n---\n\n# Test\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_dir)
    await skill_usage.mark_agent_created("managed")
    await skill_usage.bump_use("managed")
    curated = await skill_usage.curated_report()
    usage = await skill_usage.usage_report()
    unmanaged = await skill_usage.unmanaged_report()
    assert [row["name"] for row in curated] == ["managed"]
    assert curated[0]["activity_count"] == 1
    assert curated[0]["provenance"] == "agent"
    assert {row["name"] for row in usage} == {"managed", "unmanaged"}
    assert [row["name"] for row in unmanaged] == ["unmanaged"]
