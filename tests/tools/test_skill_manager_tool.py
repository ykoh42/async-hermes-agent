"""Behavior tests for the native-async skill manager."""

from __future__ import annotations

import asyncio
import errno
import gc
import inspect
import json
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import tools.skill_manager_tool as manager
import tools.skill_usage as skill_usage
import tools.skills_tool as skills_tool
from tools.registry import registry
from tools.skill_provenance import (
    BACKGROUND_REVIEW,
    reset_current_write_origin,
    set_current_write_origin,
)


VALID_SKILL = """\
---
name: test-skill
description: Use when testing skill management.
---

# Test skill

Run the original procedure.
"""

UPDATED_SKILL = """\
---
name: test-skill
description: Use when testing updated skill management.
---

# Test skill

Run the updated procedure.
"""


@asynccontextmanager
async def isolated_skills(monkeypatch: pytest.MonkeyPatch, root: Path):
    """Limit discovery and writes to one temporary skills root."""
    monkeypatch.setattr(manager, "SKILLS_DIR", root)

    async def get_roots() -> list[Path]:
        return [root]

    monkeypatch.setattr(manager, "get_all_skills_dirs", get_roots)
    yield


async def result_of(*args, **kwargs) -> dict:
    return json.loads(await manager.skill_manage(*args, **kwargs))


class TestWriteLockLifecycle:
    @pytest.mark.asyncio
    async def test_profiles_use_independent_write_locks(self, tmp_path):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile_a = tmp_path / "profile-a"
        profile_b = tmp_path / "profile-b"
        profile_a.mkdir()
        profile_b.mkdir()

        async def resolve(profile):
            token = set_hermes_home_override(profile)
            try:
                return await manager._active_skill_write_lock()
            finally:
                reset_hermes_home_override(token)

        lock_a, lock_b = await asyncio.gather(
            resolve(profile_a),
            resolve(profile_b),
        )
        assert lock_a is not lock_b

    @pytest.mark.asyncio
    async def test_symlink_profile_aliases_share_write_lock(self, tmp_path):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile = tmp_path / "profile"
        alias = tmp_path / "alias"
        profile.mkdir()
        alias.symlink_to(profile, target_is_directory=True)

        async def resolve(home):
            token = set_hermes_home_override(home)
            try:
                return await manager._active_skill_write_lock()
            finally:
                reset_hermes_home_override(token)

        assert await resolve(profile) is await resolve(alias)

    def test_write_lock_does_not_retain_closed_loop(self, tmp_path):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        loop_refs = []

        async def resolve(profile):
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            token = set_hermes_home_override(profile)
            try:
                async with await manager._active_skill_write_lock():
                    await asyncio.sleep(0)
            finally:
                reset_hermes_home_override(token)

        asyncio.run(resolve(tmp_path / "profile-a"))
        asyncio.run(resolve(tmp_path / "profile-b"))
        gc.collect()

        assert loop_refs[0]() is None


@asynccontextmanager
async def background_review_origin():
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        yield
    finally:
        reset_current_write_origin(token)


class TestValidation:
    def test_names_and_categories_reject_paths(self):
        assert manager._validate_name("valid-skill") is None
        assert "Invalid skill name" in manager._validate_name("../escape")
        assert manager._validate_category("coding") is None
        assert "Invalid category" in manager._validate_category("a/b")

    @pytest.mark.parametrize(
        "path",
        ["../SKILL.md", "/tmp/file", "C:\\temp\\file", "README.md"],
    )
    def test_supporting_files_reject_unsafe_paths(self, path):
        assert manager._validate_file_path(path) is not None

    def test_frontmatter_contract(self):
        assert manager._validate_frontmatter(VALID_SKILL, new_skill=True) is None
        assert "frontmatter" in manager._validate_frontmatter("# No metadata")


class TestAgentCreatedSecurityGuard:
    @pytest.mark.asyncio
    async def test_disabled_guard_short_circuits_scan(self, monkeypatch, tmp_path):
        enabled = AsyncMock(return_value=False)
        scan = AsyncMock()
        monkeypatch.setattr(manager, "_guard_agent_created_enabled", enabled)
        monkeypatch.setattr(manager, "scan_skill", scan)
        assert await manager._security_scan_skill(tmp_path) is None
        enabled.assert_awaited_once_with()
        scan.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dangerous_agent_skill_is_blocked(self, monkeypatch, tmp_path):
        from tools.skills_guard import Finding, ScanResult

        result = ScanResult(
            skill_name="bad",
            source="agent-created",
            trust_level="agent-created",
            verdict="dangerous",
            findings=[
                Finding(
                    "env_exfil_curl",
                    "critical",
                    "exfiltration",
                    "SKILL.md",
                    1,
                    "curl $TOKEN",
                    "exfiltration",
                )
            ],
        )
        monkeypatch.setattr(
            manager, "_guard_agent_created_enabled", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(manager, "scan_skill", AsyncMock(return_value=result))
        error = await manager._security_scan_skill(tmp_path)
        assert "Security scan blocked this skill" in error
        assert "env_exfil_curl" not in error  # report remains compact upstream shape

    @pytest.mark.asyncio
    async def test_config_error_disables_optional_guard(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(side_effect=RuntimeError("boom")),
        )
        assert await manager._guard_agent_created_enabled() is False

    @pytest.mark.asyncio
    async def test_create_rolls_back_when_guard_blocks(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            manager,
            "_security_scan_skill",
            AsyncMock(return_value="blocked by guard"),
        )
        async with isolated_skills(monkeypatch, tmp_path):
            result = await manager._create_skill("test-skill", VALID_SKILL)
        assert result == {"success": False, "error": "blocked by guard"}
        assert not (tmp_path / "test-skill").exists()


class TestSkillCrud:
    @pytest.mark.asyncio
    async def test_create_edit_patch_and_delete(self, monkeypatch, tmp_path):
        async with isolated_skills(monkeypatch, tmp_path):
            created = await result_of("create", "test-skill", VALID_SKILL)
            assert created["success"] is True
            skill_file = tmp_path / "test-skill" / "SKILL.md"
            assert await manager._read_text(skill_file) == VALID_SKILL

            duplicate = await result_of("create", "test-skill", VALID_SKILL)
            assert duplicate["success"] is False
            assert "already exists" in duplicate["error"]

            edited = await result_of("edit", "test-skill", UPDATED_SKILL)
            assert edited["success"] is True
            assert await manager._read_text(skill_file) == UPDATED_SKILL

            patched = await result_of(
                "patch",
                "test-skill",
                old_string="updated procedure",
                new_string="verified procedure",
            )
            assert patched["success"] is True
            assert "verified procedure" in await manager._read_text(skill_file)

            deleted = await result_of("delete", "test-skill")
            assert deleted["success"] is True
            assert not await manager.aiofiles.os.path.exists(skill_file.parent)

    @pytest.mark.asyncio
    async def test_category_cleanup_stops_at_skills_root(
        self,
        monkeypatch,
        tmp_path,
    ):
        async with isolated_skills(monkeypatch, tmp_path):
            result = await result_of(
                "create",
                "test-skill",
                VALID_SKILL,
                category="coding",
            )
            assert result["success"] is True
            result = await result_of("delete", "test-skill")
            assert result["success"] is True
            assert not (tmp_path / "coding").exists()
            assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_edit_existing_external_skill_in_place(
        self,
        monkeypatch,
        tmp_path,
    ):
        local_root = tmp_path / "local"
        external_root = tmp_path / "external"
        external_skill = external_root / "test-skill" / "SKILL.md"
        external_skill.parent.mkdir(parents=True)
        external_skill.write_text(VALID_SKILL)
        monkeypatch.setattr(manager, "SKILLS_DIR", local_root)

        async def get_roots() -> list[Path]:
            return [local_root, external_root]

        monkeypatch.setattr(manager, "get_all_skills_dirs", get_roots)
        result = await result_of("edit", "test-skill", UPDATED_SKILL)
        assert result["success"] is True
        assert external_skill.read_text() == UPDATED_SKILL
        assert not local_root.exists()


class TestSupportingFiles:
    @pytest.mark.asyncio
    async def test_write_patch_and_remove_file(self, monkeypatch, tmp_path):
        async with isolated_skills(monkeypatch, tmp_path):
            await result_of("create", "test-skill", VALID_SKILL)
            written = await result_of(
                "write_file",
                "test-skill",
                file_path="references/guide.md",
                file_content="original guidance",
            )
            assert written["success"] is True

            patched = await result_of(
                "patch",
                "test-skill",
                file_path="references/guide.md",
                old_string="original",
                new_string="updated",
            )
            assert patched["success"] is True
            target = tmp_path / "test-skill" / "references" / "guide.md"
            assert target.read_text() == "updated guidance"

            removed = await result_of(
                "remove_file",
                "test-skill",
                file_path="references/guide.md",
            )
            assert removed["success"] is True
            assert not target.exists()
            assert not target.parent.exists()

    @pytest.mark.asyncio
    async def test_symlink_cannot_escape_skill_directory(
        self,
        monkeypatch,
        tmp_path,
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "skills"
        async with isolated_skills(monkeypatch, root):
            await result_of("create", "test-skill", VALID_SKILL)
            (root / "test-skill" / "references").symlink_to(
                outside,
                target_is_directory=True,
            )
            result = await result_of(
                "write_file",
                "test-skill",
                file_path="references/escape.md",
                file_content="must not escape",
            )
            assert result["success"] is False
            assert "escapes allowed directory" in result["error"]
            assert not (outside / "escape.md").exists()

    @pytest.mark.asyncio
    async def test_skill_directory_symlink_is_not_deleted(
        self,
        monkeypatch,
        tmp_path,
    ):
        outside = tmp_path / "outside-skill"
        outside.mkdir()
        (outside / "SKILL.md").write_text(VALID_SKILL)
        root = tmp_path / "skills"
        root.mkdir()
        (root / "test-skill").symlink_to(outside, target_is_directory=True)
        async with isolated_skills(monkeypatch, root):
            result = await result_of("delete", "test-skill")
        assert result["success"] is False
        assert "symlink/junction" in result["error"]
        assert (outside / "SKILL.md").exists()


class TestBackgroundReviewOwnership:
    @pytest.mark.asyncio
    async def test_background_ownership_preflight_precedes_argument_validation(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            created = await result_of("create", "test-skill", VALID_SKILL)
            assert created["success"] is True

            async with background_review_origin():
                result = await result_of("edit", "test-skill")

        assert result["success"] is False
        assert "not curator-managed" in result["error"]
        assert "content is required" not in result["error"]

    @pytest.mark.asyncio
    async def test_user_owned_skill_is_off_limits_to_background_review(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            created = await result_of("create", "test-skill", VALID_SKILL)
            assert created["success"] is True

            async with background_review_origin():
                result = await result_of(
                    "patch",
                    "test-skill",
                    old_string="original procedure",
                    new_string="autonomous rewrite",
                )

        assert result["success"] is False
        assert "not curator-managed" in result["error"]
        assert "created_by=None" in result["error"]
        assert "original procedure" in (
            tmp_path / "test-skill" / "SKILL.md"
        ).read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_background_created_skill_requires_exact_file_read(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            async with background_review_origin():
                created = await result_of("create", "test-skill", VALID_SKILL)
                assert created["success"] is True
                assert await skill_usage.is_curator_managed("test-skill") is True

                blocked = await result_of(
                    "patch",
                    "test-skill",
                    old_string="original procedure",
                    new_string="verified procedure",
                )
                assert blocked["success"] is False
                assert blocked["_read_before_write_required"] is True

                await manager.mark_background_review_skill_read(
                    tmp_path / "test-skill" / "SKILL.md"
                )
                patched = await result_of(
                    "patch",
                    "test-skill",
                    old_string="original procedure",
                    new_string="verified procedure",
                )

        assert patched["success"] is True
        assert "verified procedure" in (
            tmp_path / "test-skill" / "SKILL.md"
        ).read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_read_mark_path_resolution_failure_uses_upstream_fallback(
        self,
        monkeypatch,
        tmp_path,
    ):
        async def unavailable_realpath(_path):
            raise OSError("realpath unavailable")

        monkeypatch.setattr(manager, "_realpath", unavailable_realpath)
        target = tmp_path / "test-skill" / "SKILL.md"
        async with background_review_origin():
            await manager.mark_background_review_skill_read(target)
            assert await manager._background_review_has_read(target) is True

    @pytest.mark.asyncio
    async def test_skill_view_marks_read_and_records_usage(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)

        async def roots() -> list[Path]:
            return [tmp_path]

        async def no_external_roots() -> list[Path]:
            return []

        monkeypatch.setattr(manager, "get_all_skills_dirs", roots)
        monkeypatch.setattr(skills_tool, "_external_skills_dirs", no_external_roots)
        monkeypatch.setattr(
            "agent.skill_utils.get_external_skills_dirs",
            no_external_roots,
        )

        async with background_review_origin():
            created = await result_of("create", "test-skill", VALID_SKILL)
            assert created["success"] is True
            manager._reset_background_review_read_marks()

            viewed = json.loads(
                await registry.dispatch(
                    "skill_view",
                    {"name": "test-skill"},
                    task_id="review-turn",
                )
            )
            assert viewed["success"] is True
            patched = await result_of(
                "patch",
                "test-skill",
                old_string="original procedure",
                new_string="verified procedure",
            )

        assert patched["success"] is True
        record = await skill_usage.get_record("test-skill")
        assert record["view_count"] == 1
        assert record["use_count"] == 1
        assert record["patch_count"] == 1

    @pytest.mark.asyncio
    async def test_support_file_overwrite_requires_that_exact_file_read(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            async with background_review_origin():
                assert (await result_of("create", "test-skill", VALID_SKILL))["success"]
                assert (
                    await result_of(
                        "write_file",
                        "test-skill",
                        file_path="references/workflow.md",
                        file_content="old workflow\n",
                    )
                )["success"]
                manager._reset_background_review_read_marks()
                await manager.mark_background_review_skill_read(
                    tmp_path / "test-skill" / "SKILL.md"
                )

                blocked = await result_of(
                    "write_file",
                    "test-skill",
                    file_path="references/workflow.md",
                    file_content="new workflow\n",
                )
                assert blocked["success"] is False
                assert blocked["_read_before_write_required"] is True

                await manager.mark_background_review_skill_read(
                    tmp_path / "test-skill" / "references" / "workflow.md"
                )
                allowed = await result_of(
                    "write_file",
                    "test-skill",
                    file_path="references/workflow.md",
                    file_content="new workflow\n",
                )

        assert allowed["success"] is True
        assert (
            tmp_path / "test-skill" / "references" / "workflow.md"
        ).read_text(encoding="utf-8") == "new workflow\n"

    @pytest.mark.asyncio
    async def test_background_review_cannot_mutate_external_skill(
        self,
        monkeypatch,
        tmp_path,
    ):
        local_root = tmp_path / "local"
        external_root = tmp_path / "external"
        external_skill = external_root / "test-skill" / "SKILL.md"
        external_skill.parent.mkdir(parents=True)
        external_skill.write_text(VALID_SKILL, encoding="utf-8")
        monkeypatch.setattr(manager, "SKILLS_DIR", local_root)
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: local_root)

        async def get_roots() -> list[Path]:
            return [local_root, external_root]

        async def get_external_roots() -> list[Path]:
            return [external_root]

        monkeypatch.setattr(manager, "get_all_skills_dirs", get_roots)
        monkeypatch.setattr(
            "agent.skill_utils.get_external_skills_dirs",
            get_external_roots,
        )

        async with background_review_origin():
            result = await result_of(
                "patch",
                "test-skill",
                old_string="original procedure",
                new_string="autonomous rewrite",
            )

        assert result["success"] is False
        assert "external_dirs" in result["error"]
        assert "original procedure" in external_skill.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_pinned_skill_cannot_be_deleted(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            assert (await result_of("create", "test-skill", VALID_SKILL))["success"]
            await skill_usage.set_pinned("test-skill", True)
            result = await result_of("delete", "test-skill")

        assert result["success"] is False
        assert "pinned" in result["error"]
        assert (tmp_path / "test-skill" / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_verified_background_consolidation_is_recoverably_archived(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            async with background_review_origin():
                assert (await result_of("create", "test-skill", VALID_SKILL))["success"]
                umbrella = VALID_SKILL.replace("test-skill", "umbrella")
                assert (await result_of("create", "umbrella", umbrella))["success"]

                result = await result_of(
                    "delete",
                    "test-skill",
                    absorbed_into="umbrella",
                )

        assert result["success"] is True
        assert result["_archived"] is True
        assert not (tmp_path / "test-skill").exists()
        assert (tmp_path / ".archive" / "test-skill" / "SKILL.md").exists()
        record = await skill_usage.get_record("test-skill")
        assert record["state"] == skill_usage.STATE_ARCHIVED
        assert record["archived_at"]

    @pytest.mark.asyncio
    async def test_background_archive_failure_keeps_tool_error_shape(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(skill_usage, "_skills_dir", lambda: tmp_path)
        async with isolated_skills(monkeypatch, tmp_path):
            async with background_review_origin():
                assert (await result_of("create", "test-skill", VALID_SKILL))["success"]
                umbrella = VALID_SKILL.replace("test-skill", "umbrella")
                assert (await result_of("create", "umbrella", umbrella))["success"]

                async def fail_archive(_name):
                    raise OSError("disk unavailable")

                monkeypatch.setattr(skill_usage, "archive_skill", fail_archive)
                result = await result_of(
                    "delete",
                    "test-skill",
                    absorbed_into="umbrella",
                )

        assert result == {
            "success": False,
            "error": "failed to archive 'test-skill': disk unavailable",
        }


class TestAsyncContract:
    def test_public_function_and_registry_handler_are_coroutines(self):
        entry = registry.get_entry("skill_manage")
        assert inspect.iscoroutinefunction(manager.skill_manage)
        assert entry is not None
        assert inspect.iscoroutinefunction(entry.handler)
        assert entry.schema == manager.SKILL_MANAGE_SCHEMA

    @pytest.mark.asyncio
    async def test_atomic_write_preserves_symlink_and_cross_device_fallback(
        self,
        monkeypatch,
        tmp_path,
    ):
        target = tmp_path / "target.md"
        target.write_text("old", encoding="utf-8")
        link = tmp_path / "link.md"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        original_replace = manager.aiofiles.os.replace

        async def cross_device_for_target(source, destination):
            if Path(destination) == target:
                raise OSError(errno.EXDEV, "cross-device link")
            await original_replace(source, destination)

        monkeypatch.setattr(
            manager.aiofiles.os,
            "replace",
            cross_device_for_target,
        )
        await manager._atomic_write_text(link, "new content")

        assert link.is_symlink()
        assert target.read_text(encoding="utf-8") == "new content"

    @pytest.mark.asyncio
    async def test_repeated_cancellation_waits_for_single_delete_task(
        self,
        monkeypatch,
        tmp_path,
    ):
        delete_started = asyncio.Event()
        release_delete = asyncio.Event()
        delete_completed = asyncio.Event()
        calls = 0

        async def controlled_remove_tree(_path):
            nonlocal calls
            calls += 1
            delete_started.set()
            await release_delete.wait()
            delete_completed.set()

        monkeypatch.setattr(manager, "_remove_tree", controlled_remove_tree)
        task = asyncio.create_task(manager._remove_tree_fully(tmp_path / "skill"))
        await delete_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        try:
            assert task.done() is False
            assert calls == 1
        finally:
            release_delete.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(delete_completed.wait(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancelled_agent_created_scan_rolls_back_before_reraising(
        self,
        monkeypatch,
        tmp_path,
    ):
        scan_started = asyncio.Event()
        rollback_started = asyncio.Event()
        release_rollback = asyncio.Event()
        original_remove_tree = manager._remove_tree

        async def blocked_scan(_skill_dir):
            scan_started.set()
            await asyncio.Event().wait()

        async def controlled_remove_tree(path):
            rollback_started.set()
            await release_rollback.wait()
            await original_remove_tree(path)

        monkeypatch.setattr(manager, "_security_scan_skill", blocked_scan)
        monkeypatch.setattr(manager, "_remove_tree", controlled_remove_tree)
        async with isolated_skills(monkeypatch, tmp_path):
            task = asyncio.create_task(manager._create_skill("test-skill", VALID_SKILL))
            await scan_started.wait()
            assert (tmp_path / "test-skill" / "SKILL.md").exists()
            task.cancel()
            await rollback_started.wait()
            task.cancel()
            await asyncio.sleep(0)

            try:
                assert task.done() is False
            finally:
                release_rollback.set()
                with pytest.raises(asyncio.CancelledError):
                    await task

        assert not (tmp_path / "test-skill").exists()

    @pytest.mark.asyncio
    async def test_active_path_never_calls_to_thread(
        self,
        monkeypatch,
        tmp_path,
    ):
        async def reject_to_thread(*_args, **_kwargs):
            raise AssertionError("skill_manage must not use asyncio.to_thread")

        monkeypatch.setattr(asyncio, "to_thread", reject_to_thread)
        async with isolated_skills(monkeypatch, tmp_path):
            result = await result_of("create", "test-skill", VALID_SKILL)
        assert result["success"] is True
