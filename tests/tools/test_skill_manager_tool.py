"""Behavior tests for the native-async skill manager."""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import tools.skill_manager_tool as manager
from tools.registry import registry


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


class TestAsyncContract:
    def test_public_function_and_registry_handler_are_coroutines(self):
        entry = registry.get_entry("skill_manage")
        assert inspect.iscoroutinefunction(manager.skill_manage)
        assert entry is not None
        assert inspect.iscoroutinefunction(entry.handler)
        assert entry.schema == manager.SKILL_MANAGE_SCHEMA

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
