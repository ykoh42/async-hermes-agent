"""Async parity tests for upstream skill content size limits."""

import json

import pytest

from tools.skill_manager_tool import (
    MAX_SKILL_CONTENT_CHARS,
    _validate_content_size,
    skill_manage,
)


@pytest.fixture(autouse=True)
def isolate_skills(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return skills_dir


def _make_skill_content(body_chars: int) -> str:
    frontmatter = (
        "---\n"
        "name: test-skill\n"
        "description: A test skill\n"
        "---\n"
    )
    body = "# Test Skill\n\n" + ("x" * max(0, body_chars - 15))
    return frontmatter + body


class TestValidateContentSize:
    def test_within_limit(self):
        assert _validate_content_size("a" * 1000) is None

    def test_custom_label(self):
        error = _validate_content_size(
            "a" * (MAX_SKILL_CONTENT_CHARS + 1),
            label="references/api.md",
        )
        assert "references/api.md" in error


class TestCreateSkillSizeLimit:
    @pytest.mark.asyncio
    async def test_create_within_limit(self, isolate_skills):
        content = _make_skill_content(5000)
        result = json.loads(
            await skill_manage(action="create", name="small-skill", content=content)
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_at_limit(self, isolate_skills):
        frontmatter = "---\nname: edge-skill\ndescription: Edge case\n---\n# Edge\n\n"
        body_budget = MAX_SKILL_CONTENT_CHARS - len(frontmatter)
        content = frontmatter + ("x" * body_budget)
        assert len(content) == MAX_SKILL_CONTENT_CHARS
        result = json.loads(
            await skill_manage(action="create", name="edge-skill", content=content)
        )
        assert result["success"] is True


class TestEditSkillSizeLimit:
    @pytest.mark.asyncio
    async def test_edit_over_limit(self, isolate_skills):
        small = _make_skill_content(1000)
        await skill_manage(action="create", name="grow-me", content=small)
        big = _make_skill_content(MAX_SKILL_CONTENT_CHARS + 100).replace(
            "name: test-skill", "name: grow-me"
        )
        result = json.loads(
            await skill_manage(action="edit", name="grow-me", content=big)
        )
        assert result["success"] is False
        assert "100,000" in result["error"]


class TestPatchSkillSizeLimit:
    @pytest.mark.asyncio
    async def test_patch_that_would_exceed_limit(self, isolate_skills):
        near_limit = _make_skill_content(MAX_SKILL_CONTENT_CHARS - 50)
        await skill_manage(action="create", name="near-limit", content=near_limit)
        result = json.loads(
            await skill_manage(
                action="patch",
                name="near-limit",
                old_string="# Test Skill",
                new_string="# Test Skill\n" + ("y" * 200),
            )
        )
        assert result["success"] is False
        assert "100,000" in result["error"]

    @pytest.mark.asyncio
    async def test_patch_supporting_file_size_limit(self, isolate_skills):
        await skill_manage(
            action="create",
            name="with-ref",
            content=_make_skill_content(1000),
        )
        await skill_manage(
            action="write_file",
            name="with-ref",
            file_path="references/data.md",
            file_content="# Data\n\nSmall content.",
        )
        result = json.loads(
            await skill_manage(
                action="patch",
                name="with-ref",
                old_string="Small content.",
                new_string="x" * (MAX_SKILL_CONTENT_CHARS + 100),
                file_path="references/data.md",
            )
        )
        assert result["success"] is False
        assert "references/data.md" in result["error"]


class TestWriteFileSizeLimit:
    @pytest.mark.asyncio
    async def test_write_file_over_char_limit(self, isolate_skills):
        await skill_manage(
            action="create",
            name="file-test",
            content=_make_skill_content(1000),
        )
        result = json.loads(
            await skill_manage(
                action="write_file",
                name="file-test",
                file_path="references/huge.md",
                file_content="x" * (MAX_SKILL_CONTENT_CHARS + 1),
            )
        )
        assert result["success"] is False
        assert "100,000" in result["error"]

    @pytest.mark.asyncio
    async def test_write_file_within_limit(self, isolate_skills):
        await skill_manage(
            action="create",
            name="file-ok",
            content=_make_skill_content(1000),
        )
        result = json.loads(
            await skill_manage(
                action="write_file",
                name="file-ok",
                file_path="references/normal.md",
                file_content="# Normal\n\n" + ("x" * 5000),
            )
        )
        assert result["success"] is True


class TestHandPlacedSkillsNoLimit:
    @pytest.mark.asyncio
    async def test_oversized_handplaced_skill_loads(self, isolate_skills, tmp_path):
        from tools.skills_tool import skill_view

        skill_dir = tmp_path / "skills" / "manual-giant"
        skill_dir.mkdir(parents=True)
        huge = _make_skill_content(200_000).replace(
            "name: test-skill", "name: manual-giant"
        )
        (skill_dir / "SKILL.md").write_text(huge, encoding="utf-8")
        result = json.loads(await skill_view("manual-giant"))
        assert "content" in result
        assert len(result["content"]) > MAX_SKILL_CONTENT_CHARS
