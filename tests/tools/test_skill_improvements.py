"""Async parity tests for fuzzy skill patching."""

import json

import pytest

from tools.skill_manager_tool import _create_skill, _patch_skill, skill_manage


SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

Step 1: Do the thing.
Step 2: Do another thing.
Step 3: Final step.
"""


class TestFuzzyPatchSkill:
    @pytest.fixture(autouse=True)
    def setup_skills(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self.skills_dir = skills_dir

    @pytest.mark.asyncio
    async def test_exact_match_still_works(self):
        await _create_skill("test-skill", SKILL_CONTENT)
        result = await _patch_skill(
            "test-skill", "Step 1: Do the thing.", "Step 1: Done!"
        )
        assert result["success"] is True
        content = (self.skills_dir / "test-skill" / "SKILL.md").read_text()
        assert "Step 1: Done!" in content

    @pytest.mark.asyncio
    async def test_whitespace_trimmed_match(self):
        skill = """\
---
name: ws-skill
description: Whitespace test
---

# Commands

    def hello():
        print("hi")
"""
        await _create_skill("ws-skill", skill)
        result = await _patch_skill(
            "ws-skill",
            'def hello():\n    print("hi")',
            'def hello():\n    print("hello world")',
        )
        assert result["success"] is True
        content = (self.skills_dir / "ws-skill" / "SKILL.md").read_text()
        assert 'print("hello world")' in content

    @pytest.mark.asyncio
    async def test_multiple_matches_blocked_without_replace_all(self):
        skill = """\
---
name: dup-skill
description: Duplicate test
---

# Steps

word word word
"""
        await _create_skill("dup-skill", skill)
        result = await _patch_skill("dup-skill", "word", "replaced")
        assert result["success"] is False
        assert "match" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_skill_manage_patch_uses_fuzzy(self):
        await _create_skill("test-skill", SKILL_CONTENT)
        result = json.loads(
            await skill_manage(
                action="patch",
                name="test-skill",
                old_string="  Step 1: Do the thing.",
                new_string="Step 1: Updated.",
            )
        )
        assert result["success"] is True
