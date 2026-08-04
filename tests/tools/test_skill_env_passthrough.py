"""Test that skill_view registers required env vars in the passthrough registry."""

import json

import pytest

import tools.env_passthrough as _ep_mod
from tools.env_passthrough import clear_env_passthrough, is_env_passthrough


@pytest.fixture(autouse=True)
def _clean_passthrough():
    clear_env_passthrough()
    _ep_mod._config_passthrough = None
    yield
    clear_env_passthrough()
    _ep_mod._config_passthrough = None


def _create_skill(tmp_path, name, frontmatter_extra=""):
    """Create a minimal skill directory with SKILL.md."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: Test skill\n"
        f"{frontmatter_extra}"
        f"---\n\n"
        f"# {name}\n\n"
        f"Test content.\n"
    )
    return skill_dir


class TestSkillViewRegistersPassthrough:
    @pytest.mark.asyncio
    async def test_available_env_vars_registered(self, tmp_path, monkeypatch):
        """When a skill declares required_environment_variables and the var IS set,
        it should be registered in the passthrough."""
        _create_skill(
            tmp_path,
            "test-skill",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: TENOR_API_KEY\n"
                "    prompt: Enter your Tenor API key\n"
            ),
        )
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", tmp_path
        )
        # Set the env var so it's "available"
        monkeypatch.setenv("TENOR_API_KEY", "test-value-123")

        from tools.skills_tool import skill_view

        result = json.loads(await skill_view(name="test-skill"))

        assert result["success"] is True
        assert is_env_passthrough("TENOR_API_KEY")


    @pytest.mark.asyncio
    async def test_no_env_vars_skill_no_registration(self, tmp_path, monkeypatch):
        """Skills without required_environment_variables shouldn't register anything."""
        _create_skill(tmp_path, "simple-skill")
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", tmp_path
        )

        from tools.skills_tool import skill_view

        result = json.loads(await skill_view(name="simple-skill"))

        assert result["success"] is True
        from tools.env_passthrough import get_all_passthrough
        assert len(get_all_passthrough()) == 0
