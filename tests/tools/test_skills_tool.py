"""Tests for tools/skills_tool.py — skill discovery and viewing."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skills_tool import (
    _get_required_environment_variables,
    _parse_frontmatter,
    _parse_tags,
    _get_category_from_path,
    _find_all_skills,
    skill_matches_platform,
    skills_list,
    skill_view,
    MAX_DESCRIPTION_LENGTH,
)


def _make_skill(
    skills_dir, name, frontmatter_extra="", body="Step 1: Do the thing.", category=None
):
    """Helper to create a minimal skill directory."""
    if category:
        skill_dir = skills_dir / category / name
    else:
        skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"""\
---
name: {name}
description: Description for {name}.
{frontmatter_extra}---

# {name}

{body}
"""
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


def _symlink_category(skills_dir: Path, linked_root: Path, category: str) -> Path:
    """Create a category symlink under skills_dir pointing outside the tree."""
    external_category = linked_root / category
    external_category.mkdir(parents=True, exist_ok=True)
    symlink_path = skills_dir / category
    try:
        symlink_path.symlink_to(external_category, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")
    return external_category


# ---------------------------------------------------------------------------
# _parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_and_nested_frontmatter(self):
        content = "---\nname: test\ndescription: A test.\n---\n\n# Body\n"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "A test."
        assert "# Body" in body

        nested = "---\nname: test\nmetadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        fm, _ = _parse_frontmatter(nested)
        assert fm["metadata"]["hermes"]["tags"] == ["a", "b"]


    def test_utf8_bom_frontmatter(self):
        """A leading UTF-8 BOM (Windows Notepad / PowerShell ``>`` save) must
        not drop the frontmatter. Confirms the fix reaches the tools/ surface
        via the _parse_frontmatter re-export."""
        bom = chr(0xFEFF)
        content = bom + "---\nname: test\ndescription: A test.\n---\n\n# Body\n"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "A test."
        assert not body.startswith(bom)


# ---------------------------------------------------------------------------
# _parse_tags
# ---------------------------------------------------------------------------


class TestParseTags:
    def test_accepted_input_forms(self):
        assert _parse_tags(["a", "b", "c"]) == ["a", "b", "c"]
        assert _parse_tags("a, b, c") == ["a", "b", "c"]
        assert _parse_tags("[a, b, c]") == ["a", "b", "c"]
        # Quotes are stripped.
        result = _parse_tags("\"tag1\", 'tag2'")
        assert "tag1" in result
        assert "tag2" in result

    def test_empty_and_blank_items_dropped(self):
        assert _parse_tags("") == []
        assert _parse_tags(None) == []
        assert _parse_tags([]) == []
        assert _parse_tags([None, "", "valid"]) == ["valid"]


class TestRequiredEnvironmentVariablesNormalization:
    def test_parses_new_metadata_and_normalizes_legacy_prerequisites(self):
        frontmatter = {
            "required_environment_variables": [
                {
                    "name": "TENOR_API_KEY",
                    "prompt": "Tenor API key",
                    "help": "Get a key from https://developers.google.com/tenor",
                    "required_for": "full functionality",
                }
            ]
        }
        assert _get_required_environment_variables(frontmatter) == [
            {
                "name": "TENOR_API_KEY",
                "prompt": "Tenor API key",
                "help": "Get a key from https://developers.google.com/tenor",
                "required_for": "full functionality",
            }
        ]

        legacy = {"prerequisites": {"env_vars": ["TENOR_API_KEY"]}}
        assert _get_required_environment_variables(legacy) == [
            {
                "name": "TENOR_API_KEY",
                "prompt": "Enter value for TENOR_API_KEY",
            }
        ]

# ---------------------------------------------------------------------------
# _get_category_from_path
# ---------------------------------------------------------------------------


class TestGetCategoryFromPath:
    def test_category_derived_from_layout(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            categorized = tmp_path / "mlops" / "axolotl" / "SKILL.md"
            categorized.parent.mkdir(parents=True)
            categorized.touch()
            assert _get_category_from_path(categorized) == "mlops"

            top_level = tmp_path / "my-skill" / "SKILL.md"
            top_level.parent.mkdir(parents=True)
            top_level.touch()
            assert _get_category_from_path(top_level) is None

        # Paths outside SKILLS_DIR have no category.
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
            assert _get_category_from_path(tmp_path / "other" / "SKILL.md") is None


# ---------------------------------------------------------------------------
# _find_all_skills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFindAllSkills:
    async def test_finds_skills_and_skips_non_skill_trees(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skill-a")
            _make_skill(tmp_path, "skill-b")
            _make_skill(tmp_path, "axolotl", category="mlops")

            # .git internals are not skills.
            git_dir = tmp_path / ".git" / "fake-skill"
            git_dir.mkdir(parents=True)
            (git_dir / "SKILL.md").write_text(
                "---\nname: fake\ndescription: x\n---\n\nBody.\n"
            )
            # Neither are skills vendored inside a nested virtualenv.
            typer_skill = (
                tmp_path
                / "bring"
                / "scripts"
                / ".venv"
                / "lib"
                / "python3.13"
                / "site-packages"
                / "typer"
                / ".agents"
                / "skills"
                / "typer"
            )
            typer_skill.mkdir(parents=True)
            (typer_skill / "SKILL.md").write_text(
                "---\nname: typer\ndescription: Should not be discovered.\n---\n",
                encoding="utf-8",
            )

            skills = await _find_all_skills()

        assert {s["name"] for s in skills} == {"skill-a", "skill-b", "axolotl"}
        assert [s["category"] for s in skills if s["name"] == "axolotl"] == ["mlops"]


    async def test_description_falls_back_to_body_and_is_truncated(self, tmp_path):
        no_desc = tmp_path / "no-desc"
        no_desc.mkdir()
        (no_desc / "SKILL.md").write_text(
            "---\nname: no-desc\n---\n\n# Heading\n\nFirst paragraph.\n"
        )
        long_dir = tmp_path / "long-desc"
        long_dir.mkdir()
        (long_dir / "SKILL.md").write_text(
            f"---\nname: long\ndescription: {'x' * (MAX_DESCRIPTION_LENGTH + 100)}\n---\n\nBody.\n"
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skills = {s["name"]: s for s in await _find_all_skills()}

        # If no description in frontmatter, the first non-header line is used.
        assert skills["no-desc"]["description"] == "First paragraph."
        assert len(skills["long"]["description"]) <= MAX_DESCRIPTION_LENGTH

    async def test_finds_skills_in_symlinked_category_dir(self, tmp_path):
        external_root = tmp_path / "repo"
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        external_category = _symlink_category(skills_root, external_root, "linked")
        _make_skill(external_category.parent, "knowledge-brain", category="linked")

        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            skills = await _find_all_skills()

        assert [s["name"] for s in skills] == ["knowledge-brain"]
        assert skills[0]["category"] == "linked"


# ---------------------------------------------------------------------------
# skills_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSkillsList:
    async def test_empty_creates_directory(self, tmp_path):
        skills_dir = tmp_path / "skills"
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            raw = await skills_list()
        result = json.loads(raw)
        assert result["success"] is True
        assert result["skills"] == []
        assert skills_dir.exists()

    async def test_category_filter(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skill-a", category="devops")
            _make_skill(tmp_path, "skill-b", category="mlops")
            all_result = json.loads(await skills_list())
            filtered = json.loads(await skills_list(category="devops"))

        assert all_result["count"] == 2
        assert filtered["count"] == 1
        assert filtered["skills"][0]["name"] == "skill-a"

    async def test_category_filter_finds_symlinked_category(self, tmp_path):
        external_root = tmp_path / "repo"
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        external_category = _symlink_category(skills_root, external_root, "linked")
        _make_skill(external_category.parent, "knowledge-brain", category="linked")

        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            raw = await skills_list(category="linked")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["categories"] == ["linked"]
        assert result["skills"][0]["name"] == "knowledge-brain"


# ---------------------------------------------------------------------------
# skill_view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSkillView:
    async def test_view_resolves_by_dir_name_and_frontmatter_name(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "my-skill",
                frontmatter_extra="metadata:\n  hermes:\n    tags: [fine-tuning, llm]\n",
            )
            # The on-disk directory ("alias-dir") differs from the skill's
            # frontmatter name ("real-skill-name"). skills_list() exposes the
            # frontmatter name, so skill_view(name) must resolve it too.
            alias_dir = tmp_path / "alias-dir"
            alias_dir.mkdir(parents=True, exist_ok=True)
            (alias_dir / "SKILL.md").write_text(
                "---\n"
                "name: real-skill-name\n"
                "description: A skill whose directory name differs from its name.\n"
                "---\n\n"
                "# real-skill-name\n\n"
                "Step 1: Do the thing.\n"
            )
            by_dir = json.loads(await skill_view("my-skill"))
            by_name = json.loads(await skill_view("real-skill-name"))

        assert by_dir["success"] is True
        assert by_dir["name"] == "my-skill"
        assert "Step 1" in by_dir["content"]
        assert "fine-tuning" in by_dir["tags"]
        assert "llm" in by_dir["tags"]

        assert by_name["success"] is True
        assert "Step 1" in by_name["content"]


    async def test_view_reference_files(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "my-skill")
            refs_dir = skill_dir / "references"
            refs_dir.mkdir()
            (refs_dir / "api.md").write_text("# API Docs\nEndpoint info.")

            existing = json.loads(await skill_view("my-skill", file_path="references/api.md"))
            missing = json.loads(await skill_view("my-skill", file_path="references/nope.md"))
            skill = json.loads(await skill_view("my-skill"))

        assert existing["success"] is True
        assert "Endpoint info" in existing["content"]
        assert missing["success"] is False
        # The skill view advertises what else can be opened.
        assert skill["linked_files"] is not None
        assert "references" in skill["linked_files"]

    async def test_disabled_skill_blocked_enabled_allowed(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skills_tool._is_skill_disabled", return_value=True),
        ):
            _make_skill(tmp_path, "hidden-skill")
            blocked = json.loads(await skill_view("hidden-skill"))
        assert blocked["success"] is False
        assert "disabled" in blocked["error"].lower()

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skills_tool._is_skill_disabled", return_value=False),
        ):
            _make_skill(tmp_path, "active-skill")
            allowed = json.loads(await skill_view("active-skill"))
        assert allowed["success"] is True

    async def test_view_finds_skill_in_symlinked_category_dir(self, tmp_path):
        external_root = tmp_path / "repo"
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        external_category = _symlink_category(skills_root, external_root, "linked")
        _make_skill(external_category.parent, "knowledge-brain", category="linked")

        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            raw = await skill_view("knowledge-brain")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["name"] == "knowledge-brain"


@pytest.mark.asyncio
class TestSkillViewSecureSetupOnLoad:
    async def test_reports_missing_required_environment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                    "    help: Get a key from https://developers.google.com/tenor\n"
                    "    required_for: full functionality\n"
                ),
            )
            raw = await skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["name"] == "gif-search"
        assert result["required_environment_variables"][0]["name"] == "TENOR_API_KEY"
        assert result["missing_required_environment_variables"] == ["TENOR_API_KEY"]
        assert result["setup_needed"] is True
        assert result["readiness_status"] == "setup_needed"

    async def test_missing_environment_does_not_block_skill_loading(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            raw = await skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["content"].startswith("---")

# ---------------------------------------------------------------------------
# skill_matches_platform
# ---------------------------------------------------------------------------


class TestSkillMatchesPlatform:
    """Tests for the platforms frontmatter field filtering."""

    def test_missing_or_empty_platforms_matches_everything(self):
        assert skill_matches_platform({}) is True
        assert skill_matches_platform({"name": "foo"}) is True
        assert skill_matches_platform({"platforms": []}) is True
        assert skill_matches_platform({"platforms": None}) is True


    def test_string_form_case_insensitive_and_unknown_platforms(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            # A single string value is treated as a one-element list.
            assert skill_matches_platform({"platforms": "macos"}) is True
            assert skill_matches_platform({"platforms": ["MacOS"]}) is True
            assert skill_matches_platform({"platforms": ["MACOS"]}) is True

            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": "macos"}) is False
            assert skill_matches_platform({"platforms": ["freebsd"]}) is False


# ---------------------------------------------------------------------------
# _find_all_skills — platform filtering integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFindAllSkillsPlatformFiltering:
    """Test that _find_all_skills respects the platforms field."""

    async def test_discovery_filters_on_platform(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            _make_skill(tmp_path, "universal-skill")
            _make_skill(tmp_path, "mac-only", frontmatter_extra="platforms: [macos]\n")

            mock_sys.platform = "linux"
            linux = {s["name"] for s in await _find_all_skills()}
            mock_sys.platform = "darwin"
            darwin = {s["name"] for s in await _find_all_skills()}
            mock_sys.platform = "win32"
            win = {s["name"] for s in await _find_all_skills()}

        assert linux == {"universal-skill"}
        assert darwin == {"universal-skill", "mac-only"}
        # Skills without a platforms field appear on every platform.
        assert win == {"universal-skill"}

    async def test_multi_platform_skill(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            _make_skill(
                tmp_path, "cross-plat", frontmatter_extra="platforms: [macos, linux]\n"
            )
            mock_sys.platform = "darwin"
            skills_darwin = await _find_all_skills()
            mock_sys.platform = "linux"
            skills_linux = await _find_all_skills()
            mock_sys.platform = "win32"
            skills_win = await _find_all_skills()
        assert len(skills_darwin) == 1
        assert len(skills_linux) == 1
        assert len(skills_win) == 0


# ---------------------------------------------------------------------------
# _find_all_skills — env-var prerequisites must not change the listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFindAllSkillsSecureSetup:
    async def test_listing_shape_independent_of_env_var_prereqs(self, tmp_path, monkeypatch):
        # A remote backend must not be probed just to build the listing.
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.delenv("NONEXISTENT_API_KEY_XYZ", raising=False)
        monkeypatch.setenv("MY_PRESENT_KEY", "val")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "needs-key",
                frontmatter_extra="prerequisites:\n  env_vars: [NONEXISTENT_API_KEY_XYZ]\n",
            )
            _make_skill(
                tmp_path,
                "has-key",
                frontmatter_extra="prerequisites:\n  env_vars: [MY_PRESENT_KEY]\n",
            )
            _make_skill(tmp_path, "simple-skill")
            skills = await _find_all_skills()

        assert {s["name"] for s in skills} == {"needs-key", "has-key", "simple-skill"}
        for skill in skills:
            assert "readiness_status" not in skill
            assert "missing_prerequisites" not in skill


@pytest.mark.asyncio
class TestSkillViewPrerequisites:
    async def test_legacy_prerequisites_expose_required_env_setup_metadata(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gated-skill",
                frontmatter_extra="prerequisites:\n  env_vars: [MISSING_KEY_XYZ]\n",
            )
            raw = await skill_view("gated-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == ["MISSING_KEY_XYZ"]
        assert result["required_environment_variables"] == [
            {
                "name": "MISSING_KEY_XYZ",
                "prompt": "Enter value for MISSING_KEY_XYZ",
            }
        ]


    async def test_remote_backend_treats_persisted_env_as_available(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "remote-ready",
                frontmatter_extra="prerequisites:\n  env_vars: [PERSISTED_REMOTE_KEY]\n",
            )
            from hermes_cli.config import save_env_value

            save_env_value("PERSISTED_REMOTE_KEY", "persisted-value")
            monkeypatch.delenv("PERSISTED_REMOTE_KEY", raising=False)
            raw = await skill_view("remote-ready")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert result["readiness_status"] == "available"

    async def test_no_setup_metadata_when_no_required_envs(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "plain-skill")
            raw = await skill_view("plain-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["required_environment_variables"] == []

    async def test_skill_view_treats_backend_only_env_as_setup_needed(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "backend-ready",
                frontmatter_extra="prerequisites:\n  env_vars: [BACKEND_ONLY_KEY]\n",
            )
            raw = await skill_view("backend-ready")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == ["BACKEND_ONLY_KEY"]

    async def test_local_env_missing_keeps_setup_needed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.delenv("SHELL_ONLY_KEY", raising=False)

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "shell-ready",
                frontmatter_extra="prerequisites:\n  env_vars: [SHELL_ONLY_KEY]\n",
            )
            raw = await skill_view("shell-ready")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == ["SHELL_ONLY_KEY"]
        assert result["readiness_status"] == "setup_needed"

    @pytest.mark.parametrize(
        "backend",
        ["ssh", "daytona", "docker", "singularity", "modal", "vercel_sandbox"],
    )
    async def test_remote_backend_reports_missing_environment(
        self, tmp_path, monkeypatch, backend
    ):
        monkeypatch.setenv("TERMINAL_ENV", backend)
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            raw = await skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["readiness_status"] == "setup_needed"
        assert result["missing_required_environment_variables"] == ["TENOR_API_KEY"]


    async def test_legacy_flat_md_skill_preserves_frontmatter_metadata(self, tmp_path):
        flat_skill = tmp_path / "legacy-skill.md"
        flat_skill.write_text(
            """\
---
name: legacy-flat
description: Legacy flat skill.
metadata:
  hermes:
    tags: [legacy, flat]
required_environment_variables:
  - name: LEGACY_KEY
    prompt: Legacy key
---

# Legacy Flat

Do the legacy thing.
""",
            encoding="utf-8",
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            raw = await skill_view("legacy-skill")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["name"] == "legacy-flat"
        assert result["description"] == "Legacy flat skill."
        assert result["tags"] == ["legacy", "flat"]
        assert result["required_environment_variables"] == [
            {"name": "LEGACY_KEY", "prompt": "Legacy key"}
        ]

    async def test_persisted_environment_makes_skill_available(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            (tmp_path / ".env").write_text("TENOR_API_KEY=captured-value\n")
            raw = await skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert result["readiness_status"] == "available"


@pytest.mark.asyncio
class TestSkillViewCollisionDetection:
    """Regression tests for skill_view name collision handling.

    When a skill name resolves to multiple paths across the local skills
    dir and external_dirs, skill_view must refuse to guess. Silent
    shadowing — where ``/skills`` shows the local version but
    ``skill_view`` loads the external one — is the bug class this guards
    against. Reproduces with `skills.external_dirs` registered in
    config.yaml and a same-name skill nested under a category locally.

    Adapted from a regression suite originally proposed by @polkn in PR
    #6136 (which used local-first precedence). The collision-refusal
    behavior preserves the same protection without silently picking a
    side, and gives the user an actionable hint (use the categorized
    path) to recover.
    """

    def _patch_dirs(self, local_dir, external_dirs):
        """Patch SKILLS_DIR (module-level) and get_external_skills_dirs at source."""
        return (
            patch("tools.skills_tool.SKILLS_DIR", local_dir),
            patch(
                "agent.skill_utils.get_external_skills_dirs",
                return_value=list(external_dirs),
            ),
        )

    async def test_nested_local_collides_with_top_level_external(self, tmp_path):
        """The original bug scenario: nested local + top-level external,
        same name. Now refuses with both paths surfaced."""
        local_dir = tmp_path / "local"
        external_dir = tmp_path / "external"
        local_dir.mkdir()
        external_dir.mkdir()

        _make_skill(
            local_dir,
            "explore-codebase",
            category="foundations/runtime",
            body="LOCAL VERSION",
        )
        _make_skill(external_dir, "explore-codebase", body="EXTERNAL VERSION")

        p1, p2 = self._patch_dirs(local_dir, [external_dir])
        with p1, p2:
            raw = await skill_view("explore-codebase")

        result = json.loads(raw)
        assert result["success"] is False
        assert "Ambiguous skill name 'explore-codebase'" in result["error"]
        assert "matches" in result
        assert len(result["matches"]) == 2
        # Both paths surfaced
        assert any("foundations/runtime" in p for p in result["matches"])
        assert any("external" in p for p in result["matches"])
        assert "hint" in result


    async def test_support_markdown_does_not_collide_with_real_skill(self, tmp_path):
        """Supporting reference docs named <skill>.md are not skills.

        A real-world regression had creative/sketch/SKILL.md become
        unloadable because another skill carried
        references/styles/sketch.md. Support files are loaded via
        skill_view(skill, file_path=...), not as bare skill names.
        """
        local_dir = tmp_path / "local"
        external_dir = tmp_path / "external"
        local_dir.mkdir()
        external_dir.mkdir()

        _make_skill(local_dir, "article-illustrator", category="creative")
        support_file = (
            local_dir
            / "creative"
            / "article-illustrator"
            / "references"
            / "styles"
            / "sketch.md"
        )
        support_file.parent.mkdir(parents=True, exist_ok=True)
        support_file.write_text("# Sketch style support doc\n")
        _make_skill(local_dir, "sketch", category="creative", body="REAL SKETCH SKILL")

        p1, p2 = self._patch_dirs(local_dir, [external_dir])
        with p1, p2:
            raw = await skill_view("sketch")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["path"] == "creative/sketch/SKILL.md"
        assert "REAL SKETCH SKILL" in result["content"]


    async def test_two_externals_same_name_also_refuse(self, tmp_path):
        """Collision detection is symmetric — two external dirs with
        same-name skills also trigger the refusal."""
        local_dir = tmp_path / "local"
        ext_a = tmp_path / "ext_a"
        ext_b = tmp_path / "ext_b"
        local_dir.mkdir()
        ext_a.mkdir()
        ext_b.mkdir()

        _make_skill(ext_a, "pr", body="EXT_A VERSION")
        _make_skill(ext_b, "pr", body="EXT_B VERSION")

        p1, p2 = self._patch_dirs(local_dir, [ext_a, ext_b])
        with p1, p2:
            raw = await skill_view("pr")

        result = json.loads(raw)
        assert result["success"] is False
        assert "Ambiguous" in result["error"]
        assert len(result["matches"]) == 2
