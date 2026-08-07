"""Tests for agent/skill_bundles.py — YAML-defined skill bundles."""

import asyncio
import os
from pathlib import Path

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.skill_bundles import (
    _slugify,
    build_bundle_invocation_message,
    delete_bundle,
    get_bundle,
    get_skill_bundles,
    list_bundles,
    reload_bundles,
    resolve_bundle_command_key,
    save_bundle,
    scan_bundles,
)

pytestmark = pytest.mark.asyncio


def _make_bundle_yaml(
    bundles_dir: Path, slug: str, skills: list[str],
    description: str = "", instruction: str = "", name: str | None = None,
) -> Path:
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    if name is not None:
        lines.append(f"name: {name}")
    else:
        lines.append(f"name: {slug}")
    if description:
        lines.append(f"description: {description}")
    lines.append("skills:")
    for s in skills:
        lines.append(f"  - {s}")
    if instruction:
        lines.append("instruction: |")
        for ln in instruction.splitlines():
            lines.append(f"  {ln}")
    path = bundles_dir / f"{slug}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_skill(skills_dir: Path, name: str, body: str = "Do the thing.") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


@pytest.fixture
def bundles_env(tmp_path, monkeypatch):
    """Isolated bundles dir + skills dir."""
    bundles_dir = tmp_path / "skill-bundles"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    # Patch SKILLS_DIR so skill loading hits our temp tree.
    import tools.skills_tool as skills_tool_module
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    # Reset module-level cache between tests.
    import agent.skill_bundles as mod
    mod._bundles_cache = {}
    mod._bundles_cache_mtime = None
    return bundles_dir, skills_dir


class TestSlugify:
    async def test_basic(self):
        assert _slugify("Backend Dev") == "backend-dev"




    async def test_empty(self):
        assert _slugify("") == ""
        assert _slugify("!!!") == ""


class TestScanBundles:

    async def test_finds_bundle(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "backend", ["skill-a", "skill-b"])
        result = await scan_bundles()
        assert "/backend" in result
        assert result["/backend"]["name"] == "backend"
        assert result["/backend"]["skills"] == ["skill-a", "skill-b"]

    async def test_skips_invalid_yaml(self, bundles_env):
        bundles_dir, _ = bundles_env
        bundles_dir.mkdir(parents=True)
        (bundles_dir / "broken.yaml").write_text("{not: valid yaml: [")
        _make_bundle_yaml(bundles_dir, "good", ["skill-a"])
        result = await scan_bundles()
        assert "/good" in result
        assert "/broken" not in result





class TestGetSkillBundles:
    async def test_returns_cache(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "a", ["s1"])
        first = await get_skill_bundles()
        # Second call should hit cache (no rescan unless mtime changed).
        second = await get_skill_bundles()
        assert first is second or first == second

    async def test_rescans_on_change(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "a", ["s1"])
        assert "/a" in await get_skill_bundles()
        # Add a second bundle and bump mtime.
        await asyncio.sleep(0.05)  # ensure mtime granularity is exceeded
        _make_bundle_yaml(bundles_dir, "b", ["s2"])
        import aiofiles.os

        await aiofiles.os.wrap(os.utime)(bundles_dir, None)
        result = await get_skill_bundles()
        assert "/a" in result
        assert "/b" in result


class TestResolveBundleCommandKey:
    async def test_exact_match(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "my-bundle", ["s1"])
        await scan_bundles()
        assert await resolve_bundle_command_key("my-bundle") == "/my-bundle"


    async def test_unknown(self, bundles_env):
        await scan_bundles()
        assert await resolve_bundle_command_key("missing") is None

    async def test_empty(self, bundles_env):
        assert await resolve_bundle_command_key("") is None


class TestBuildBundleInvocationMessage:
    async def test_loads_all_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a", body="Skill A content.")
        _make_skill(skills_dir, "skill-b", body="Skill B content.")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-b"])
        await scan_bundles()

        result = await build_bundle_invocation_message("/combo")
        assert result is not None
        msg, loaded, missing = result
        assert set(loaded) == {"skill-a", "skill-b"}
        assert missing == []
        assert "Skill A content." in msg
        assert "Skill B content." in msg
        assert "combo" in msg

    async def test_skips_missing_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-ghost"])
        await scan_bundles()

        result = await build_bundle_invocation_message("/combo")
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a"]
        assert missing == ["skill-ghost"]
        assert "skill-ghost" in msg  # called out in header

    async def test_skips_platform_disabled_skills(self, bundles_env, monkeypatch):
        """A skill disabled for the invoking platform must not be injected
        via a bundle (mirrors the stacked-skill gate, #58888)."""
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a", body="Skill A content.")
        _make_skill(skills_dir, "skill-b", body="SECRET DISABLED CONTENT.")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-b"])
        await scan_bundles()

        async def _fake_disabled(platform=None):
            return {"skill-b"} if platform == "telegram" else set()

        import tools.skills_tool as skills_tool_module
        monkeypatch.setattr(
            skills_tool_module, "_get_disabled_skill_names", _fake_disabled
        )

        result = await build_bundle_invocation_message("/combo", platform="telegram")
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a"]
        assert "SECRET DISABLED CONTENT." not in msg
        assert "skill-b" in msg  # called out in the disabled-skipped header line
        assert "disabled" in msg.lower()

        # Positive control: without the platform the skill loads normally.
        result2 = await build_bundle_invocation_message("/combo")
        assert result2 is not None
        msg2, loaded2, _ = result2
        assert set(loaded2) == {"skill-a", "skill-b"}
        assert "SECRET DISABLED CONTENT." in msg2








class TestSaveAndDeleteBundle:
    async def test_save_creates_file(self, bundles_env):
        bundles_dir, _ = bundles_env
        path = await save_bundle("test-bundle", ["s1", "s2"], description="d", instruction="i")
        assert path.exists()
        assert path.parent == bundles_dir
        content = path.read_text()
        assert "test-bundle" in content
        assert "s1" in content
        assert "s2" in content
        assert "description: d" in content


    async def test_save_overwrites_with_force(self, bundles_env):
        await save_bundle("dup", ["s1"])
        await save_bundle("dup", ["s2"], overwrite=True)
        info = await get_bundle("dup")
        assert info is not None
        assert info["skills"] == ["s2"]



    async def test_delete_removes_file(self, bundles_env):
        bundles_dir, _ = bundles_env
        await save_bundle("doomed", ["s1"])
        assert await get_bundle("doomed") is not None
        await delete_bundle("doomed")
        assert await get_bundle("doomed") is None



class TestReloadBundles:
    async def test_reports_added_and_removed(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "old", ["s1"])
        await scan_bundles()  # populate cache with {old}

        # Mutate the disk WITHOUT going through save/delete helpers (which
        # would refresh the cache mid-way). reload_bundles() diffs the
        # in-memory cache against the freshly-scanned disk state.
        (bundles_dir / "old.yaml").unlink()
        _make_bundle_yaml(bundles_dir, "new", ["s2"])

        diff = await reload_bundles()
        added_names = {e["name"] for e in diff["added"]}
        removed_names = {e["name"] for e in diff["removed"]}
        assert "new" in added_names
        assert "old" in removed_names
        assert diff["total"] == 1


class TestListBundles:
    async def test_sorted_by_slug(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "zebra", ["s1"])
        _make_bundle_yaml(bundles_dir, "apple", ["s2"])
        _make_bundle_yaml(bundles_dir, "mango", ["s3"])
        await scan_bundles()
        info_list = await list_bundles()
        slugs = [b["slug"] for b in info_list]
        assert slugs == sorted(slugs)


async def test_bundle_lifecycle_does_not_block_or_leak(bundles_env):
    bundles_dir, skills_dir = bundles_env
    _make_skill(skills_dir, "skill-a", body="Skill A content.")
    _make_bundle_yaml(bundles_dir, "combo", ["skill-a"])

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            await scan_bundles()
            invocation = await build_bundle_invocation_message("/combo")
            saved = await save_bundle("temporary", ["skill-a"])
            deleted = await delete_bundle("temporary")
        finally:
            blockbuster.deactivate()

    assert invocation is not None
    assert saved == deleted
