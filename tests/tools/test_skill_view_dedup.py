"""Tests for skill_view repeat-view dedup (unchanged-skill stub)."""

import json
import os
import time
from pathlib import Path

import pytest

from tools.skills_tool import (
    _handle_skill_view,
    reset_skill_view_dedup,
)


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    d = skills / "demo-dedup-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: demo-dedup-skill\ndescription: Demo skill for dedup tests.\n---\n"
        "# Demo\n\nStep one: run the demo procedure fully.\n"
    )
    refs = d / "references"
    refs.mkdir()
    (refs / "guide.md").write_text("# Guide\n\nDetailed reference content here.\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    return home


async def _view(name, file_path=None, task="t-svd"):
    args = {"name": name}
    if file_path:
        args["file_path"] = file_path
    return json.loads(await _handle_skill_view(args, task_id=task))


@pytest.mark.asyncio
class TestSkillViewDedup:
    async def test_first_view_returns_full_content(self, skills_home):
        r = await _view("demo-dedup-skill")
        assert r["success"] is True
        assert "Step one" in r.get("content", "")

    async def test_repeat_view_returns_stub(self, skills_home):
        await _view("demo-dedup-skill")
        r2 = await _view("demo-dedup-skill")
        assert r2["success"] is True
        assert r2.get("dedup") is True
        assert r2.get("content_returned") is False
        assert "unchanged" in r2["message"]
        assert "content" not in r2

    async def test_modified_skill_returns_full_content(self, skills_home):
        await _view("demo-dedup-skill")
        md = skills_home / "skills" / "demo-dedup-skill" / "SKILL.md"
        time.sleep(0.01)
        md.write_text(md.read_text() + "\nStep two: new instruction.\n")
        r2 = await _view("demo-dedup-skill")
        assert "Step two" in r2.get("content", "")
        assert r2.get("dedup") is None

    async def test_linked_file_dedup_is_independent(self, skills_home):
        await _view("demo-dedup-skill")
        # First view of a DIFFERENT file within the skill: full content.
        r = await _view("demo-dedup-skill", file_path="references/guide.md")
        assert "Detailed reference" in r.get("content", "")
        # Repeat of that file: stub.
        r2 = await _view("demo-dedup-skill", file_path="references/guide.md")
        assert r2.get("dedup") is True

    async def test_different_tasks_do_not_share_cache(self, skills_home):
        await _view("demo-dedup-skill", task="task-A")
        r = await _view("demo-dedup-skill", task="task-B")
        assert "Step one" in r.get("content", "")

    async def test_reset_returns_full_content(self, skills_home):
        await _view("demo-dedup-skill")
        reset_skill_view_dedup("t-svd")
        r2 = await _view("demo-dedup-skill")
        assert "Step one" in r2.get("content", "")

    async def test_no_task_id_never_dedups(self, skills_home):
        args = {"name": "demo-dedup-skill"}
        await _handle_skill_view(args, task_id=None)
        r2 = json.loads(await _handle_skill_view(args, task_id=None))
        assert "Step one" in r2.get("content", "")

    async def test_compression_hook_importable(self):
        # conversation_compression imports this lazily; keep the seam stable.
        from tools.skills_tool import reset_skill_view_dedup as f
        f(None)
