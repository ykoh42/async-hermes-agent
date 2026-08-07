"""Tests for the sandbox-mirror write guard in agent/file_safety."""

from __future__ import annotations

from pathlib import Path

import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction


pytestmark = pytest.mark.asyncio


class TestClassifySandboxMirrorTarget:
    async def test_docker_mirror_soul_md_classified(self, tmp_path):
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "profiles"
            / "group1"
            / "sandboxes"
            / "docker"
            / "default"
            / "home"
            / ".hermes"
            / "profiles"
            / "group1"
            / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("# mirror copy\n")

        async with no_task_leaks(action=LeakAction.RAISE):
            blockbuster = BlockBuster()
            blockbuster.activate()
            try:
                result = await classify_sandbox_mirror_target(str(target))
            finally:
                blockbuster.deactivate()

        assert result is not None
        assert result["target_path"] == str(target.resolve())
        assert result["mirror_root"].endswith(
            "sandboxes/docker/default/home/.hermes"
        )
        assert result["inner_path"] == "profiles/group1/SOUL.md"

    @pytest.mark.parametrize(
        "backend,inner",
        [
            ("docker", "profiles/coder/memories/MEMORY.md"),
            ("daytona", "profiles/default/cron/jobs.json"),
            ("podman", ".env"),
        ],
    )
    async def test_other_backends_and_inner_files_match(
        self, tmp_path, backend, inner
    ):
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "sandboxes"
            / backend
            / "task-42"
            / "home"
            / ".hermes"
            / Path(inner)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")

        result = await classify_sandbox_mirror_target(str(target))
        assert result is not None
        assert result["inner_path"] == inner
        assert backend in result["mirror_root"]


class TestGetSandboxMirrorWarning:
    async def test_non_mirror_returns_none(self, tmp_path):
        from agent.file_safety import get_sandbox_mirror_warning

        target = tmp_path / ".hermes" / "profiles" / "group1" / "SOUL.md"
        target.parent.mkdir(parents=True)
        target.write_text("# real SOUL\n")

        assert await get_sandbox_mirror_warning(str(target)) is None

    async def test_mirror_warning_names_mirror_root_and_inner_path(self, tmp_path):
        from agent.file_safety import get_sandbox_mirror_warning

        target = (
            tmp_path
            / "profiles"
            / "group1"
            / "sandboxes"
            / "docker"
            / "default"
            / "home"
            / ".hermes"
            / "profiles"
            / "group1"
            / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("# mirror copy\n")

        warning = await get_sandbox_mirror_warning(str(target))
        assert warning is not None
        assert "sandboxes/docker/default/home/.hermes" in warning
        assert "profiles/group1/SOUL.md" in warning
        assert "cross_profile=True" in warning

    async def test_warning_is_defense_in_depth_not_boundary(self, tmp_path):
        from agent.file_safety import get_sandbox_mirror_warning

        target = (
            tmp_path
            / "sandboxes"
            / "docker"
            / "t"
            / "home"
            / ".hermes"
            / "profiles"
            / "g"
            / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("x")

        warning = await get_sandbox_mirror_warning(str(target))
        assert "not a security boundary" in warning.lower()


class TestSandboxMirrorIsOrthogonalToCrossProfile:
    async def test_same_profile_mirror_still_flagged(self, tmp_path, monkeypatch):
        import agent.file_safety as fs

        monkeypatch.setattr(fs, "_hermes_root_path", lambda: tmp_path)
        monkeypatch.setattr(
            fs, "_hermes_home_path", lambda: tmp_path / "profiles" / "group1"
        )
        target = (
            tmp_path
            / "profiles"
            / "group1"
            / "sandboxes"
            / "docker"
            / "default"
            / "home"
            / ".hermes"
            / "profiles"
            / "group1"
            / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("x")

        assert await fs.classify_cross_profile_target(str(target)) is None
        assert await fs.classify_sandbox_mirror_target(str(target)) is not None


async def test_file_tool_guard_rejects_host_side_sandbox_mirror(tmp_path):
    import tools.file_tools as file_tools

    target = (
        tmp_path
        / "sandboxes"
        / "docker"
        / "task"
        / "home"
        / ".hermes"
        / "memories"
        / "MEMORY.md"
    )
    warning = await file_tools._check_cross_profile_path(str(target))
    assert warning is not None
    assert "Sandbox-mirror write blocked" in warning
    assert "memories/MEMORY.md" in warning
