"""Tests for the container-context sandbox-mirror guard."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.asyncio


class TestClassifyContainerMirrorTarget:
    async def test_catches_soul_md_with_context(self):
        from agent.file_safety import classify_container_mirror_target

        result = await classify_container_mirror_target(
            "/root/.hermes/profiles/group1/SOUL.md",
            mirror_prefix="/root/.hermes",
        )
        assert result is not None
        assert result["mirror_root"].replace("\\", "/").endswith("root/.hermes")
        assert result["inner_path"] == "profiles/group1/SOUL.md"

    @pytest.mark.parametrize("inner", ["SOUL.md", "memories/MEMORY.md"])
    async def test_catches_authoritative_profile_files(self, inner):
        from agent.file_safety import classify_container_mirror_target

        result = await classify_container_mirror_target(
            f"/root/.hermes/{inner}",
            mirror_prefix="/root/.hermes",
        )
        assert result is not None
        assert result["inner_path"] == inner


class TestGetContainerMirrorWarning:
    async def test_warning_names_inner_path_and_bypass(self):
        from agent.file_safety import get_container_mirror_warning

        warning = await get_container_mirror_warning(
            "/root/.hermes/profiles/group1/SOUL.md",
            mirror_prefix="/root/.hermes",
        )
        assert warning is not None
        assert "profiles/group1/SOUL.md" in warning
        assert "cross_profile=True" in warning


class TestOrthogonality:
    async def test_inner_container_path_caught_by_context_guard(self):
        from agent.file_safety import classify_container_mirror_target

        path = "/root/.hermes/profiles/group1/SOUL.md"
        assert await classify_container_mirror_target(path) is None
        assert (
            await classify_container_mirror_target(
                path, mirror_prefix="/root/.hermes"
            )
            is not None
        )
