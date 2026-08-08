"""Regression tests for local terminal initial cwd normalization."""

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools.environments.local import LocalEnvironment, _resolve_local_initial_cwd


@pytest.mark.asyncio
async def test_relative_initial_cwd_resolves_from_parent(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            resolved = await _resolve_local_initial_cwd("hermes-agent")
        finally:
            blocker.deactivate()

    assert resolved == str(project)


@pytest.mark.asyncio
async def test_relative_initial_cwd_does_not_skip_existing_nested_directory(
    tmp_path, monkeypatch
):
    project = tmp_path / "hermes-agent"
    nested = project / "hermes-agent"
    nested.mkdir(parents=True)
    monkeypatch.chdir(project)

    assert await _resolve_local_initial_cwd("hermes-agent") == str(nested)


@pytest.mark.asyncio
async def test_local_environment_keeps_existing_relative_child_cwd(
    tmp_path, monkeypatch
):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    environment = LocalEnvironment(cwd="hermes-agent", timeout=5)
    try:
        result = await environment.execute("pwd", timeout=5)
    finally:
        await environment.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == str(project)


@pytest.mark.asyncio
async def test_missing_nested_relative_cwd_recovers_on_async_execute(
    tmp_path, monkeypatch
):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(project)

    environment = LocalEnvironment(cwd="hermes-agent", timeout=5)
    try:
        result = await environment.execute("pwd", timeout=5)
    finally:
        await environment.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == str(project)
    assert environment.cwd == str(project)
