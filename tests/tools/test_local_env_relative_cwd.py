"""Regression tests for local terminal initial cwd normalization."""

import pytest

from tools.environments.local import LocalEnvironment, _resolve_local_initial_cwd


def test_relative_initial_cwd_resolves_from_parent(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    assert _resolve_local_initial_cwd("hermes-agent") == str(project)


def test_relative_initial_cwd_does_not_skip_existing_nested_directory(
    tmp_path, monkeypatch
):
    project = tmp_path / "hermes-agent"
    nested = project / "hermes-agent"
    nested.mkdir(parents=True)
    monkeypatch.chdir(project)

    assert _resolve_local_initial_cwd("hermes-agent") == str(nested)


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
