"""Profile-scoped gcloud project resolution for the Vertex adapter."""

from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from agent import vertex_adapter
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_gcloud_project_id_isolates_concurrent_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "process-profile-project")
    secret_scope.set_multiplex_active(True)

    async def resolve(profile, project):
        home_token = set_hermes_home_override(profile)
        secret_token = secret_scope.set_secret_scope({
            "GOOGLE_CLOUD_PROJECT": project,
        })
        try:
            await asyncio.sleep(0)
            return await vertex_adapter._gcloud_project_id()
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    assert await asyncio.gather(
        resolve(tmp_path / "profile-a", "profile-a-project"),
        resolve(tmp_path / "profile-b", "profile-b-project"),
    ) == ["profile-a-project", "profile-b-project"]


@pytest.mark.asyncio
async def test_gcloud_project_id_missing_scope_value_does_not_borrow_process_env(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "process-profile-project")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    called = False

    class Process:
        returncode = 0

        async def communicate(self):
            return b"gcloud-profile-project\n", b""

    async def create_process(*_args, **_kwargs):
        nonlocal called
        called = True
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    try:
        assert await vertex_adapter._gcloud_project_id() is None
    finally:
        secret_scope.reset_secret_scope(token)

    assert called is False
