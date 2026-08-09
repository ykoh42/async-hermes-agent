"""Tests for the native-async Vertex AI adapter."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction


class _Credentials:
    def __init__(self, *, project_id=None, token="ya29.FAKE"):
        self.project_id = project_id
        self.token = None
        self.expiry = None
        self.expired = False
        self._next_token = token
        self.refresh_count = 0

    async def refresh(self, request):
        await asyncio.sleep(0)
        self.refresh_count += 1
        self.token = self._next_token
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


class _Request:
    instances = []

    def __init__(self):
        self.closed = False
        self.instances.append(self)

    async def close(self):
        self.closed = True


@pytest.fixture
def vertex_adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in (
        "VERTEX_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_PROJECT_ID",
        "VERTEX_REGION",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GCE_METADATA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    import agent.vertex_adapter as va

    va = importlib.reload(va)
    va._creds_cache.clear()
    va._cache_locks.clear()
    _Request.instances.clear()
    monkeypatch.setattr(
        va,
        "aiohttp_requests",
        SimpleNamespace(Request=_Request),
    )
    monkeypatch.setattr(
        va,
        "service_account",
        SimpleNamespace(
            Credentials=SimpleNamespace(
                from_service_account_info=lambda info, scopes: _Credentials(
                    project_id=info.get("project_id")
                )
            )
        ),
    )
    monkeypatch.setattr(
        va,
        "user_credentials",
        SimpleNamespace(
            Credentials=SimpleNamespace(
                from_authorized_user_info=lambda info, scopes: _Credentials()
            )
        ),
    )
    adc_path = tmp_path / "missing-adc.json"
    monkeypatch.setattr(
        va,
        "_cloud_sdk",
        SimpleNamespace(
            get_application_default_credentials_path=lambda: str(adc_path)
        ),
    )
    return va


@pytest.mark.asyncio
async def test_service_account_refresh_is_awaited_and_cached(
    vertex_adapter, monkeypatch, tmp_path
):
    credentials_path = tmp_path / "service-account.json"
    credentials_path.write_text(
        '{"type":"service_account","project_id":"sa-project"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(credentials_path))

    first = await vertex_adapter.get_vertex_credentials()
    second = await vertex_adapter.get_vertex_credentials()

    assert first == second == ("ya29.FAKE", "sa-project")
    creds, _ = vertex_adapter._creds_cache[str(credentials_path)]
    assert creds.refresh_count == 1
    assert _Request.instances[0].closed is True


@pytest.mark.asyncio
async def test_authorized_user_adc_uses_async_refresh_and_gcloud_project(
    vertex_adapter, monkeypatch, tmp_path
):
    adc_path = tmp_path / "application_default_credentials.json"
    adc_path.write_text('{"type":"authorized_user"}', encoding="utf-8")
    monkeypatch.setattr(
        vertex_adapter,
        "_cloud_sdk",
        SimpleNamespace(get_application_default_credentials_path=lambda: str(adc_path)),
    )

    async def project_id():
        return "gcloud-project"

    monkeypatch.setattr(vertex_adapter, "_gcloud_project_id", project_id)

    assert await vertex_adapter.get_vertex_credentials() == (
        "ya29.FAKE",
        "gcloud-project",
    )


@pytest.mark.asyncio
async def test_gcloud_probe_reaps_process_through_repeated_cancellation(
    vertex_adapter, monkeypatch
):
    communicate_started = asyncio.Event()
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()
    wait_completed = asyncio.Event()

    class BlockingProcess:
        returncode = None
        killed = False

        async def communicate(self):
            communicate_started.set()
            await asyncio.Event().wait()

        async def wait(self):
            wait_started.set()
            await release_wait.wait()
            wait_completed.set()
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(vertex_adapter._gcloud_project_id())
    await communicate_started.wait()
    task.cancel()
    await wait_started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wait_completed.is_set()
    assert process.killed is True


@pytest.mark.asyncio
async def test_metadata_adc_is_cached_with_refresh_margin(vertex_adapter, monkeypatch):
    calls = 0

    async def metadata_credentials():
        nonlocal calls
        calls += 1
        vertex_adapter._creds_cache["__metadata__"] = {
            "token": "metadata-token",
            "project_id": "metadata-project",
            "expires_at": vertex_adapter.time.time() + 3600,
        }
        return "metadata-token", "metadata-project"

    monkeypatch.setattr(vertex_adapter, "_metadata_credentials", metadata_credentials)

    assert await vertex_adapter.get_vertex_credentials() == (
        "metadata-token",
        "metadata-project",
    )
    assert await vertex_adapter.get_vertex_credentials() == (
        "metadata-token",
        "metadata-project",
    )
    assert calls == 1


@pytest.mark.asyncio
async def test_get_vertex_config_preserves_regional_url(
    vertex_adapter, monkeypatch
):
    async def credentials(path=None):
        return "token", "project"

    monkeypatch.setattr(vertex_adapter, "get_vertex_credentials", credentials)

    assert await vertex_adapter.get_vertex_config(region="us-central1") == (
        "token",
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/"
        "project/locations/us-central1/endpoints/openapi",
    )


def test_build_vertex_base_url_global_and_regional(vertex_adapter):
    assert vertex_adapter.build_vertex_base_url("p") == (
        "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/"
        "global/endpoints/openapi"
    )
    assert vertex_adapter.build_vertex_base_url("p", "europe-west4") == (
        "https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/p/"
        "locations/europe-west4/endpoints/openapi"
    )


@pytest.mark.asyncio
async def test_has_vertex_credentials_via_config_project(vertex_adapter, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "vertex:\n  project_id: p\n",
        encoding="utf-8",
    )

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            available = await vertex_adapter.has_vertex_credentials()
        finally:
            blockbuster.deactivate()

    assert available is True


@pytest.mark.asyncio
async def test_has_vertex_credentials_false_when_nothing_set(vertex_adapter):
    assert await vertex_adapter.has_vertex_credentials() is False


@pytest.mark.asyncio
async def test_multiplex_scope_takes_precedence_over_raw_environ(
    vertex_adapter, monkeypatch
):
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "other-profile-project")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(
        {"VERTEX_PROJECT_ID": "this-profile-project"}
    )
    try:
        assert await vertex_adapter._resolve_project_override() == "this-profile-project"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_multiplex_unscoped_read_fails_closed(vertex_adapter, monkeypatch):
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "leaked-project")
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            await vertex_adapter._resolve_project_override()
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_adc_refuses_foreign_profile_google_application_credentials(
    vertex_adapter, monkeypatch, tmp_path
):
    from agent import secret_scope

    sa_file = tmp_path / "other_profile_sa.json"
    sa_file.write_text(
        '{"type":"service_account","project_id":"other-profile"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_file))

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        assert await vertex_adapter.get_vertex_credentials() == (None, None)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_refresh_cancellation_closes_transport(vertex_adapter, monkeypatch):
    started = asyncio.Event()

    class BlockingCredentials:
        async def refresh(self, request):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        vertex_adapter._refresh_credentials(BlockingCredentials())
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _Request.instances[-1].closed is True
