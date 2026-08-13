"""Profile isolation for browser and computer-use runtime ownership."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_browser_auxiliary_models_are_profile_scoped(monkeypatch):
    from tools import browser_tool

    monkeypatch.setenv("AUXILIARY_VISION_MODEL", "process-vision")
    monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_MODEL", "process-extract")
    set_multiplex_active(True)

    async def resolve(name: str):
        token = set_secret_scope(
            {
                "AUXILIARY_VISION_MODEL": f"vision-{name}",
                "AUXILIARY_WEB_EXTRACT_MODEL": f"extract-{name}",
            }
        )
        try:
            await asyncio.sleep(0)
            return (
                browser_tool._get_vision_model(),
                browser_tool._get_extraction_model(),
            )
        finally:
            reset_secret_scope(token)

    alpha, beta = await asyncio.gather(resolve("alpha"), resolve("beta"))
    assert alpha == ("vision-alpha", "extract-alpha")
    assert beta == ("vision-beta", "extract-beta")


def test_browser_auxiliary_model_lookup_fails_closed_without_scope(monkeypatch):
    from tools import browser_tool

    monkeypatch.setenv("AUXILIARY_VISION_MODEL", "foreign-vision")
    set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError, match="AUXILIARY_VISION_MODEL"):
        browser_tool._get_vision_model()


async def _browser_session_for(profile, browser_tool):
    token = set_hermes_home_override(profile)
    try:
        return await browser_tool._get_session_info("same-task")
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_browser_same_task_is_concurrent_and_cleanup_is_profile_scoped(
    monkeypatch,
    tmp_path,
):
    from tools import browser_tool

    monkeypatch.setattr(
        browser_tool, "_start_browser_cleanup_thread", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool, "_get_cdp_override", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool, "_get_cloud_provider", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool, "_ensure_cdp_supervisor", AsyncMock(return_value=None)
    )

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    session_a, session_b = await asyncio.gather(
        _browser_session_for(profile_a, browser_tool),
        _browser_session_for(profile_b, browser_tool),
    )
    assert session_a is not session_b
    assert session_a["session_name"] != session_b["session_name"]

    monkeypatch.setattr(
        browser_tool, "_stop_cdp_supervisor", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool, "_is_camofox_mode", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        AsyncMock(return_value={"success": True}),
    )
    monkeypatch.setattr(
        browser_tool, "_maybe_stop_recording", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool.aiofiles.os.path,
        "exists",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        browser_tool, "_stop_browser_cleanup_thread", AsyncMock(return_value=None)
    )

    token = set_hermes_home_override(profile_b)
    try:
        await browser_tool.cleanup_browser("same-task")
        assert "same-task" not in browser_tool._active_sessions
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_a)
    try:
        await browser_tool._activate_browser_scope()
        assert browser_tool._active_sessions["same-task"] is session_a
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_browser_provider_and_command_config_caches_are_profile_scoped(
    monkeypatch,
    tmp_path,
):
    from tools import browser_tool

    class Provider:
        def __init__(self):
            self.profile = get_hermes_home().name

    async def config():
        await asyncio.sleep(0)
        timeout = 11 if get_hermes_home().name == "profile-a" else 22
        return {
            "browser": {
                "cloud_provider": "browser-use",
                "command_timeout": timeout,
            }
        }

    monkeypatch.setattr(browser_tool, "load_config_readonly", config)
    monkeypatch.setattr(
        browser_tool,
        "_PROVIDER_REGISTRY",
        {"browser-use": Provider},
    )
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def resolve(profile):
        token = set_hermes_home_override(profile)
        try:
            provider = await browser_tool._get_cloud_provider()
            timeout = await browser_tool._get_command_timeout()
            return provider, timeout
        finally:
            reset_hermes_home_override(token)

    (provider_a, timeout_a), (provider_b, timeout_b) = await asyncio.gather(
        resolve(profile_a),
        resolve(profile_b),
    )
    assert provider_a is not None and provider_a.profile == "profile-a"
    assert provider_b is not None and provider_b.profile == "profile-b"
    assert provider_a is not provider_b
    assert (timeout_a, timeout_b) == (11, 22)

    again_a, again_b = await asyncio.gather(
        resolve(profile_a),
        resolve(profile_b),
    )
    assert again_a == (provider_a, 11)
    assert again_b == (provider_b, 22)


@pytest.mark.asyncio
async def test_browser_cleanup_tasks_and_supervisors_are_profile_owned(
    monkeypatch,
    tmp_path,
):
    from tools import browser_supervisor, browser_tool

    release = asyncio.Event()

    async def worker():
        await release.wait()

    monkeypatch.setattr(browser_tool, "_browser_cleanup_thread_worker", worker)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_inactivity_timeout",
        AsyncMock(return_value=120),
    )

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def start(profile):
        token = set_hermes_home_override(profile)
        try:
            await browser_tool._start_browser_cleanup_thread()
            return browser_tool._browser_state().cleanup_task
        finally:
            reset_hermes_home_override(token)

    task_a, task_b = await asyncio.gather(start(profile_a), start(profile_b))
    assert task_a is not None and task_b is not None and task_a is not task_b

    class Registry:
        def __init__(self):
            self.by_key = {}

        async def get_or_start(self, *, task_id, cdp_url, **_kwargs):
            key = (str(get_hermes_home()), task_id)
            supervisor = self.by_key.get(key)
            if supervisor is None:
                supervisor = SimpleNamespace(task_id=task_id, cdp_url=cdp_url)
                self.by_key[key] = supervisor
            return supervisor

        def get(self, task_id):
            return self.by_key.get((str(get_hermes_home()), task_id))

        async def stop(self, task_id):
            self.by_key.pop((str(get_hermes_home()), task_id), None)

    registry = Registry()
    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", registry)

    async def cdp_url():
        return f"ws://browser/{get_hermes_home().name}"

    monkeypatch.setattr(browser_tool, "_get_cdp_override", cdp_url)
    monkeypatch.setattr(
        browser_tool,
        "_get_dialog_policy_config",
        AsyncMock(return_value=("must_respond", 300.0)),
    )

    async def start_supervisor(profile):
        token = set_hermes_home_override(profile)
        try:
            await browser_tool._ensure_cdp_supervisor("same-task")
            return str(get_hermes_home()), "same-task"
        finally:
            reset_hermes_home_override(token)

    key_a, key_b = await asyncio.gather(
        start_supervisor(profile_a), start_supervisor(profile_b)
    )
    assert key_a != key_b
    assert registry.by_key[key_a].task_id == "same-task"
    assert registry.by_key[key_b].task_id == "same-task"

    token = set_hermes_home_override(profile_b)
    try:
        await browser_tool._stop_cdp_supervisor("same-task")
        await browser_tool._stop_browser_cleanup_thread()
    finally:
        reset_hermes_home_override(token)
    assert key_b not in registry.by_key
    assert key_a in registry.by_key
    assert task_b.done()
    assert not task_a.done()

    token = set_hermes_home_override(profile_a)
    try:
        await browser_tool._stop_cdp_supervisor("same-task")
        await browser_tool._stop_browser_cleanup_thread()
    finally:
        reset_hermes_home_override(token)
    assert task_a.done()


@pytest.mark.asyncio
async def test_orphan_reaper_sees_live_sessions_owned_by_another_profile(
    monkeypatch,
    tmp_path,
):
    from tools import browser_tool

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    socket_dir = tmp_path / "agent-browser-h_owned"
    socket_dir.mkdir()

    token = set_hermes_home_override(profile_a)
    try:
        await browser_tool._activate_browser_scope()
        browser_tool._active_sessions["same-task"] = {
            "session_name": "h_owned"
        }
    finally:
        reset_hermes_home_override(token)

    monkeypatch.setattr(
        browser_tool, "_socket_safe_tmpdir", AsyncMock(return_value=str(tmp_path))
    )
    remove_tree = AsyncMock()
    terminate = AsyncMock()
    monkeypatch.setattr(browser_tool, "_remove_tree", remove_tree)
    monkeypatch.setattr(browser_tool, "_terminate_host_pid", terminate)

    token = set_hermes_home_override(profile_b)
    try:
        await browser_tool._reap_orphaned_browser_sessions()
    finally:
        reset_hermes_home_override(token)

    assert socket_dir.exists()
    remove_tree.assert_not_awaited()
    terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_camofox_same_task_is_isolated_and_b_drop_does_not_drop_a(
    monkeypatch,
    tmp_path,
):
    from tools import browser_camofox as camofox

    monkeypatch.setattr(
        camofox,
        "_get_camofox_config",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        camofox,
        "_adopt_existing_tab",
        AsyncMock(side_effect=lambda session: session),
    )
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def get(profile):
        token = set_hermes_home_override(profile)
        try:
            return await camofox._get_session("same-task")
        finally:
            reset_hermes_home_override(token)

    session_a, session_b = await asyncio.gather(get(profile_a), get(profile_b))
    assert session_a is not session_b
    assert session_a["user_id"] != session_b["user_id"]

    token = set_hermes_home_override(profile_b)
    try:
        assert await camofox._drop_session("same-task") is session_b
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_a)
    try:
        await camofox._activate_camofox_scope()
        assert camofox._sessions["same-task"] is session_a
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_computer_use_same_session_backend_and_approvals_are_profile_scoped(
    monkeypatch,
    tmp_path,
):
    from tools.computer_use import cua_backend
    from tools.computer_use import tool as computer_use

    created = []

    class Backend:
        def __init__(self, permission_mode="standard"):
            self.profile = get_hermes_home().name
            self.permission_mode = permission_mode
            self.stopped = False
            created.append(self)

        async def start(self):
            await asyncio.sleep(0)

        async def stop(self):
            self.stopped = True

    monkeypatch.setattr(cua_backend, "CuaDriverBackend", Backend)
    monkeypatch.setattr(computer_use, "_backend", None)
    monkeypatch.setattr(computer_use, "_cua_permission_mode", lambda _sid: "standard")
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def get(profile):
        token = set_hermes_home_override(profile)
        try:
            return await computer_use._get_backend("same-session")
        finally:
            reset_hermes_home_override(token)

    backend_a, backend_b = await asyncio.gather(get(profile_a), get(profile_b))
    assert backend_a is not backend_b
    assert {backend_a.profile, backend_b.profile} == {"profile-a", "profile-b"}

    token = set_hermes_home_override(profile_a)
    try:
        await computer_use._activate_computer_use_scope()
        computer_use.set_approval_callback(
            lambda _action, _args, _summary: "always_approve"
        )
        assert await computer_use._request_approval("click", {}, "same-session") is None
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_b)
    try:
        await computer_use._activate_computer_use_scope()
        computer_use.set_approval_callback(lambda *_args: "deny")
        denied = await computer_use._request_approval(
            "click", {}, "same-session"
        )
        assert denied is not None and "denied" in denied
        assert await computer_use.release_computer_use_session("same-session")
    finally:
        reset_hermes_home_override(token)

    assert backend_b.stopped is True
    assert backend_a.stopped is False

    token = set_hermes_home_override(profile_a)
    try:
        assert await computer_use._get_backend("same-session") is backend_a
        assert await computer_use.release_computer_use_session("same-session")
    finally:
        reset_hermes_home_override(token)
    assert backend_a.stopped is True
    assert len(created) == 2
    assert {id(item) for item in created} == {id(backend_a), id(backend_b)}


def test_browser_no_loop_private_mapping_staging_migrates_at_first_await(tmp_path):
    from tools import browser_tool

    profile = tmp_path / "profile"
    token = set_hermes_home_override(profile)
    try:
        staged = {"session_name": "h_staged"}
        browser_tool._active_sessions["staged-task"] = staged

        async def read_staged():
            await browser_tool._activate_browser_scope()
            value = browser_tool._active_sessions.pop("staged-task")
            return value

        assert asyncio.run(read_staged()) is staged
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_canonical_profile_alias_reuses_computer_use_backend(
    monkeypatch,
    tmp_path,
):
    from tools.computer_use import cua_backend
    from tools.computer_use import tool as computer_use

    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)

    class Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(cua_backend, "CuaDriverBackend", Backend)
    monkeypatch.setattr(computer_use, "_backend", None)
    monkeypatch.setattr(computer_use, "_cua_permission_mode", lambda _sid: "standard")

    token = set_hermes_home_override(profile)
    try:
        backend = await computer_use._get_backend("same-session")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(alias)
    try:
        assert await computer_use._get_backend("same-session") is backend
        await computer_use.release_computer_use_session("same-session")
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_computer_use_profile_release_survives_repeated_cancellation(
    monkeypatch,
    tmp_path,
):
    from tools.computer_use import tool as computer_use

    first_stop = asyncio.Event()
    cleanup_stop = asyncio.Event()
    finish = asyncio.Event()

    class Backend:
        stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                first_stop.set()
            else:
                cleanup_stop.set()
            await finish.wait()

    backend = Backend()
    profile = tmp_path / "profile"
    token = set_hermes_home_override(profile)
    try:
        await computer_use._activate_computer_use_scope()
        monkeypatch.setattr(computer_use, "_backend", None)
        computer_use._backends["same-session"] = backend
        computer_use._backend_call_locks["same-session"] = asyncio.Lock()
        computer_use._backend_permission_modes["same-session"] = "standard"

        release = asyncio.create_task(
            computer_use.release_computer_use_session("same-session")
        )
        await first_stop.wait()
        release.cancel()
        await cleanup_stop.wait()
        release.cancel()
        await asyncio.sleep(0)
        release.cancel()
        await asyncio.sleep(0)
        assert not release.done()

        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await release
        assert backend.stop_calls == 2
        assert "same-session" not in computer_use._backends
    finally:
        reset_hermes_home_override(token)
