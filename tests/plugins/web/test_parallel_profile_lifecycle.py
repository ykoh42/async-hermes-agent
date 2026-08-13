from __future__ import annotations

import asyncio
import gc
import os
import sys
import types
import weakref
from contextvars import ContextVar
from types import SimpleNamespace

import aiofiles.os
import pytest
from blockbuster import BlockBuster

from agent.secret_scope import (
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
from plugins.web.parallel import provider as parallel_provider


class _Owner:
    pass


class _FakeParallelClient:
    instances: list[_FakeParallelClient] = []
    search_started: asyncio.Event | None = None
    search_gate: asyncio.Event | None = None
    close_started: asyncio.Event | None = None
    close_gate: asyncio.Event | None = None
    extract_response = None
    search_calls: list[dict] = []
    extract_calls: list[dict] = []
    fail_init = False

    def __init__(self, *, api_key: str) -> None:
        if type(self).fail_init:
            raise RuntimeError("client construction failed")
        self.api_key = api_key
        self.beta = self
        self.close_calls = 0
        type(self).instances.append(self)

    async def search(self, **kwargs):
        type(self).search_calls.append(kwargs)
        started = type(self).search_started
        if started is not None:
            started.set()
        gate = type(self).search_gate
        if gate is not None:
            await gate.wait()
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    url=f"https://{self.api_key}.example",
                    title=self.api_key,
                    excerpts=["result"],
                )
            ]
        )

    async def extract(self, **kwargs):
        type(self).extract_calls.append(kwargs)
        if type(self).extract_response is not None:
            return type(self).extract_response
        return SimpleNamespace(results=[], errors=[])

    async def close(self) -> None:
        self.close_calls += 1
        started = type(self).close_started
        if started is not None:
            started.set()
        gate = type(self).close_gate
        if gate is not None:
            await gate.wait()


@pytest.fixture(autouse=True)
def _isolate_parallel_state(monkeypatch: pytest.MonkeyPatch):
    token = parallel_provider._parallel_scope_context.set(None)
    parallel_provider._parallel_scope_states.clear()
    parallel_provider._parallel_scope_aliases.clear()
    parallel_provider._parallel_owner_scopes.clear()
    parallel_provider._parallel_reset_profiles.clear()
    _FakeParallelClient.instances = []
    _FakeParallelClient.search_started = None
    _FakeParallelClient.search_gate = None
    _FakeParallelClient.close_started = None
    _FakeParallelClient.close_gate = None
    _FakeParallelClient.extract_response = None
    _FakeParallelClient.search_calls = []
    _FakeParallelClient.extract_calls = []
    _FakeParallelClient.fail_init = False
    monkeypatch.setitem(
        sys.modules,
        "parallel",
        types.SimpleNamespace(AsyncParallel=_FakeParallelClient),
    )
    yield
    parallel_provider._parallel_scope_states.clear()
    parallel_provider._parallel_scope_aliases.clear()
    parallel_provider._parallel_owner_scopes.clear()
    parallel_provider._parallel_reset_profiles.clear()
    parallel_provider._parallel_scope_context.reset(token)


def _install_profile_credentials(
    monkeypatch: pytest.MonkeyPatch,
    credentials: dict[str, str],
) -> None:
    async def get_provider_env(_name: str) -> str:
        return credentials[os.fspath(get_hermes_home())]

    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        get_provider_env,
    )


async def _under_profile(home, operation, *args):
    token = set_hermes_home_override(home)
    try:
        return await operation(*args)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_same_profile_agents_share_cache_until_final_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _install_profile_credentials(monkeypatch, {str(home): "shared-key"})
    first = _Owner()
    second = _Owner()

    await _under_profile(home, parallel_provider._retain_parallel_lifecycle, first)
    await _under_profile(home, parallel_provider._retain_parallel_lifecycle, second)
    provider = parallel_provider.ParallelWebSearchProvider()
    first_result = await _under_profile(home, provider.search, "first")
    second_result = await _under_profile(home, provider.search, "second")

    assert first_result["success"] is True
    assert second_result["success"] is True
    assert _FakeParallelClient.search_calls == [
        {
            "search_queries": ["first"],
            "objective": "first",
            "mode": "agentic",
            "max_results": 5,
        },
        {
            "search_queries": ["second"],
            "objective": "second",
            "mode": "agentic",
            "max_results": 5,
        },
    ]
    assert len(_FakeParallelClient.instances) == 1
    client = _FakeParallelClient.instances[0]
    await _under_profile(home, parallel_provider._release_parallel_lifecycle, first)
    assert client.close_calls == 0
    await _under_profile(home, parallel_provider._release_parallel_lifecycle, second)
    assert client.close_calls == 1
    assert not parallel_provider._parallel_scope_states


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_credentials_and_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    _install_profile_credentials(
        monkeypatch,
        {str(home_a): "key-a", str(home_b): "key-b"},
    )

    async def use_profile(home):
        owner = _Owner()
        await _under_profile(home, parallel_provider._retain_parallel_lifecycle, owner)
        result = await _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "query",
        )
        return owner, result

    (owner_a, result_a), (owner_b, result_b) = await asyncio.gather(
        use_profile(home_a),
        use_profile(home_b),
    )

    assert result_a["data"]["web"][0]["title"] == "key-a"
    assert result_b["data"]["web"][0]["title"] == "key-b"
    assert {client.api_key for client in _FakeParallelClient.instances} == {
        "key-a",
        "key-b",
    }
    await _under_profile(home_b, parallel_provider._release_parallel_lifecycle, owner_a)
    assert {client.api_key for client in _FakeParallelClient.instances if client.close_calls} == {
        "key-a"
    }
    await _under_profile(home_a, parallel_provider._release_parallel_lifecycle, owner_b)
    assert all(client.close_calls == 1 for client in _FakeParallelClient.instances)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_search_mode_and_explicit_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home_a = tmp_path / "mode-a"
    home_b = tmp_path / "mode-b"
    home_empty = tmp_path / "mode-empty"
    monkeypatch.setenv("PARALLEL_API_KEY", "process-key")
    monkeypatch.setenv("PARALLEL_SEARCH_MODE", "agentic")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def search(home, label: str, mode: str):
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope(
            {
                "PARALLEL_API_KEY": f"key-{label}",
                "PARALLEL_SEARCH_MODE": mode,
            }
        )
        try:
            return await parallel_provider.ParallelWebSearchProvider().search(label)
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        result_a, result_b, result_empty = await asyncio.gather(
            search(home_a, "alpha", "fast"),
            search(home_b, "beta", "one-shot"),
            search(home_empty, "empty", ""),
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert result_a["success"] is True
    assert result_b["success"] is True
    assert result_empty["success"] is True
    assert {
        call["objective"]: call["mode"]
        for call in _FakeParallelClient.search_calls
    } == {
        "alpha": "fast",
        "beta": "one-shot",
        "empty": "agentic",
    }
    assert {client.api_key for client in _FakeParallelClient.instances} == {
        "key-alpha",
        "key-beta",
        "key-empty",
    }


@pytest.mark.asyncio
async def test_single_profile_search_mode_keeps_process_env_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(False)
    try:
        monkeypatch.setenv("PARALLEL_SEARCH_MODE", " ONE-SHOT ")
        assert await parallel_provider._resolve_search_mode() == "one-shot"
        monkeypatch.setenv("PARALLEL_SEARCH_MODE", "invalid")
        assert await parallel_provider._resolve_search_mode() == "agentic"
    finally:
        set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
async def test_symlink_aliases_share_one_canonical_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    home.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(home, target_is_directory=True)
    _install_profile_credentials(
        monkeypatch,
        {str(home): "same-key", str(alias): "same-key"},
    )
    first = _Owner()
    second = _Owner()

    await _under_profile(home, parallel_provider._retain_parallel_lifecycle, first)
    await _under_profile(alias, parallel_provider._retain_parallel_lifecycle, second)
    provider = parallel_provider.ParallelWebSearchProvider()
    await _under_profile(home, provider.search, "first")
    await _under_profile(alias, provider.search, "second")

    assert len(_FakeParallelClient.instances) == 1
    await _under_profile(alias, parallel_provider._release_parallel_lifecycle, first)
    await _under_profile(home, parallel_provider._release_parallel_lifecycle, second)
    assert _FakeParallelClient.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_standalone_calls_close_their_owned_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "standalone"
    _install_profile_credentials(monkeypatch, {str(home): "standalone-key"})
    provider = parallel_provider.ParallelWebSearchProvider()

    await _under_profile(home, provider.search, "first")
    await _under_profile(home, provider.search, "second")

    assert len(_FakeParallelClient.instances) == 2
    assert all(client.close_calls == 1 for client in _FakeParallelClient.instances)
    assert not parallel_provider._parallel_scope_states


@pytest.mark.asyncio
async def test_extract_preserves_upstream_arguments_and_return_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "extract"
    _install_profile_credentials(monkeypatch, {str(home): "extract-key"})
    _FakeParallelClient.extract_response = SimpleNamespace(
        results=[
            SimpleNamespace(
                url="https://example.test/ok",
                title="Example",
                full_content="body",
                excerpts=["fallback"],
            )
        ],
        errors=[
            SimpleNamespace(
                url="https://example.test/fail",
                content="denied",
                error_type="fetch_error",
            )
        ],
    )

    result = await _under_profile(
        home,
        parallel_provider.ParallelWebSearchProvider().extract,
        ["https://example.test/ok", "https://example.test/fail"],
    )

    assert _FakeParallelClient.extract_calls == [
        {
            "urls": ["https://example.test/ok", "https://example.test/fail"],
            "full_content": True,
        }
    ]
    assert result == [
        {
            "url": "https://example.test/ok",
            "title": "Example",
            "content": "body",
            "raw_content": "body",
            "metadata": {
                "sourceURL": "https://example.test/ok",
                "title": "Example",
            },
        },
        {
            "url": "https://example.test/fail",
            "title": "",
            "content": "",
            "error": "denied",
            "metadata": {"sourceURL": "https://example.test/fail"},
        },
    ]
    assert _FakeParallelClient.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_final_agent_release_waits_for_active_search_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _install_profile_credentials(monkeypatch, {str(home): "active-key"})
    owner = _Owner()
    await _under_profile(home, parallel_provider._retain_parallel_lifecycle, owner)
    _FakeParallelClient.search_started = asyncio.Event()
    old_gate = asyncio.Event()
    _FakeParallelClient.search_gate = old_gate

    search_task = asyncio.create_task(
        _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "query",
        )
    )
    await _FakeParallelClient.search_started.wait()
    client = _FakeParallelClient.instances[0]
    await _under_profile(home, parallel_provider._release_parallel_lifecycle, owner)
    assert client.close_calls == 0

    _FakeParallelClient.search_gate.set()
    assert (await search_task)["success"] is True
    assert client.close_calls == 1
    assert not parallel_provider._parallel_scope_states


@pytest.mark.asyncio
async def test_one_agent_close_does_not_interrupt_sibling_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _install_profile_credentials(monkeypatch, {str(home): "sibling-key"})
    closing_owner = _Owner()
    active_owner = _Owner()
    await _under_profile(
        home,
        parallel_provider._retain_parallel_lifecycle,
        closing_owner,
    )
    await _under_profile(
        home,
        parallel_provider._retain_parallel_lifecycle,
        active_owner,
    )
    _FakeParallelClient.search_started = asyncio.Event()
    _FakeParallelClient.search_gate = asyncio.Event()
    search_task = asyncio.create_task(
        _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "query",
        )
    )
    await _FakeParallelClient.search_started.wait()
    client = _FakeParallelClient.instances[0]

    await _under_profile(
        home,
        parallel_provider._release_parallel_lifecycle,
        closing_owner,
    )
    assert client.close_calls == 0
    _FakeParallelClient.search_gate.set()
    assert (await search_task)["success"] is True
    assert client.close_calls == 0

    await _under_profile(
        home,
        parallel_provider._release_parallel_lifecycle,
        active_owner,
    )
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_standalone_search_releases_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "cancelled"
    _install_profile_credentials(monkeypatch, {str(home): "cancel-key"})
    _FakeParallelClient.search_started = asyncio.Event()
    _FakeParallelClient.search_gate = asyncio.Event()
    task = asyncio.create_task(
        _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "query",
        )
    )
    await _FakeParallelClient.search_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _FakeParallelClient.instances[0].close_calls == 1
    assert not parallel_provider._parallel_scope_states


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_final_client_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _install_profile_credentials(monkeypatch, {str(home): "close-key"})
    owner = _Owner()
    await _under_profile(home, parallel_provider._retain_parallel_lifecycle, owner)
    await _under_profile(
        home,
        parallel_provider.ParallelWebSearchProvider().search,
        "query",
    )
    _FakeParallelClient.close_started = asyncio.Event()
    _FakeParallelClient.close_gate = asyncio.Event()

    release_task = asyncio.create_task(
        _under_profile(home, parallel_provider._release_parallel_lifecycle, owner)
    )
    await _FakeParallelClient.close_started.wait()
    release_task.cancel()
    await asyncio.sleep(0)
    release_task.cancel()
    _FakeParallelClient.close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert _FakeParallelClient.instances[0].close_calls == 1
    assert owner not in parallel_provider._parallel_owner_scopes
    assert not parallel_provider._parallel_scope_states


@pytest.mark.asyncio
async def test_credential_rotation_retires_only_the_old_active_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    active_key: ContextVar[str] = ContextVar("parallel-test-key", default="old")

    async def get_provider_env(_name: str) -> str:
        return active_key.get()

    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        get_provider_env,
    )
    owner = _Owner()
    await _under_profile(home, parallel_provider._retain_parallel_lifecycle, owner)
    _FakeParallelClient.search_started = asyncio.Event()
    old_gate = asyncio.Event()
    _FakeParallelClient.search_gate = old_gate
    old_search = asyncio.create_task(
        _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "old",
        )
    )
    await _FakeParallelClient.search_started.wait()

    token = active_key.set("new")
    _FakeParallelClient.search_started = None
    _FakeParallelClient.search_gate = None
    try:
        result = await _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "new",
        )
    finally:
        active_key.reset(token)
    old_client, new_client = _FakeParallelClient.instances
    assert result["data"]["web"][0]["title"] == "new"
    assert old_client.close_calls == 0
    assert new_client.close_calls == 0

    old_gate.set()
    assert (await old_search)["success"] is True
    assert old_client.close_calls == 1
    assert new_client.close_calls == 0

    await _under_profile(home, parallel_provider._release_parallel_lifecycle, owner)
    assert new_client.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_credential_load_rolls_back_empty_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    started = asyncio.Event()
    gate = asyncio.Event()

    async def get_provider_env(_name: str) -> str:
        started.set()
        await gate.wait()
        return "unused"

    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        get_provider_env,
    )
    task = asyncio.create_task(
        _under_profile(
            tmp_path / "profile",
            parallel_provider.ParallelWebSearchProvider().search,
            "query",
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not parallel_provider._parallel_scope_states
    assert not _FakeParallelClient.instances


@pytest.mark.asyncio
async def test_profile_activation_and_client_lifecycle_do_not_block_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _install_profile_credentials(monkeypatch, {str(home): "key"})
    owner = _Owner()
    # Warm aiofiles' owned executor before BlockBuster starts inspecting the
    # event-loop thread; all profile path operations below still execute.
    await aiofiles.os.getcwd()
    blocker = BlockBuster()
    blocker.activate()
    try:
        await _under_profile(
            home,
            parallel_provider._retain_parallel_lifecycle,
            owner,
        )
        result = await _under_profile(
            home,
            parallel_provider.ParallelWebSearchProvider().search,
            "query",
        )
        await _under_profile(
            home,
            parallel_provider._release_parallel_lifecycle,
            owner,
        )
    finally:
        blocker.deactivate()

    assert result["success"] is True
    assert _FakeParallelClient.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_client_construction_failure_rolls_back_empty_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _install_profile_credentials(monkeypatch, {str(home): "key"})
    _FakeParallelClient.fail_init = True

    result = await _under_profile(
        home,
        parallel_provider.ParallelWebSearchProvider().search,
        "query",
    )

    assert result == {
        "success": False,
        "error": "Parallel search failed: client construction failed",
    }
    assert not parallel_provider._parallel_scope_states


def test_sequential_event_loops_do_not_retain_closed_loop_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    homes = [tmp_path / "first", tmp_path / "second"]
    _install_profile_credentials(
        monkeypatch,
        {str(homes[0]): "first", str(homes[1]): "second"},
    )
    loop_refs: list[weakref.ReferenceType[asyncio.AbstractEventLoop]] = []

    for home in homes:
        async def cycle() -> None:
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            owner = _Owner()
            await _under_profile(
                home,
                parallel_provider._retain_parallel_lifecycle,
                owner,
            )
            await _under_profile(
                home,
                parallel_provider.ParallelWebSearchProvider().search,
                "query",
            )
            await _under_profile(
                home,
                parallel_provider._release_parallel_lifecycle,
                owner,
            )

        asyncio.run(cycle())

    gc.collect()
    assert all(loop_ref() is None for loop_ref in loop_refs)
    assert not parallel_provider._parallel_scope_states
    assert not parallel_provider._parallel_scope_aliases
    assert all(client.close_calls == 1 for client in _FakeParallelClient.instances)
