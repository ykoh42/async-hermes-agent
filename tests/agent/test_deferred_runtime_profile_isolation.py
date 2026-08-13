"""Profile isolation for deferred custom-provider credential resolution."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent import agent_init, agent_runtime_helpers, secret_scope, ssl_guard
from hermes_cli import env_loader
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _restore_secret_scope_state():
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(False)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(token)


@pytest.fixture
def _deferred_runtime_boundaries(monkeypatch):
    async def _noop(*_args, **_kwargs):  # noqa: ASYNC124 - coroutine test double
        return None

    monkeypatch.setattr(
        "tools.async_delegation.restore_undelivered_completions", _noop
    )
    monkeypatch.setattr("providers._ensure_provider_profiles_loaded", _noop)
    monkeypatch.setattr(agent_init, "_select_context_engine", _noop)
    monkeypatch.setattr(agent_init, "_initialize_memory_manager", _noop)
    monkeypatch.setattr(agent_init, "_initialize_context_engine", _noop)
    monkeypatch.setattr(ssl_guard, "verify_ca_bundle_with_fallback", _noop)


@pytest.fixture
def dotenv_profile_homes(tmp_path: Path) -> dict[str, Path]:
    homes: dict[str, Path] = {}
    for label in ("a", "b"):
        home = tmp_path / f"profile-{label}"
        home.mkdir()
        (home / ".env").write_text(
            f"OPENAI_API_KEY=profile-{label}-key\n",
            encoding="utf-8",
        )
        homes[label] = home
    return homes


@dataclass
class _OwnedClient:
    api_key: str
    close_calls: int = 0
    closed: bool = False
    close_started: asyncio.Event | None = None
    allow_close: asyncio.Event | None = None

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_started is not None:
            self.close_started.set()
        if self.allow_close is not None:
            await self.allow_close.wait()
        self.closed = True


def _make_custom_agent(
    label: str,
    *,
    api_key: str | None = None,
    dotenv_loaded: bool = True,
) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            model=f"model-{label}",
            provider="custom",
            api_key=api_key,
            base_url=f"https://{label}.example.test/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
    agent._dotenv_loaded = dotenv_loaded
    agent._runtime_config_loaded = True
    agent._runtime_config_snapshot = {}
    agent.context_compressor = None
    return agent


def test_unscoped_multiplex_constructor_defers_terminal_cwd(monkeypatch) -> None:
    monkeypatch.setenv("TERMINAL_CWD", "/foreign/process/workspace")
    secret_scope.set_multiplex_active(True)

    agent = _make_custom_agent("unscoped-constructor")

    assert agent._subdirectory_hints.working_dir is None


def test_constructor_terminal_cwd_single_profile_and_scoped_parity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    process_cwd = tmp_path / "process"
    scoped_cwd = tmp_path / "scoped"
    process_cwd.mkdir()
    scoped_cwd.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(process_cwd))

    single = _make_custom_agent("single-profile")
    assert single._subdirectory_hints.working_dir == process_cwd

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TERMINAL_CWD": str(scoped_cwd)})
    try:
        scoped = _make_custom_agent("scoped-constructor")
    finally:
        secret_scope.reset_secret_scope(token)

    assert scoped._subdirectory_hints.working_dir == scoped_cwd


@pytest.mark.asyncio
async def test_first_awaited_runtime_binds_concurrent_profile_cwds(
    monkeypatch,
    tmp_path: Path,
    _deferred_runtime_boundaries,
) -> None:
    monkeypatch.setenv("TERMINAL_CWD", "/foreign/process/workspace")
    secret_scope.set_multiplex_active(True)
    agents = {label: _make_custom_agent(label) for label in ("a", "b")}
    for agent in agents.values():
        agent._deferred_provider_runtime = None

    profile_cwds = {label: tmp_path / label for label in agents}
    for cwd in profile_cwds.values():
        cwd.mkdir()

    assert await agents["a"]._ensure_provider_runtime() is False
    assert agents["a"]._subdirectory_hints.working_dir is None

    async def initialize(label: str) -> Path | None:
        token = secret_scope.set_secret_scope(
            {"TERMINAL_CWD": str(profile_cwds[label])}
        )
        try:
            assert await agents[label]._ensure_provider_runtime() is False
            return agents[label]._subdirectory_hints.working_dir
        finally:
            secret_scope.reset_secret_scope(token)

    assert await asyncio.gather(initialize("a"), initialize("b")) == [
        profile_cwds["a"],
        profile_cwds["b"],
    ]


@pytest.mark.asyncio
async def test_custom_explicit_base_url_uses_concurrent_profile_keys(
    monkeypatch,
    _deferred_runtime_boundaries,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-key")
    secret_scope.set_multiplex_active(True)
    agents = {label: _make_custom_agent(label) for label in ("a", "b")}
    clients: dict[str, _OwnedClient] = {}
    both_creating = asyncio.Event()
    arrival_count = 0

    async def _create_client(_agent, client_kwargs, **_kwargs):
        nonlocal arrival_count
        arrival_count += 1
        if arrival_count == len(agents):
            both_creating.set()
        await both_creating.wait()
        client = _OwnedClient(str(client_kwargs["api_key"]))
        clients[_agent.model] = client
        return client

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )

    async def _initialize(label: str) -> None:
        token = secret_scope.set_secret_scope(
            {"OPENAI_API_KEY": f"profile-{label}-key"}
        )
        try:
            assert await agents[label]._ensure_provider_runtime() is True
        finally:
            secret_scope.reset_secret_scope(token)

    await asyncio.gather(*(_initialize(label) for label in agents))

    assert clients["model-a"].api_key == "profile-a-key"
    assert clients["model-b"].api_key == "profile-b-key"
    assert all(agent.api_key != "foreign-process-key" for agent in agents.values())

    await asyncio.gather(*(agent.release_clients() for agent in agents.values()))
    assert all(client.closed for client in clients.values())
    assert all(client.close_calls == 1 for client in clients.values())


@pytest.mark.asyncio
async def test_multiplex_profile_dotenv_isolated_from_process_and_subprocess(
    monkeypatch,
    _deferred_runtime_boundaries,
    dotenv_profile_homes,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secret_scope.set_multiplex_active(True)
    agents = {
        label: _make_custom_agent(label, dotenv_loaded=False)
        for label in dotenv_profile_homes
    }
    observed: dict[str, tuple[str, str | None, str]] = {}
    both_creating = asyncio.Event()
    arrival_count = 0
    dotenv_loader = AsyncMock(wraps=env_loader.load_hermes_dotenv)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", dotenv_loader)

    async def _create_client(_agent, client_kwargs, **_kwargs):
        nonlocal arrival_count
        arrival_count += 1
        if arrival_count == len(agents):
            both_creating.set()
        await both_creating.wait()

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            (
                "import os, sys; "
                "sys.stdout.write(os.getenv('OPENAI_API_KEY', '<missing>'))"
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await process.communicate()
        assert process.returncode == 0
        label = _agent.model.removeprefix("model-")
        observed[label] = (
            str(client_kwargs["api_key"]),
            os.getenv("OPENAI_API_KEY"),
            stdout.decode(),
        )
        return _OwnedClient(str(client_kwargs["api_key"]))

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )

    async def _initialize(label: str) -> None:
        home = dotenv_profile_homes[label]
        home_token = set_hermes_home_override(home)
        scope_token = secret_scope.set_secret_scope(
            await secret_scope.build_profile_secret_scope(home)
        )
        try:
            assert await agents[label]._ensure_provider_runtime() is True
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        await asyncio.gather(*(_initialize(label) for label in agents))
    finally:
        await asyncio.gather(*(agent.release_clients() for agent in agents.values()))

    dotenv_loader.assert_not_awaited()
    assert observed == {
        "a": ("profile-a-key", None, "<missing>"),
        "b": ("profile-b-key", None, "<missing>"),
    }
    assert os.getenv("OPENAI_API_KEY") is None


@pytest.mark.asyncio
async def test_multiplex_profile_dotenv_isolated_across_sequential_agents(
    monkeypatch,
    _deferred_runtime_boundaries,
    dotenv_profile_homes,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secret_scope.set_multiplex_active(True)
    observed: list[tuple[str, str | None]] = []
    dotenv_loader = AsyncMock(wraps=env_loader.load_hermes_dotenv)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", dotenv_loader)

    async def _create_client(_agent, client_kwargs, **_kwargs):
        observed.append(
            (str(client_kwargs["api_key"]), os.getenv("OPENAI_API_KEY"))
        )
        return _OwnedClient(str(client_kwargs["api_key"]))

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )

    for label, home in dotenv_profile_homes.items():
        agent = _make_custom_agent(label, dotenv_loaded=False)
        home_token = set_hermes_home_override(home)
        scope_token = secret_scope.set_secret_scope(
            await secret_scope.build_profile_secret_scope(home)
        )
        try:
            assert await agent._ensure_provider_runtime() is True
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)
            await agent.release_clients()

    dotenv_loader.assert_not_awaited()
    assert observed == [
        ("profile-a-key", None),
        ("profile-b-key", None),
    ]
    assert os.getenv("OPENAI_API_KEY") is None


@pytest.mark.asyncio
async def test_custom_explicit_base_url_fails_closed_without_profile_scope(
    monkeypatch,
    _deferred_runtime_boundaries,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-key")
    secret_scope.set_multiplex_active(True)
    agent = _make_custom_agent("unscoped", dotenv_loaded=False)
    dotenv_loader = AsyncMock(wraps=env_loader.load_hermes_dotenv)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", dotenv_loader)
    create_calls = 0

    async def _create_client(  # noqa: ASYNC124 - coroutine-shaped test double
        *_args, **_kwargs
    ):
        nonlocal create_calls
        create_calls += 1
        return _OwnedClient("unexpected")

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )

    with pytest.raises(secret_scope.UnscopedSecretError, match="OPENAI_API_KEY"):
        await agent._ensure_provider_runtime()

    assert create_calls == 0
    dotenv_loader.assert_not_awaited()
    assert os.getenv("OPENAI_API_KEY") == "foreign-process-key"
    assert agent.client is None
    assert agent._deferred_provider_runtime is not None


@pytest.mark.asyncio
async def test_single_profile_process_environment_fallback_is_preserved(
    monkeypatch,
    _deferred_runtime_boundaries,
):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-process-key")
    agent = _make_custom_agent("legacy")
    created: list[_OwnedClient] = []

    async def _create_client(  # noqa: ASYNC124 - coroutine-shaped test double
        _agent, client_kwargs, **_kwargs
    ):
        client = _OwnedClient(str(client_kwargs["api_key"]))
        created.append(client)
        return client

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )

    assert await agent._ensure_provider_runtime() is True
    assert agent.api_key == "legacy-process-key"
    assert created[0].api_key == "legacy-process-key"
    await agent.release_clients()
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_single_profile_dotenv_precedence_is_preserved(
    monkeypatch,
    _deferred_runtime_boundaries,
    dotenv_profile_homes,
):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-process-key")
    home = dotenv_profile_homes["a"]
    agent = _make_custom_agent("legacy-dotenv", dotenv_loaded=False)
    created: list[_OwnedClient] = []
    dotenv_loader = AsyncMock(wraps=env_loader.load_hermes_dotenv)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", dotenv_loader)

    async def _create_client(_agent, client_kwargs, **_kwargs):
        client = _OwnedClient(str(client_kwargs["api_key"]))
        created.append(client)
        return client

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )
    home_token = set_hermes_home_override(home)
    try:
        assert await agent._ensure_provider_runtime() is True
    finally:
        reset_hermes_home_override(home_token)

    dotenv_loader.assert_awaited_once()
    assert agent.api_key == "profile-a-key"
    assert created[0].api_key == "profile-a-key"
    assert os.getenv("OPENAI_API_KEY") == "profile-a-key"
    await agent.release_clients()
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_explicit_key_precedes_unscoped_environment_lookup(
    monkeypatch,
    _deferred_runtime_boundaries,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-key")
    secret_scope.set_multiplex_active(True)
    agent = _make_custom_agent(
        "explicit",
        api_key="  explicit-key  ",
        dotenv_loaded=False,
    )
    created: list[_OwnedClient] = []
    dotenv_loader = AsyncMock(wraps=env_loader.load_hermes_dotenv)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", dotenv_loader)

    async def _create_client(  # noqa: ASYNC124 - coroutine-shaped test double
        _agent, client_kwargs, **_kwargs
    ):
        client = _OwnedClient(str(client_kwargs["api_key"]))
        created.append(client)
        return client

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )

    assert await agent._ensure_provider_runtime() is True
    dotenv_loader.assert_not_awaited()
    assert agent.api_key == "explicit-key"
    assert created[0].api_key == "explicit-key"
    await agent.release_clients()
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_whitespace_scoped_key_preserves_runtime_resolver_fallback(
    monkeypatch,
    _deferred_runtime_boundaries,
):
    secret_scope.set_multiplex_active(True)
    agent = _make_custom_agent("empty")
    resolver_calls: list[dict[str, object]] = []

    async def _resolve_runtime_provider(  # noqa: ASYNC124 - coroutine test double
        **kwargs,
    ):
        resolver_calls.append(kwargs)
        return {
            "provider": "custom",
            "requested_provider": "custom",
            "api_key": "resolver-key",
            "base_url": kwargs["explicit_base_url"],
        }

    async def _create_client(  # noqa: ASYNC124 - coroutine-shaped test double
        _agent, client_kwargs, **_kwargs
    ):
        return _OwnedClient(str(client_kwargs["api_key"]))

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _resolve_runtime_provider,
    )
    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )
    token = secret_scope.set_secret_scope({"OPENAI_API_KEY": "  \t "})
    try:
        assert await agent._ensure_provider_runtime() is True
    finally:
        secret_scope.reset_secret_scope(token)

    assert resolver_calls == [
        {
            "requested": "custom",
            "explicit_api_key": None,
            "explicit_base_url": "https://empty.example.test/v1",
            "target_model": "model-empty",
        }
    ]
    assert agent.api_key == "resolver-key"
    await agent.release_clients()


@pytest.mark.asyncio
async def test_cancelled_initialization_keeps_owned_client_releasable(
    monkeypatch,
    _deferred_runtime_boundaries,
):
    secret_scope.set_multiplex_active(True)
    agent = _make_custom_agent("cancel", dotenv_loaded=False)
    dotenv_loader = AsyncMock(wraps=env_loader.load_hermes_dotenv)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", dotenv_loader)
    initialization_blocked = asyncio.Event()
    never_finish = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    client = _OwnedClient(
        "profile-cancel-key",
        close_started=close_started,
        allow_close=allow_close,
    )

    async def _create_client(  # noqa: ASYNC124 - coroutine-shaped test double
        *_args, **_kwargs
    ):
        return client

    async def _block_memory_initialization(*_args, **_kwargs):
        initialization_blocked.set()
        await never_finish.wait()

    monkeypatch.setattr(
        agent_runtime_helpers, "create_openai_client", _create_client
    )
    monkeypatch.setattr(
        agent_init,
        "_initialize_memory_manager",
        _block_memory_initialization,
    )

    token = secret_scope.set_secret_scope(
        {"OPENAI_API_KEY": "profile-cancel-key"}
    )
    try:
        initialize_task = asyncio.create_task(agent._ensure_provider_runtime())
        await initialization_blocked.wait()
        initialize_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await initialize_task
    finally:
        secret_scope.reset_secret_scope(token)

    assert agent.client is client
    release_task = asyncio.create_task(agent.release_clients())
    await close_started.wait()
    release_task.cancel()
    await asyncio.sleep(0)
    release_task.cancel()
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert client.closed is True
    assert client.close_calls == 1
    assert agent.client is None
    dotenv_loader.assert_not_awaited()
