"""Profile and lifecycle coverage for the retained single-task runner."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

import mini_swe_runner as runner_module
from agent import secret_scope
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from mini_swe_runner import MiniSWERunner


pytestmark = pytest.mark.asyncio


@pytest.fixture
def multiplex_secret_scope():
    previous = secret_scope.is_multiplex_active()
    outer_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(outer_token)
        secret_scope.set_multiplex_active(previous)


class _Environment:
    async def _ensure_initialized(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None


async def test_concurrent_profiles_use_only_their_scoped_provider_key(
    tmp_path,
    monkeypatch,
    multiplex_secret_scope,
):
    both_started = asyncio.Event()
    started = 0
    created: list[tuple[str, str, object]] = []
    load_dotenv = AsyncMock()

    class _Completions:
        async def create(self, **_kwargs):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="done", tool_calls=[])
                    )
                ]
            )

    class _Client:
        def __init__(self, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=_Completions())
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    def _make_client(*, base_url: str, api_key: str):
        client = _Client(base_url, api_key)
        created.append((str(get_hermes_home()), api_key, client))
        return client

    monkeypatch.setenv("OPENROUTER_API_KEY", "foreign-router-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "foreign-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai-key")
    monkeypatch.setattr(runner_module, "AsyncOpenAI", _make_client)
    monkeypatch.setattr(runner_module, "load_hermes_dotenv", load_dotenv)
    monkeypatch.setattr(
        runner_module,
        "create_environment",
        lambda **_kwargs: _Environment(),
    )

    async def _run_profile(label: str):
        home = tmp_path / label
        home.mkdir()
        home_token = set_hermes_home_override(home)
        scope_token = secret_scope.set_secret_scope(
            {"OPENROUTER_API_KEY": f"{label}-router-key"}
        )
        try:
            runner = MiniSWERunner(
                model="test/model",
                base_url="https://provider.example/v1",
                cwd=str(home),
            )
            result = await runner.run_task(f"task-{label}")
            return runner, result
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    async with no_task_leaks(action=LeakAction.RAISE):
        (runner_a, result_a), (runner_b, result_b) = await asyncio.gather(
            _run_profile("profile-a"),
            _run_profile("profile-b"),
        )

    assert {(home, key) for home, key, _client in created} == {
        (str(tmp_path / "profile-a"), "profile-a-router-key"),
        (str(tmp_path / "profile-b"), "profile-b-router-key"),
    }
    assert all(client.closed for _home, _key, client in created)
    assert runner_a.client is None
    assert runner_b.client is None
    assert [item["from"] for item in result_a["conversations"]] == [
        "system",
        "human",
        "gpt",
    ]
    assert [item["from"] for item in result_b["conversations"]] == [
        "system",
        "human",
        "gpt",
    ]
    load_dotenv.assert_not_awaited()


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        (
            {
                "OPENROUTER_API_KEY": "router-key",
                "ANTHROPIC_API_KEY": "anthropic-key",
                "OPENAI_API_KEY": "openai-key",
            },
            "router-key",
        ),
        (
            {
                "OPENROUTER_API_KEY": "",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
            "",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "anthropic-key",
                "OPENAI_API_KEY": "openai-key",
            },
            "anthropic-key",
        ),
        ({"OPENAI_API_KEY": "openai-key"}, "openai-key"),
    ],
)
async def test_explicit_endpoint_preserves_upstream_key_fallback_precedence(
    monkeypatch,
    multiplex_secret_scope,
    credentials,
    expected,
):
    captured: list[str] = []

    class _Client:
        async def close(self) -> None:
            return None

    def _make_client(*, base_url: str, api_key: str):
        assert base_url == "https://provider.example/v1"
        captured.append(api_key)
        return _Client()

    monkeypatch.setattr(runner_module, "AsyncOpenAI", _make_client)
    token = secret_scope.set_secret_scope(credentials)
    try:
        runner = MiniSWERunner(base_url="https://provider.example/v1")
        await runner._ensure_client()
        await runner._close_owned_client()
    finally:
        secret_scope.reset_secret_scope(token)

    assert captured == [expected]


async def test_single_profile_dotenv_load_uses_context_local_hermes_home(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    profile = tmp_path / "profile"
    project.mkdir()
    profile.mkdir()
    monkeypatch.chdir(project)
    load_dotenv = AsyncMock()

    class _Client:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(runner_module, "load_hermes_dotenv", load_dotenv)
    monkeypatch.setattr(runner_module, "AsyncOpenAI", lambda **_kwargs: _Client())
    home_token = set_hermes_home_override(profile)
    try:
        runner = MiniSWERunner(
            base_url="https://provider.example/v1",
            api_key="explicit-key",
        )
        await runner._ensure_client()
        await runner._close_owned_client()
    finally:
        reset_hermes_home_override(home_token)

    load_dotenv.assert_awaited_once_with(
        hermes_home=profile,
        project_env=project / ".env",
    )


@pytest.mark.parametrize("explicit_endpoint", [False, True])
async def test_unscoped_multiplex_provider_resolution_fails_closed(
    monkeypatch,
    multiplex_secret_scope,
    explicit_endpoint,
):
    load_dotenv = AsyncMock()
    resolver = AsyncMock(return_value=(None, None))
    openai = MagicMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "foreign-router-key")
    monkeypatch.setattr(runner_module, "load_hermes_dotenv", load_dotenv)
    monkeypatch.setattr(runner_module, "resolve_provider_client", resolver)
    monkeypatch.setattr(runner_module, "AsyncOpenAI", openai)
    runner = MiniSWERunner(
        base_url=("https://provider.example/v1" if explicit_endpoint else None)
    )

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="OPENROUTER_API_KEY",
    ):
        await runner._ensure_client()

    load_dotenv.assert_not_awaited()
    openai.assert_not_called()
    if explicit_endpoint:
        resolver.assert_not_awaited()
    else:
        assert resolver.await_count == 2


async def test_repeated_cancellation_finishes_owned_client_close(monkeypatch):
    provider_started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()
    close_calls = 0

    async def _stalled_create(**_kwargs):
        provider_started.set()
        await asyncio.Event().wait()

    class _Client:
        base_url = "https://provider.example/v1"
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=_stalled_create),
        )

        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            close_started.set()
            await release_close.wait()
            close_finished.set()

    monkeypatch.setattr(
        runner_module,
        "create_environment",
        lambda **_kwargs: _Environment(),
    )
    runner = MiniSWERunner(model="test/model")
    runner.client = _Client()
    runner._owns_client = True

    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(runner.run_task("cancel twice"))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        task.cancel()
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0)
        assert not close_finished.is_set()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert task.cancelling() >= 2
    assert close_calls == 1
    assert close_finished.is_set()
    assert runner.client is None
    assert runner.env is None
