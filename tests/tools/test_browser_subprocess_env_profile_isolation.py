from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from tools import browser_tool
from tools.environments import local


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(outer_token)
        secret_scope.set_multiplex_active(previous_multiplex)


async def _in_scope(secrets: dict[str, str]):
    token = secret_scope.set_secret_scope(secrets)
    try:
        await asyncio.sleep(0)
        return await browser_tool._build_browser_env()
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_browser_child_env_isolates_concurrent_profiles(monkeypatch):
    async def base_env(*, inherit_credentials: bool):
        assert inherit_credentials is False
        await asyncio.sleep(0)
        return {
            "PATH": "/safe/bin",
            **{
                key: f"foreign-base-{key.lower()}"
                for key in browser_tool._BROWSER_PASSTHROUGH_KEYS
            },
        }

    monkeypatch.setattr(local, "hermes_subprocess_env", base_env)
    for key in browser_tool._BROWSER_PASSTHROUGH_KEYS:
        monkeypatch.setenv(key, f"foreign-{key.lower()}")
    secret_scope.set_multiplex_active(True)

    profile_a, profile_b = await asyncio.gather(
        _in_scope(
            {
                "BROWSERBASE_API_KEY": "browser-a",
                "BROWSERBASE_PROJECT_ID": "project-a",
                "FIRECRAWL_API_URL": "https://firecrawl-a.example",
            }
        ),
        _in_scope(
            {
                "BROWSER_USE_API_KEY": "browser-b",
                "FIRECRAWL_API_KEY": "firecrawl-b",
                "FIRECRAWL_BROWSER_TTL": "120",
            }
        ),
    )

    assert profile_a == {
        "PATH": "/safe/bin",
        "BROWSERBASE_API_KEY": "browser-a",
        "BROWSERBASE_PROJECT_ID": "project-a",
        "FIRECRAWL_API_URL": "https://firecrawl-a.example",
    }
    assert profile_b == {
        "PATH": "/safe/bin",
        "BROWSER_USE_API_KEY": "browser-b",
        "FIRECRAWL_API_KEY": "firecrawl-b",
        "FIRECRAWL_BROWSER_TTL": "120",
    }


@pytest.mark.asyncio
async def test_browser_child_env_fails_closed_without_profile_scope(
    monkeypatch,
):
    async def base_env(*, inherit_credentials: bool):
        assert inherit_credentials is False
        return {}

    monkeypatch.setattr(local, "hermes_subprocess_env", base_env)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "foreign-browser-secret")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="BROWSERBASE_API_KEY",
    ):
        await browser_tool._build_browser_env()


@pytest.mark.asyncio
async def test_browser_child_env_preserves_single_profile_and_empty_values(
    monkeypatch,
):
    async def base_env(*, inherit_credentials: bool):
        assert inherit_credentials is False
        return {"PATH": "/safe/bin"}

    monkeypatch.setattr(local, "hermes_subprocess_env", base_env)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "single-profile-key")
    monkeypatch.setenv("FIRECRAWL_API_URL", "")

    assert await browser_tool._build_browser_env() == {
        "PATH": "/safe/bin",
        "BROWSERBASE_API_KEY": "single-profile-key",
        "FIRECRAWL_API_URL": "",
    }


@pytest.mark.asyncio
async def test_browser_child_env_propagates_cancellation(monkeypatch):
    entered = asyncio.Event()

    async def base_env(*, inherit_credentials: bool):
        assert inherit_credentials is False
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(local, "hermes_subprocess_env", base_env)
    task = asyncio.create_task(browser_tool._build_browser_env())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
