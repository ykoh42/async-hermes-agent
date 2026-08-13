"""Profile isolation for browser-provider endpoint and feature settings."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from plugins.browser.browserbase import provider as browserbase
from plugins.browser.firecrawl import provider as firecrawl


pytestmark = pytest.mark.asyncio


async def test_concurrent_browserbase_profiles_use_their_own_full_configuration(
    monkeypatch,
):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "process-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "process-project")
    monkeypatch.setenv("BROWSERBASE_BASE_URL", "https://process.invalid")
    monkeypatch.setenv("BROWSERBASE_SESSION_TIMEOUT", "999")
    requests: list[tuple[str, str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(
            (
                request.url.host or "",
                request.headers["X-BB-API-Key"],
                payload["projectId"],
                payload,
            )
        )
        return httpx.Response(
            201,
            json={"id": request.url.host, "connectUrl": "wss://browser.invalid"},
            request=request,
        )

    async def create_client(**_kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(browserbase, "_create_httpx_client", create_client)
    provider = browserbase.BrowserbaseBrowserProvider()
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def create(name: str):
        token = set_secret_scope(
            {
                "BROWSERBASE_API_KEY": f"key-{name}",
                "BROWSERBASE_PROJECT_ID": f"project-{name}",
                "BROWSERBASE_BASE_URL": f"https://{name}.example/",
                "BROWSERBASE_PROXIES": "true" if name == "alpha" else "false",
                "BROWSERBASE_ADVANCED_STEALTH": (
                    "false" if name == "alpha" else "true"
                ),
                "BROWSERBASE_KEEP_ALIVE": "false",
                "BROWSERBASE_SESSION_TIMEOUT": "111" if name == "alpha" else "222",
            }
        )
        try:
            return await provider.create_session(name)
        finally:
            reset_secret_scope(token)

    try:
        results = await asyncio.gather(create("alpha"), create("beta"))
    finally:
        set_multiplex_active(previous_multiplex)

    by_host = {host: (key, project, payload) for host, key, project, payload in requests}
    assert by_host["alpha.example"] == (
        "key-alpha",
        "project-alpha",
        {"projectId": "project-alpha", "timeout": 111, "proxies": True},
    )
    assert by_host["beta.example"] == (
        "key-beta",
        "project-beta",
        {
            "projectId": "project-beta",
            "timeout": 222,
            "browserSettings": {"advancedStealth": True},
        },
    )
    assert {result["bb_session_id"] for result in results} == {
        "alpha.example",
        "beta.example",
    }
    features_by_id = {
        result["bb_session_id"]: result["features"] for result in results
    }
    assert features_by_id == {
        "alpha.example": {
            "basic_stealth": True,
            "proxies": True,
            "advanced_stealth": False,
            "keep_alive": False,
            "custom_timeout": True,
        },
        "beta.example": {
            "basic_stealth": True,
            "proxies": False,
            "advanced_stealth": True,
            "keep_alive": False,
            "custom_timeout": True,
        },
    }


async def test_concurrent_firecrawl_profiles_use_their_own_url_key_and_ttl(
    monkeypatch,
):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "process-key")
    monkeypatch.setenv("FIRECRAWL_API_URL", "https://process.invalid")
    monkeypatch.setenv("FIRECRAWL_BROWSER_TTL", "999")
    requests: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(
            (
                request.url.host or "",
                request.headers["Authorization"],
                payload,
            )
        )
        return httpx.Response(
            201,
            json={"id": request.url.host, "cdpUrl": "wss://browser.invalid"},
            request=request,
        )

    async def create_client(**_kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(firecrawl, "_create_httpx_client", create_client)
    provider = firecrawl.FirecrawlBrowserProvider()
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def create(name: str, ttl: int):
        token = set_secret_scope(
            {
                "FIRECRAWL_API_KEY": f"key-{name}",
                "FIRECRAWL_API_URL": f"https://{name}.example",
                "FIRECRAWL_BROWSER_TTL": str(ttl),
            }
        )
        try:
            return await provider.create_session(name)
        finally:
            reset_secret_scope(token)

    try:
        results = await asyncio.gather(create("alpha", 111), create("beta", 222))
    finally:
        set_multiplex_active(previous_multiplex)

    by_host = {host: (auth, payload) for host, auth, payload in requests}
    assert by_host == {
        "alpha.example": ("Bearer key-alpha", {"ttl": 111}),
        "beta.example": ("Bearer key-beta", {"ttl": 222}),
    }
    assert {result["bb_session_id"] for result in results} == {
        "alpha.example",
        "beta.example",
    }
    assert all(result["features"] == {"firecrawl": True} for result in results)


async def test_single_profile_empty_values_preserve_upstream_precedence(
    monkeypatch,
):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "project")
    monkeypatch.setenv("BROWSERBASE_BASE_URL", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "key")
    monkeypatch.setenv("FIRECRAWL_API_URL", "")

    assert browserbase.BrowserbaseBrowserProvider()._get_config() == {
        "api_key": "key",
        "project_id": "project",
        "base_url": "",
    }
    assert firecrawl.FirecrawlBrowserProvider()._api_url() == ""


@pytest.mark.parametrize(
    "provider",
    [
        browserbase.BrowserbaseBrowserProvider(),
        firecrawl.FirecrawlBrowserProvider(),
    ],
)
async def test_missing_multiplex_scope_fails_closed_before_env_fallback(
    provider,
    monkeypatch,
):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "process-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "process-project")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "process-key")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError):
            await provider.is_available()
    finally:
        set_multiplex_active(previous_multiplex)


@pytest.mark.parametrize(
    ("provider_module", "provider", "credentials"),
    [
        (
            browserbase,
            browserbase.BrowserbaseBrowserProvider(),
            {
                "BROWSERBASE_API_KEY": "key",
                "BROWSERBASE_PROJECT_ID": "project",
            },
        ),
        (
            firecrawl,
            firecrawl.FirecrawlBrowserProvider(),
            {"FIRECRAWL_API_KEY": "key"},
        ),
    ],
)
async def test_create_cancellation_closes_owned_http_client(
    provider_module,
    provider,
    credentials,
    monkeypatch,
):
    entered = asyncio.Event()
    exited = asyncio.Event()

    class SlowClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc_info):
            exited.set()

        async def post(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    async def create_client(**_kwargs):
        return SlowClient()

    monkeypatch.setattr(provider_module, "_create_httpx_client", create_client)
    token = set_secret_scope(credentials)
    try:
        task = asyncio.create_task(provider.create_session("cancelled"))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        reset_secret_scope(token)

    assert exited.is_set()
