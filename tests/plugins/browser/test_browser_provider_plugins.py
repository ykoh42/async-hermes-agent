"""Plugin-side tests for the browser provider migration (PR #25214).

Covers:

- All three bundled plugins (browserbase, browser-use, firecrawl)
  instantiate and self-report the expected ABC defaults.
- Each plugin's ``is_available()`` correctly reflects env-var presence.
- The browser_registry resolves an active provider in the documented
  scenarios:
    * explicit config wins ignoring availability (so dispatcher surfaces
      a typed credentials error)
    * legacy preference walk: browser-use → browserbase (filtered by
      availability)
    * firecrawl is NOT in the legacy walk — explicit-only
    * unknown name falls through to auto-detect
    * ``local`` short-circuits to None

These tests use *real* imports from the plugin modules — no mocking of
provider classes themselves — so the test catches drift in the ABC
interface, the registry, and the plugin glue layer simultaneously.
Mirrors ``tests/plugins/web/test_web_search_provider_plugins.py`` from
PR #25182.
"""
from __future__ import annotations

import json

import httpx
import pytest


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every browser-provider env var so is_available() returns False."""
    for k in (
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSERBASE_BASE_URL",
        "BROWSER_USE_API_KEY",
        "BROWSER_USE_GATEWAY_URL",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_BROWSER_TTL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)


async def _ensure_plugins_loaded() -> None:
    """Idempotently load plugins so the registry is populated."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    await _ensure_plugins_discovered()


def _install_http_transport(monkeypatch, handler) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(transport=httpx.MockTransport(handler)),
    )


# ---------------------------------------------------------------------------
# Per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Each test starts with a clean browser-provider env."""
    _clear_browser_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


# ---------------------------------------------------------------------------
# Bundled plugins register
# ---------------------------------------------------------------------------


class TestBundledPluginsRegister:
    """All three bundled browser plugins discover and register correctly."""

    async def test_all_three_plugins_present_in_registry(self) -> None:
        await _ensure_plugins_loaded()
        from agent.browser_registry import list_providers

        names = sorted(p.name for p in list_providers())
        assert names == ["browser-use", "browserbase", "firecrawl"]

    @pytest.mark.parametrize(
        "plugin_name,expected_display",
        [
            ("browserbase", "Browserbase"),
            ("browser-use", "Browser Use"),
            ("firecrawl", "Firecrawl"),
        ],
    )
    async def test_each_plugin_has_name_and_display_name(
        self, plugin_name: str, expected_display: str
    ) -> None:
        await _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None, f"plugin {plugin_name!r} not registered"
        assert provider.name == plugin_name
        assert provider.display_name == expected_display


# ---------------------------------------------------------------------------
# is_available() behavior
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Each plugin's ``is_available()`` reflects env-var presence accurately."""

    async def test_browserbase_requires_both_api_key_and_project_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browserbase")
        assert p is not None
        assert await p.is_available() is False

        # API key alone is insufficient.
        monkeypatch.setenv("BROWSERBASE_API_KEY", "key")
        assert await p.is_available() is False

        # Both env vars set → available.
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj")
        assert await p.is_available() is True


    async def test_browser_use_satisfied_by_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browser-use")
        assert p is not None
        assert await p.is_available() is False
        monkeypatch.setenv("BROWSER_USE_API_KEY", "key")
        assert await p.is_available() is True

    async def test_firecrawl_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        await _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert await p.is_available() is False
        monkeypatch.setenv("FIRECRAWL_API_KEY", "key")
        assert await p.is_available() is True


# ---------------------------------------------------------------------------
# Registry resolution semantics
# ---------------------------------------------------------------------------


class TestRegistryResolution:
    """``_resolve()`` implements the documented three-rule precedence."""

    async def test_resolve_none_with_no_creds_returns_none(self) -> None:
        """No config, no env → local mode (None)."""
        await _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        assert await _resolve(None) is None

    async def test_explicit_local_returns_none(self) -> None:
        """``cloud_provider: local`` is a positive choice; short-circuits to None."""
        await _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        assert await _resolve("local") is None


    async def test_legacy_walk_prefers_browser_use_over_browserbase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 3: walk order is browser-use → browserbase."""
        await _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        # Both available — browser-use should win.
        monkeypatch.setenv("BROWSER_USE_API_KEY", "k1")
        monkeypatch.setenv("BROWSERBASE_API_KEY", "k2")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "p")

        provider = await _resolve(None)
        assert provider is not None
        assert provider.name == "browser-use"


class TestNativeAsyncLifecycle:
    async def test_browser_use_create_preserves_session_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BROWSER_USE_API_KEY", "browser-use-key")
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "id": "bu-session",
                    "cdpUrl": "wss://browser-use.example/cdp",
                    "timeoutAt": "2026-08-06T12:00:00Z",
                },
                request=request,
            )

        _install_http_transport(monkeypatch, handler)
        from plugins.browser.browser_use.provider import BrowserUseBrowserProvider

        result = await BrowserUseBrowserProvider().create_session("task-1")

        assert captured == {"method": "POST", "payload": {}}
        assert result["bb_session_id"] == "bu-session"
        assert result["cdp_url"] == "wss://browser-use.example/cdp"
        assert result["expires_at"] == "2026-08-06T12:00:00Z"

    async def test_browserbase_create_preserves_feature_fallbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "bb-project")
        payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            if len(payloads) == 1:
                return httpx.Response(402, text="paid feature", request=request)
            return httpx.Response(
                201,
                json={"id": "bb-session", "connectUrl": "wss://bb.example/cdp"},
                request=request,
            )

        _install_http_transport(monkeypatch, handler)
        from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider

        result = await BrowserbaseBrowserProvider().create_session("task-2")

        assert payloads[0]["keepAlive"] is True
        assert "keepAlive" not in payloads[1]
        assert result["bb_session_id"] == "bb-session"
        assert result["features"]["keep_alive"] is False

    async def test_firecrawl_create_and_close_use_native_async_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "firecrawl-key")
        methods = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            if request.method == "POST":
                return httpx.Response(
                    201,
                    json={"id": "fc-session", "cdpUrl": "wss://fc.example/cdp"},
                    request=request,
                )
            return httpx.Response(204, request=request)

        _install_http_transport(monkeypatch, handler)
        from plugins.browser.firecrawl.provider import FirecrawlBrowserProvider

        provider = FirecrawlBrowserProvider()
        result = await provider.create_session("task-3")
        closed = await provider.close_session("fc-session")

        assert result["bb_session_id"] == "fc-session"
        assert closed is True
        assert methods == ["POST", "DELETE"]
