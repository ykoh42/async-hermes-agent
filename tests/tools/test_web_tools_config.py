"""Configuration and native-async dispatch contracts for web tools."""

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.web_tools as web_tools


@pytest.fixture(autouse=True)
def clean_web_registry(monkeypatch):
    for key in (
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TAVILY_API_KEY",
        "SEARXNG_URL",
        "BRAVE_SEARCH_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_configured_backend_is_preserved():
    with patch.object(web_tools, "_load_web_config", return_value={"backend": "parallel"}):
        assert web_tools._get_backend() == "parallel"


@pytest.mark.parametrize(
    ("env", "backend"),
    [
        ({"TAVILY_API_KEY": "key"}, "tavily"),
        ({"EXA_API_KEY": "key"}, "exa"),
        ({"PARALLEL_API_KEY": "key"}, "parallel"),
        ({"FIRECRAWL_API_KEY": "key"}, "firecrawl"),
    ],
)
def test_direct_credentials_select_backend(env, backend):
    with patch.object(web_tools, "_load_web_config", return_value={}), patch.dict(
        "os.environ", env
    ), patch.object(web_tools, "_ddgs_package_importable", return_value=False):
        assert web_tools._get_backend() == backend


def test_null_backend_config_is_safe():
    with patch.object(web_tools, "_load_web_config", return_value={"backend": None}), patch.object(
        web_tools, "_ddgs_package_importable", return_value=False
    ), patch(
        "agent.web_search_registry.get_active_search_provider", return_value=None
    ), patch("agent.web_search_registry.get_active_extract_provider", return_value=None):
        assert web_tools.check_web_api_key() is False


def test_search_schema_exposes_bounded_limit():
    limit = web_tools.WEB_SEARCH_SCHEMA["parameters"]["properties"]["limit"]
    assert limit["type"] == "integer"
    assert limit["default"] == 5
    assert limit["minimum"] == 1
    assert limit["maximum"] == 100


def test_public_web_functions_are_coroutines():
    assert inspect.iscoroutinefunction(web_tools.web_search_tool)
    assert inspect.iscoroutinefunction(web_tools.web_extract_tool)


@pytest.mark.asyncio
async def test_search_clamps_limit_before_async_provider_call(monkeypatch):
    provider = MagicMock()
    provider.name = "test"
    provider.supports_search.return_value = True
    provider.search = AsyncMock(
        return_value={"success": True, "data": {"web": []}}
    )
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "test")
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda _name: provider
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *_args: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(await web_tools.web_search_tool("docs", limit=500))

    assert result == {"success": True, "data": {"web": []}}
    provider.search.assert_awaited_once_with("docs", 100)


@pytest.mark.asyncio
async def test_sync_provider_fails_fast(monkeypatch):
    provider = MagicMock()
    provider.name = "sync-only"
    provider.supports_search.return_value = True
    provider.search = lambda *_args: {"success": True}
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "sync-only")
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda _name: provider
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *_args: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(await web_tools.web_search_tool("docs"))

    assert "native async" in result["error"]


@pytest.mark.asyncio
async def test_provider_error_does_not_leak_traceback(monkeypatch):
    provider = MagicMock()
    provider.name = "broken"
    provider.supports_search.return_value = True
    provider.search = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "broken")
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda _name: provider
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(web_tools._debug, "log_call", lambda *_args: None)
    monkeypatch.setattr(web_tools._debug, "save", lambda: None)

    result = json.loads(await web_tools.web_search_tool("docs"))

    assert result == {"error": "Error searching web: boom"}
    assert "traceback" not in result
