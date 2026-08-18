"""Tests for the X (Twitter) Search tool backed by xAI Responses API.

Covers:
- HTTP request shape (URL, headers, payload, model from config)
- Handle filter validation (allowed vs excluded mutual exclusion)
- Inline url_citation extraction from message annotations
- Structured error handling
- Credential resolution: API key path, OAuth path, and none-set
- check_x_search_requirements gating in registry
"""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest


def _response(url, payload, *, status_code=200):
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", url),
    )


def _install_post(monkeypatch, post):
    class _FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, headers=None, json=None, timeout=None):
            return post(url, headers=headers, json=json, timeout=timeout)

    monkeypatch.setattr("tools.x_search_tool.httpx.AsyncClient", _FakeAsyncClient)


def _install_credentials(monkeypatch, *, provider="xai", api_key="xai-test-key"):
    async def _fake_resolve():
        return {
            "provider": provider,
            "api_key": api_key,
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _fake_resolve
    )


# ---------------------------------------------------------------------------
# Original PR #10786 test coverage (HTTP shape, handle validation, citations,
# and error behavior), adapted only for the native-async transport.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_search_posts_responses_request(monkeypatch):
    from hermes_cli import __version__
    from tools.x_search_tool import x_search_tool

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _response(
            url,
            {
                "output_text": "People on X are discussing xAI's latest launch.",
                "citations": [
                    {
                        "url": "https://x.com/example/status/1",
                        "title": "Example post",
                    }
                ],
            },
        )

    _install_credentials(monkeypatch)
    _install_post(monkeypatch, _fake_post)

    result = json.loads(
        await x_search_tool(
            query="What are people saying about xAI on X?",
            allowed_x_handles=["xai", "@grok"],
            from_date="2026-04-01",
            to_date="2026-04-10",
            enable_image_understanding=True,
        )
    )

    tool_def = captured["json"]["tools"][0]
    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["User-Agent"] == f"Hermes-Agent/{__version__}"
    assert captured["json"]["model"] == "grok-4.5"
    assert captured["json"]["store"] is False
    assert "reasoning" not in captured["json"]
    assert tool_def["type"] == "x_search"
    assert tool_def["allowed_x_handles"] == ["xai", "grok"]
    assert tool_def["from_date"] == "2026-04-01"
    assert tool_def["to_date"] == "2026-04-10"
    assert tool_def["enable_image_understanding"] is True
    assert result["success"] is True
    assert result["answer"] == "People on X are discussing xAI's latest launch."


@pytest.mark.asyncio
async def test_x_search_rejects_conflicting_handle_filters(monkeypatch):
    from tools.x_search_tool import x_search_tool

    _install_credentials(monkeypatch)

    result = json.loads(
        await x_search_tool(
            query="latest xAI discussion",
            allowed_x_handles=["xai"],
            excluded_x_handles=["grok"],
        )
    )

    assert (
        result["error"]
        == "allowed_x_handles and excluded_x_handles cannot be used together"
    )


def test_x_search_schema_is_read_only_without_cross_tool_names():
    """Static schema must state read-only scope without naming other surfaces.

    AGENTS.md forbids hardcoding cross-tool/skill names in tool schemas because
    those surfaces may be unavailable. Keep out-of-scope guidance generic here;
    xurl routing lives in the skill and feature docs.
    """
    from tools.x_search_tool import X_SEARCH_SCHEMA

    description = X_SEARCH_SCHEMA["description"]
    lowered = description.lower()

    assert "read-only" in lowered
    assert "public x" in lowered
    for action in ("post", "reply", "like", "dm", "upload media", "delete"):
        assert action in lowered
    assert "authenticated" in lowered
    assert "xurl" not in lowered
    assert "web_search" not in lowered


@pytest.mark.asyncio
async def test_x_search_extracts_inline_url_citations(monkeypatch):
    from tools.x_search_tool import x_search_tool

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _response(
            url,
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "xAI posted an update on X.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://x.com/xai/status/123",
                                        "title": "xAI update",
                                        "start_index": 0,
                                        "end_index": 3,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        )

    _install_credentials(monkeypatch)
    _install_post(monkeypatch, _fake_post)

    result = json.loads(await x_search_tool(query="latest post from xai"))

    assert result["success"] is True
    assert result["answer"] == "xAI posted an update on X."
    assert result["inline_citations"] == [
        {
            "url": "https://x.com/xai/status/123",
            "title": "xAI update",
            "start_index": 0,
            "end_index": 3,
        }
    ]


@pytest.mark.asyncio
async def test_x_search_returns_structured_http_error(monkeypatch):
    from tools.x_search_tool import x_search_tool

    def _fake_post(url, **_kwargs):
        return _response(
            url,
            {
                "code": "forbidden",
                "error": "x_search is not enabled for this model",
            },
            status_code=403,
        )

    _install_credentials(monkeypatch)
    _install_post(monkeypatch, _fake_post)

    result = json.loads(await x_search_tool(query="latest xai discussion"))

    assert result["success"] is False
    assert result["provider"] == "xai"
    assert result["tool"] == "x_search"
    assert result["error_type"] == "HTTPError"
    assert result["error"] == "forbidden: x_search is not enabled for this model"


# ---------------------------------------------------------------------------
# Credential-resolution coverage — the OAuth-or-API-key gating contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_search_uses_xai_oauth_when_only_oauth_available(monkeypatch):
    """OAuth-only user: credential_source should be ``xai-oauth``."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    _install_credentials(
        monkeypatch, provider="xai-oauth", api_key="oauth-bearer-token"
    )
    invalidate_check_fn_cache()

    assert await check_x_search_requirements() is True

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _response(url, {"output_text": "Found posts via OAuth."})

    _install_post(monkeypatch, _fake_post)

    result = json.loads(await x_search_tool(query="anything about xai"))

    assert result["success"] is True
    assert result["credential_source"] == "xai-oauth"
    assert captured["headers"]["Authorization"] == "Bearer oauth-bearer-token"


@pytest.mark.asyncio
async def test_x_search_returns_tool_error_when_no_credentials(monkeypatch):
    """No credentials anywhere: tool returns a clear error, not a 401 from xAI."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    _install_credentials(monkeypatch, api_key="")
    invalidate_check_fn_cache()

    assert await check_x_search_requirements() is False

    result = await x_search_tool(query="anything")
    assert "No xAI credentials available" in result
    assert "hermes auth add xai-oauth" in result


# ---------------------------------------------------------------------------
# Degraded-result flag — distinguish citation-backed answers from
# unsourced fluff when narrowing filters returned nothing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_search_not_degraded_when_no_filters_active(monkeypatch):
    """A broad query that returns no citations isn't necessarily degraded."""
    from tools.x_search_tool import x_search_tool

    _install_credentials(monkeypatch)
    _install_post(
        monkeypatch,
        lambda url, **_kwargs: _response(
            url, {"output_text": "broad answer", "citations": []}
        ),
    )

    result = json.loads(await x_search_tool(query="anything"))

    assert result["success"] is True
    assert result["degraded"] is False
    assert result["degraded_reason"] is None


# ---------------------------------------------------------------------------
# Native-async boundary coverage for the otherwise parity-preserving port.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_search_awaits_config_and_preserves_configured_request(monkeypatch):
    from hermes_cli import config as config_module
    from tools.x_search_tool import x_search_tool

    load_config = AsyncMock(
        return_value={
            "x_search": {
                "model": "grok-configured",
                "reasoning_effort": "high",
                "timeout_seconds": 73,
                "retries": 0,
            }
        }
    )
    monkeypatch.setattr(config_module, "load_config_readonly", load_config)
    _install_credentials(monkeypatch)
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        captured["timeout"] = timeout
        return _response(url, {"output_text": "configured answer"})

    _install_post(monkeypatch, _fake_post)

    result = json.loads(await x_search_tool(query="configured search"))

    assert result["success"] is True
    assert captured["json"]["model"] == "grok-configured"
    assert captured["json"]["reasoning"] == {"effort": "high"}
    assert captured["timeout"] == 73
    assert load_config.await_count == 4


@pytest.mark.asyncio
async def test_x_search_retries_5xx_with_async_sleep(monkeypatch):
    from tools import x_search_tool as module

    _install_credentials(monkeypatch)
    monkeypatch.setattr(module, "_get_x_search_retries", AsyncMock(return_value=1))
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    attempts = 0

    def _fake_post(url, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _response(url, {"error": "temporary"}, status_code=503)
        return _response(url, {"output_text": "recovered"})

    _install_post(monkeypatch, _fake_post)

    result = json.loads(await module.x_search_tool(query="retry search"))

    assert result["success"] is True
    assert result["answer"] == "recovered"
    assert attempts == 2
    sleep.assert_awaited_once_with(1.5)


@pytest.mark.asyncio
async def test_x_search_cancellation_closes_owned_http_client(monkeypatch):
    from tools import x_search_tool as module

    _install_credentials(monkeypatch)
    entered = asyncio.Event()
    closed = asyncio.Event()

    class _BlockingAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            closed.set()
            return False

        async def post(self, url, headers=None, json=None, timeout=None):
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(module.httpx, "AsyncClient", _BlockingAsyncClient)
    task = asyncio.create_task(module.x_search_tool(query="cancel search"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()


@pytest.mark.asyncio
async def test_x_search_prefers_explicit_api_key_over_oauth(monkeypatch):
    from tools.x_search_tool import _resolve_xai_bearer

    paid_key = "paid-key-x1"
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        AsyncMock(
            side_effect=lambda name, default=None: {
                "XAI_API_KEY": paid_key,
                "XAI_BASE_URL": None,
            }.get(name, default)
        ),
    )
    oauth_token = "oauth-key-x1"
    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials",
        AsyncMock(
            return_value={
                "provider": "xai-oauth",
                "api_key": oauth_token,
                "base_url": "https://api.x.ai/v1",
            }
        ),
    )

    assert await _resolve_xai_bearer() == (
        paid_key,
        "https://api.x.ai/v1",
        "xai",
    )


@pytest.mark.asyncio
async def test_x_search_bearer_falls_back_to_oauth_without_api_key(monkeypatch):
    from tools.x_search_tool import _resolve_xai_bearer

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        AsyncMock(return_value=None),
    )
    oauth_token = "oauth-key-x1"
    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials",
        AsyncMock(
            return_value={
                "provider": "xai-oauth",
                "api_key": oauth_token,
                "base_url": "https://api.x.ai/v1",
            }
        ),
    )

    assert await _resolve_xai_bearer() == (
        oauth_token,
        "https://api.x.ai/v1",
        "xai-oauth",
    )
