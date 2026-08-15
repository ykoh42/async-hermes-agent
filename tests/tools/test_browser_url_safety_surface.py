"""Retained browser URL-safety helper contract."""

import pytest

from tools.browser_tool import evaluate_url_safety


@pytest.mark.asyncio
async def test_evaluate_url_safety_blocks_embedded_provider_token():
    result = await evaluate_url_safety(
        "https://example.test/?token=sk-" + "a" * 24
    )
    assert result is not None
    assert result["success"] is False
    assert "API key or token" in result["error"]


@pytest.mark.asyncio
async def test_evaluate_url_safety_allows_normal_url(monkeypatch):
    async def local_backend():
        return True

    async def never_blocked(_url):
        return False

    async def safe_url(_url):
        return True

    async def policy(_url):
        return None

    monkeypatch.setattr("tools.browser_tool._is_local_backend", local_backend)
    monkeypatch.setattr("tools.browser_tool._is_always_blocked_url", never_blocked)
    monkeypatch.setattr("tools.browser_tool._is_safe_url", safe_url)
    monkeypatch.setattr("tools.browser_tool.check_website_access", policy)

    assert await evaluate_url_safety("https://example.test/path") is None
