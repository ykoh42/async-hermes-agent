"""Tests for ``_get_cloud_provider()`` caching policy.

Regression coverage for issue #22324: a transient ``None`` from the resolver
must not be cached for the lifetime of the process. Cache only when:

* The user explicitly opts in to ``cloud_provider: local``, OR
* A provider is successfully resolved.

All other ``None`` outcomes (no credentials yet, config read error, explicit
provider instantiation failure) leave the cache unset so the next call retries.
"""
import logging
from unittest.mock import AsyncMock, Mock

import pytest

import tools.browser_tool as browser_tool


@pytest.fixture(autouse=True)
def _reset_resolver_state(monkeypatch):
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    yield


class TestCloudProviderCachePolicy:
    pytestmark = pytest.mark.asyncio

    async def test_explicit_local_caches_permanently(self, monkeypatch):
        """`cloud_provider: local` is a positive choice and must stick."""
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(return_value={"browser": {"cloud_provider": "local"}}),
        )

        assert await browser_tool._get_cloud_provider() is None
        assert browser_tool._cloud_provider_resolved is True

        # Even if config later changes, the cache stays.
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(
                return_value={"browser": {"cloud_provider": "browser-use"}}
            ),
        )
        assert await browser_tool._get_cloud_provider() is None


    async def test_no_credentials_yet_does_not_cache_none(self, monkeypatch):
        """Auto-detect path with no creds: must NOT poison the cache."""
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(return_value={"browser": {}}),
        )

        bu_unconfigured = Mock()
        bu_unconfigured.is_available = AsyncMock(return_value=False)
        bb_unconfigured = Mock()
        bb_unconfigured.is_available = AsyncMock(return_value=False)
        monkeypatch.setattr(
            browser_tool, "BrowserUseProvider", lambda: bu_unconfigured
        )
        monkeypatch.setattr(
            browser_tool, "BrowserbaseProvider", lambda: bb_unconfigured
        )

        assert await browser_tool._get_cloud_provider() is None
        assert browser_tool._cloud_provider_resolved is False

        # Credentials self-heal — next call must retry and pick up the provider.
        healed = Mock(name="healed-provider")
        healed.is_available = AsyncMock(return_value=True)
        monkeypatch.setattr(browser_tool, "BrowserUseProvider", lambda: healed)

        assert await browser_tool._get_cloud_provider() is healed
        assert browser_tool._cloud_provider_resolved is True


    async def test_explicit_provider_instantiation_failure_does_not_cache(
        self, monkeypatch, caplog
    ):
        """If `_PROVIDER_REGISTRY[key]()` raises, log warning and don't cache."""
        def exploding_factory():
            raise RuntimeError("missing dependency")

        monkeypatch.setattr(
            browser_tool, "_PROVIDER_REGISTRY", {"browser-use": exploding_factory}
        )
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(
                return_value={"browser": {"cloud_provider": "browser-use"}}
            ),
        )

        with caplog.at_level(logging.WARNING, logger="tools.browser_tool"):
            assert await browser_tool._get_cloud_provider() is None

        assert browser_tool._cloud_provider_resolved is False
        assert any(
            "browser-use" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
