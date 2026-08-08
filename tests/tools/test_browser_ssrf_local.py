"""Tests that browser_navigate SSRF checks respect local-backend mode and
the allow_private_urls setting.

Local backends (Camofox, headless Chromium without a cloud provider) skip
SSRF checks entirely — the agent already has full local-network access via
the terminal tool.

Cloud backends (Browserbase, BrowserUse) enforce SSRF by default.  Users
can opt out for cloud mode via ``browser.allow_private_urls: true``.
"""

import json

import pytest
from unittest.mock import AsyncMock

from tools import browser_tool


pytestmark = pytest.mark.asyncio


def _make_browser_result(url="https://example.com"):
    """Return a mock successful browser command result."""
    return {"success": True, "data": {"title": "OK", "url": url}}


# ---------------------------------------------------------------------------
# Pre-navigation SSRF check
# ---------------------------------------------------------------------------


class TestPreNavigationSsrf:
    PRIVATE_URL = "http://127.0.0.1:8080/dashboard"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for pre-navigation tests that pass the SSRF check."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "check_website_access", AsyncMock(return_value=None))
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            AsyncMock(side_effect=lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            }),
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            AsyncMock(return_value=_make_browser_result()),
        )

    # -- Cloud mode: SSRF active -----------------------------------------------

    async def test_cloud_blocks_private_url_by_default(self, monkeypatch, _common_patches):
        """SSRF protection blocks private URLs in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=False))

        result = json.loads(await browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    async def test_cloud_allows_private_url_when_setting_true(self, monkeypatch, _common_patches):
        """Private URLs pass in cloud mode when allow_private_urls is True."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=True))
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=False))

        result = json.loads(await browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True


    # -- Local mode: SSRF skipped ----------------------------------------------

    async def test_local_allows_private_url(self, monkeypatch, _common_patches):
        """Local backends skip SSRF — private URLs are always allowed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=True))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=False))

        result = json.loads(await browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    async def test_local_allows_public_url(self, monkeypatch, _common_patches):
        """Local backends pass public URLs too (sanity check)."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=True))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=True))

        result = json.loads(await browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True

    # -- Always-blocked floor: hybrid routing bypass regression (#16234) -------

    # Hybrid-routing feature flips auto_local_this_nav=True for private URLs,
    # which previously short-circuited _is_safe_url() entirely. An agent
    # running on EC2/GCP/Azure could navigate to 169.254.169.254 via the
    # spawned local Chromium sidecar and read IAM credentials via
    # browser_snapshot. The always-blocked floor must fire regardless of
    # routing.
    IMDS_URLS = [
        "http://169.254.169.254/latest/meta-data/",      # AWS / GCP / Azure / DO / Oracle
        "http://169.254.169.253/metadata/instance",        # Azure IMDS wire server
        "http://169.254.170.2/v2/credentials",             # AWS ECS task metadata
        "http://100.100.100.200/latest/meta-data/",        # Alibaba Cloud
        "http://metadata.google.internal/computeMetadata/v1/",  # GCP hostname
    ]

    @pytest.mark.parametrize("imds_url", IMDS_URLS)
    async def test_cloud_blocks_imds_even_when_routing_to_local_sidecar(
        self, monkeypatch, _common_patches, imds_url
    ):
        """Hybrid routing must not let cloud metadata endpoints through."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        # Simulate hybrid routing kicking in for this URL (what happens on
        # main pre-fix — cloud provider configured, _url_is_private → True,
        # so the session key routes to a local Chromium sidecar).
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        # _is_safe_url would catch IMDS, but pre-fix it never ran. Force
        # it to return True here so the test is specifically pinning the
        # always-blocked floor as an independent gate.
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=True))

        result = json.loads(await browser_tool.browser_navigate(imds_url))

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]

    async def test_cloud_allows_ordinary_private_url_via_sidecar(
        self, monkeypatch, _common_patches
    ):
        """Hybrid routing still works for ordinary private URLs — floor
        must be narrow enough to not break the PR #16136 feature."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=False))

        for private in (
            "http://127.0.0.1:8080/dashboard",
            "http://192.168.1.1/admin",
            "http://10.0.0.5/",
            "http://myservice.local/",
        ):
            result = json.loads(await browser_tool.browser_navigate(private))
            assert result["success"] is True, f"Unexpected block for {private}: {result}"


# ---------------------------------------------------------------------------
# _is_local_backend() unit tests
# ---------------------------------------------------------------------------


class TestIsLocalBackend:
    async def test_camofox_is_local(self, monkeypatch):
        """Camofox mode counts as a local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", AsyncMock(return_value=True))
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", AsyncMock(return_value="anything"))

        assert await browser_tool._is_local_backend() is True

    async def test_no_cloud_provider_is_local(self, monkeypatch):
        """No cloud provider configured → local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", AsyncMock(return_value=None))

        assert await browser_tool._is_local_backend() is True


# ---------------------------------------------------------------------------
# Post-redirect SSRF check
# ---------------------------------------------------------------------------


class TestPostRedirectSsrf:
    PUBLIC_URL = "https://example.com/redirect"
    PRIVATE_FINAL_URL = "http://192.168.1.1/internal"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for redirect tests."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "check_website_access", AsyncMock(return_value=None))
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            AsyncMock(side_effect=lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            }),
        )

    # -- Cloud mode: redirect SSRF active --------------------------------------

    async def test_cloud_blocks_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses are blocked in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(
            browser_tool,
            "_is_safe_url",
            AsyncMock(side_effect=lambda url: "192.168" not in url),
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            AsyncMock(return_value=_make_browser_result(url=self.PRIVATE_FINAL_URL)),
        )

        result = json.loads(await browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]

    async def test_cloud_allows_redirect_to_private_when_setting_true(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in cloud mode with allow_private_urls."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=True))
        monkeypatch.setattr(
            browser_tool,
            "_is_safe_url",
            AsyncMock(side_effect=lambda url: "192.168" not in url),
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            AsyncMock(return_value=_make_browser_result(url=self.PRIVATE_FINAL_URL)),
        )

        result = json.loads(await browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    # -- Local mode: redirect SSRF skipped -------------------------------------


    async def test_cloud_allows_redirect_to_public(self, monkeypatch, _common_patches):
        """Redirects to public addresses always pass (cloud mode)."""
        final = "https://example.com/final"
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=True))
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            AsyncMock(return_value=_make_browser_result(url=final)),
        )

        result = json.loads(await browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == final

    # -- Always-blocked floor: redirect to IMDS via hybrid sidecar (#16234) ----

    async def test_cloud_blocks_redirect_to_imds_even_via_sidecar(
        self, monkeypatch, _common_patches
    ):
        """Redirect to a cloud metadata endpoint is blocked regardless of
        routing — even the hybrid local sidecar path can't return IMDS
        content to the agent."""
        imds_final = "http://169.254.169.254/latest/meta-data/"
        monkeypatch.setattr(browser_tool, "_is_local_backend", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_allow_private_urls", AsyncMock(return_value=False))
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        # _is_safe_url would catch it on main; force True to pin the
        # always-blocked floor as an independent gate.
        monkeypatch.setattr(browser_tool, "_is_safe_url", AsyncMock(return_value=True))
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            AsyncMock(return_value=_make_browser_result(url=imds_final)),
        )

        result = json.loads(await browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]


class TestAllowPrivateUrlsConfig:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        browser_tool._allow_private_urls_resolved = False
        browser_tool._cached_allow_private_urls = None
        yield
        browser_tool._allow_private_urls_resolved = False
        browser_tool._cached_allow_private_urls = None

    async def test_browser_config_string_false_stays_disabled(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(return_value={"browser": {"allow_private_urls": "false"}}),
        )

        assert await browser_tool._allow_private_urls() is False

    @pytest.mark.parametrize(
        "profile_order",
        [("allowed", "blocked"), ("blocked", "allowed")],
        ids=["allowed-then-blocked", "blocked-then-allowed"],
    )
    async def test_profile_scoped_config_does_not_reuse_another_profiles_opt_out(
        self, tmp_path, profile_order
    ):
        """The browser's independent guard must follow the active profile."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        allowed_home = tmp_path / "allowed"
        blocked_home = tmp_path / "blocked"
        allowed_home.mkdir()
        blocked_home.mkdir()
        (allowed_home / "config.yaml").write_text(
            "browser:\n  allow_private_urls: true\n", encoding="utf-8"
        )
        (blocked_home / "config.yaml").write_text(
            "browser:\n  allow_private_urls: false\n", encoding="utf-8"
        )

        async def under_profile(home):
            token = set_hermes_home_override(home)
            try:
                return await browser_tool._allow_private_urls()
            finally:
                reset_hermes_home_override(token)

        homes = {"allowed": allowed_home, "blocked": blocked_home}
        expected = {"allowed": True, "blocked": False}
        for profile in profile_order:
            assert await under_profile(homes[profile]) is expected[profile]
