"""Tests for Gemini free-tier detection and blocking."""
from __future__ import annotations

from agent.gemini_native_adapter import (
    gemini_http_error,
    is_free_tier_quota_error,
)
class TestIsFreeTierQuotaError:
    def test_detects_free_tier_marker(self):
        assert is_free_tier_quota_error(
            "Quota exceeded for metric: generate_content_free_tier_requests"
        )


    def test_no_free_tier_marker(self):
        assert not is_free_tier_quota_error("rate limited")


    def test_none(self):
        assert not is_free_tier_quota_error(None)  # type: ignore[arg-type]


class TestGeminiHttpErrorFreeTierGuidance:
    """gemini_http_error should append free-tier guidance for free-tier 429s."""

    class _FakeResp:
        def __init__(self, status: int, text: str):
            self.status_code = status
            self.headers: dict = {}
            self.text = text

    def test_free_tier_429_appends_guidance(self):
        body = (
            '{"error":{"code":429,"message":"Quota exceeded for metric: '
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            'limit: 20","status":"RESOURCE_EXHAUSTED"}}'
        )
        err = gemini_http_error(self._FakeResp(429, body))
        msg = str(err)
        assert "free tier" in msg.lower()
        assert "aistudio.google.com/apikey" in msg

    def test_paid_429_has_no_billing_url(self):
        body = '{"error":{"code":429,"message":"Rate limited","status":"RESOURCE_EXHAUSTED"}}'
        err = gemini_http_error(self._FakeResp(429, body))
        assert "aistudio.google.com/apikey" not in str(err)

