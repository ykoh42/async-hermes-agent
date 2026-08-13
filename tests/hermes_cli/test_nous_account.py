"""Tests for normalized Nous Portal account entitlement helpers."""

from __future__ import annotations

import base64
import asyncio
import gc
import json
import weakref
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hermes_cli.nous_account import (
    NousPaidServiceAccessInfo,
    NousPortalAccountInfo,
    format_nous_portal_entitlement_message,
    get_nous_portal_account_info,
    nous_portal_topup_url,
    reset_nous_portal_account_info_cache,
)


def _jwt(claims: dict[str, Any]) -> str:
    def _part(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _state(token: str) -> dict[str, Any]:
    return {
        "access_token": token,
        "portal_base_url": "https://portal.example.test",
        "client_id": "hermes-cli",
    }


def _account_payload(
    *,
    allowed: bool,
    subscription: dict[str, Any] | None,
    subscription_credits: float,
    purchased_credits: float,
) -> dict[str, Any]:
    return {
        "user": {
            "email": "alice@example.test",
            "privy_did": "did:privy:alice",
        },
        "organisation": {
            "id": "org_123",
        },
        "subscription": subscription,
        "purchased_credits_remaining": purchased_credits,
        "paid_service_access": {
            "allowed": allowed,
            "paid_access": allowed,
            "reason": "usable_credits" if allowed else "no_usable_credits",
            "organisation_id": "org_123",
            "effective_at_ms": 123456789,
            "has_active_subscription": subscription is not None,
            "active_subscription_is_paid": bool(
                subscription and subscription.get("monthly_charge", 0) > 0
            ),
            "subscription_tier": subscription.get("tier") if subscription else None,
            "subscription_monthly_charge": (
                subscription.get("monthly_charge") if subscription else None
            ),
            "subscription_credits_remaining": subscription_credits,
            "purchased_credits_remaining": purchased_credits,
            "total_usable_credits": subscription_credits + purchased_credits,
        },
    }


def test_account_info_cache_lock_is_loop_scoped():
    from hermes_cli.nous_account import _account_info_cache_lock

    first = asyncio.run(_return_value(_account_info_cache_lock))
    second = asyncio.run(_return_value(_account_info_cache_lock))

    assert first is not second


def test_contended_account_info_cache_lock_does_not_retain_closed_loop():
    from hermes_cli.nous_account import _account_info_cache_lock

    holder: dict[str, object] = {}

    async def contend() -> None:
        lock = _account_info_cache_lock()
        await lock.acquire()
        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0)
        lock.release()
        await waiter
        lock.release()
        holder["loop"] = weakref.ref(asyncio.get_running_loop())

    asyncio.run(contend())
    gc.collect()
    assert holder["loop"]() is None


async def _return_value(factory):
    return factory()


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_nous_portal_account_info_cache()
    yield
    reset_nous_portal_account_info_cache()






@pytest.mark.parametrize(
    ("payload", "expected_paid"),
    [
        (
            _account_payload(
                allowed=True,
                subscription={
                    "plan": "Tier 2",
                    "tier": 2,
                    "monthly_charge": 20,
                    "current_period_end": "2026-05-01T00:00:00.000Z",
                    "credits_remaining": 12.25,
                    "rollover_credits": 3.5,
                },
                subscription_credits=12.25,
                purchased_credits=7.75,
            ),
            True,
        ),
        (
            _account_payload(
                allowed=False,
                subscription={
                    "plan": "Tier 2",
                    "tier": 2,
                    "monthly_charge": 20,
                    "current_period_end": "2026-05-01T00:00:00.000Z",
                    "credits_remaining": 0,
                    "rollover_credits": 0,
                },
                subscription_credits=0,
                purchased_credits=0,
            ),
            False,
        ),
        (
            _account_payload(
                allowed=True,
                subscription=None,
                subscription_credits=0,
                purchased_credits=7.75,
            ),
            True,
        ),
        (
            _account_payload(
                allowed=False,
                subscription=None,
                subscription_credits=0,
                purchased_credits=0,
            ),
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_fresh_account_payload_normalization(monkeypatch, payload, expected_paid):
    token = _jwt({"sub": "user_123", "org_id": "org_123", "exp": int(time.time()) + 900})
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        AsyncMock(return_value=_state(token)),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token",
        AsyncMock(return_value="fresh-token"),
    )
    monkeypatch.setattr(
        "hermes_cli.nous_account._fetch_nous_account_info",
        AsyncMock(return_value=payload),
    )

    info = await get_nous_portal_account_info(force_fresh=True)

    assert isinstance(info, NousPortalAccountInfo)
    assert info.source == "account_api"
    assert info.fresh is True
    assert info.email == "alice@example.test"
    assert info.privy_did == "did:privy:alice"
    assert info.org_id == "org_123"
    assert info.paid_service_access is expected_paid
    assert info.is_paid is expected_paid
    assert info.is_free_tier is (not expected_paid)


@pytest.mark.asyncio
async def test_no_oauth_token_reports_inference_key_present(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        AsyncMock(return_value={}),
    )

    class _Entry:
        label = "manual-nous"
        access_token = ""
        agent_key = "opaque-runtime-key"
        agent_key_expires_at = "2099-01-01T00:00:00+00:00"
        expires_at = None
        inference_base_url = "https://inference.example.test/v1"
        base_url = "https://inference.example.test/v1"
        priority = 0

        @property
        def runtime_api_key(self):
            return self.agent_key

        @property
        def runtime_base_url(self):
            return self.inference_base_url

    class _Pool:
        def has_credentials(self):
            return True

        def entries(self):
            return [_Entry()]

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(return_value=_Pool()),
    )

    info = await get_nous_portal_account_info()

    assert info.logged_in is False
    assert info.source == "inference_key"
    assert info.inference_credential_present is True
    assert info.credential_source == "pool:manual-nous"
    assert info.paid_service_access is None


@pytest.mark.asyncio
async def test_pool_oauth_entry_force_fresh_uses_account_api(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "exp": int(time.time()) + 900,
            "paid_access": False,
        }
    )
    payload = _account_payload(
        allowed=True,
        subscription=None,
        subscription_credits=0,
        purchased_credits=3,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "hermes_cli.nous_account._fetch_nous_account_info",
        AsyncMock(return_value=payload),
    )

    class _Entry:
        label = "dashboard device_code"
        auth_type = "oauth"
        access_token = token
        refresh_token = "refresh-token"
        agent_key = "opaque-runtime-key"
        agent_key_expires_at = "2099-01-01T00:00:00+00:00"
        expires_at = "2099-01-01T00:00:00+00:00"
        portal_base_url = "https://portal.example.test"
        inference_base_url = "https://inference.example.test/v1"
        base_url = "https://inference.example.test/v1"
        priority = 0

        @property
        def runtime_api_key(self):
            return self.agent_key

        @property
        def runtime_base_url(self):
            return self.inference_base_url

    class _Pool:
        def has_credentials(self):
            return True

        def entries(self):
            return [_Entry()]

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(return_value=_Pool()),
    )

    info = await get_nous_portal_account_info(force_fresh=True)

    assert info.logged_in is True
    assert info.source == "account_api"
    assert info.fresh is True
    assert info.paid_service_access is True
    assert info.credential_source == "pool:dashboard device_code"


@pytest.mark.asyncio
async def test_jwt_free_tool_pool_entitlement_preserves_upstream_categories(
    monkeypatch,
):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "exp": int(time.time()) + 900,
            "paid_access": False,
            "tool_access": {
                "enabled": True,
                "coverage": {"fal": True, "fal-video": False},
            },
        }
    )
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        AsyncMock(return_value=_state(token)),
    )

    info = await get_nous_portal_account_info()

    assert info.paid_service_access is False
    assert info.tool_gateway_entitled is True
    assert info.tool_gateway_entitled_for("fal") is True
    assert info.tool_gateway_entitled_for("fal-video") is False
    assert (
        format_nous_portal_entitlement_message(
            info,
            capability="image generation",
            coverage_category="fal",
        )
        is None
    )
    uncovered = format_nous_portal_entitlement_message(
        info,
        capability="video generation",
        coverage_category="fal-video",
    )
    assert uncovered is not None
    assert "isn't included with your current Nous Portal access" in uncovered


@pytest.mark.asyncio
async def test_account_api_tool_access_parser_fails_closed(monkeypatch):
    token = _jwt({"sub": "user_123", "exp": int(time.time()) + 900})
    payload = _account_payload(
        allowed=False,
        subscription=None,
        subscription_credits=0,
        purchased_credits=0,
    )
    payload["tool_access"] = {
        "enabled": True,
        "coverage": {"fal": True, "fal-video": 1, 7: True},
    }
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        AsyncMock(return_value=_state(token)),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token",
        AsyncMock(return_value="fresh-token"),
    )
    monkeypatch.setattr(
        "hermes_cli.nous_account._fetch_nous_account_info",
        AsyncMock(return_value=payload),
    )

    info = await get_nous_portal_account_info(force_fresh=True)

    assert info.tool_access is not None
    assert info.tool_access.enabled is True
    assert info.tool_access.coverage == {"fal": True, "fal-video": False}






# ── org slug/name parsing + top-up URL builder ──────────────────────────────


