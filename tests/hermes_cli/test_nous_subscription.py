"""Async parity tests for retained Nous subscription feature detection."""

from unittest.mock import AsyncMock

import pytest
from blockbuster import BlockBuster

from hermes_cli import nous_subscription as ns
from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo


_POOL_COVERAGE = {
    "firecrawl": True,
    "fal": True,
    "fal-video": False,
    "openai-audio": True,
    "browser-use": True,
    "modal": True,
}


def _account(*, logged_in: bool, paid: bool | None = None) -> NousPortalAccountInfo:
    return NousPortalAccountInfo(
        logged_in=logged_in,
        source="jwt" if logged_in else "none",
        fresh=False,
        paid_service_access=paid,
    )


def _pool_account() -> NousPortalAccountInfo:
    """A $0 subscriber with a live free tool pool (no paid access)."""
    return NousPortalAccountInfo(
        logged_in=True,
        source="jwt",
        fresh=False,
        paid_service_access=False,
        tool_access=NousToolAccessInfo(enabled=True, coverage=_POOL_COVERAGE),
    )


def _stub_common_async_boundaries(monkeypatch, *, account=None, env=None) -> None:
    env = env or {}
    account = account or _account(logged_in=False)
    monkeypatch.setattr(
        ns, "_get_env_value", AsyncMock(side_effect=lambda name: env.get(name, ""))
    )
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", AsyncMock(return_value=account)
    )
    monkeypatch.setattr(ns, "fal_key_is_configured", AsyncMock(return_value=False))
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", AsyncMock(return_value=""))
    monkeypatch.setattr(
        ns, "has_direct_modal_credentials", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        ns, "is_managed_tool_gateway_ready", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(ns, "_has_agent_browser", AsyncMock(return_value=False))
    monkeypatch.setattr(ns, "_local_browser_runnable", AsyncMock(return_value=False))


@pytest.mark.asyncio
async def test_get_nous_subscription_features_recognizes_direct_exa_backend(
    monkeypatch,
):
    _stub_common_async_boundaries(monkeypatch, env={"EXA_API_KEY": "exa-test"})
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")

    features = await ns.get_nous_subscription_features({"web": {"backend": "exa"}})

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.managed_by_nous is False
    assert features.web.direct_override is True
    assert features.web.current_provider == "exa"


@pytest.mark.asyncio
async def test_local_browser_unavailable_without_chromium(monkeypatch):
    """Feature state must match a local browser runtime without Chromium."""
    _stub_common_async_boundaries(monkeypatch)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", AsyncMock(return_value=True))
    monkeypatch.setattr(ns, "_local_browser_runnable", AsyncMock(return_value=False))

    features = await ns.get_nous_subscription_features({
        "browser": {"cloud_provider": "local"}
    })

    assert features.browser.available is False
    assert features.browser.active is False
    assert features.browser.managed_by_nous is False
    assert features.browser.current_provider == "Local browser"


@pytest.mark.asyncio
async def test_pool_entitlements_preserve_feature_and_item_order(monkeypatch):
    _stub_common_async_boundaries(monkeypatch, account=_pool_account())
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: True)
    gateway_ready = AsyncMock(return_value=True)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", gateway_ready)
    monkeypatch.setattr(ns, "_has_agent_browser", AsyncMock(return_value=True))
    monkeypatch.setattr(ns, "_local_browser_runnable", AsyncMock(return_value=True))

    features = await ns.get_nous_subscription_features({
        "web": {"backend": "firecrawl"},
        "image_gen": {"use_gateway": True},
        "video_gen": {"use_gateway": True},
        "tts": {"provider": "openai", "use_gateway": True},
        "stt": {"provider": "openai", "use_gateway": True},
        "browser": {"cloud_provider": "browser-use", "use_gateway": True},
        "terminal": {"backend": "modal", "modal_mode": "managed"},
    })

    assert [item.key for item in features.items()] == [
        "web",
        "image_gen",
        "video_gen",
        "tts",
        "stt",
        "browser",
        "modal",
    ]
    assert features.web.managed_by_nous is True
    assert features.image_gen.managed_by_nous is True
    assert features.video_gen.available is False
    assert features.tts.managed_by_nous is True
    assert features.stt.managed_by_nous is True
    assert features.browser.managed_by_nous is True
    assert features.modal.managed_by_nous is True
    assert [call.args[0] for call in gateway_ready.await_args_list] == [
        "firecrawl",
        "fal-queue",
        "fal-queue",
        "openai-audio",
        "browser-use",
        "modal",
    ]


@pytest.mark.asyncio
async def test_force_fresh_is_forwarded_only_when_requested(monkeypatch):
    _stub_common_async_boundaries(monkeypatch)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: False)
    account_reader = AsyncMock(return_value=_account(logged_in=False))
    monkeypatch.setattr(ns, "get_nous_portal_account_info", account_reader)

    await ns.get_nous_subscription_features({}, force_fresh=True)

    account_reader.assert_awaited_once_with(force_fresh=True)


@pytest.mark.asyncio
async def test_default_config_and_feature_probes_do_not_block(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        ns,
        "get_nous_portal_account_info",
        AsyncMock(return_value=_account(logged_in=False)),
    )
    blocker = BlockBuster()
    blocker.activate()
    try:
        features = await ns.get_nous_subscription_features()
    finally:
        blocker.deactivate()

    assert [item.key for item in features.items()] == [
        "web",
        "image_gen",
        "video_gen",
        "tts",
        "stt",
        "browser",
        "modal",
    ]
