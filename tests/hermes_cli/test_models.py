"""Tests for retained runtime behavior in :mod:`hermes_cli.models`."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import hermes_cli.models as models
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.nous_account import NousPortalAccountInfo


def test_curated_openrouter_and_nous_catalogs_track_upstream_flash_model():
    """The upstream 2026.8.13 catalog replaces the retired 3.6 flash route."""
    assert ("google/gemini-3.7-flash", "") in models.OPENROUTER_MODELS
    assert ("google/gemini-3.6-flash", "") not in models.OPENROUTER_MODELS
    assert "google/gemini-3.7-flash" in models._PROVIDER_MODELS["nous"]
    assert "google/gemini-3.6-flash" not in models._PROVIDER_MODELS["nous"]


def test_curated_openrouter_and_nous_catalogs_track_upstream_qwen_model():
    """The upstream catalog replaces the retired Qwen 3.7 Max route."""
    assert ("qwen/qwen3.8-max", "") in models.OPENROUTER_MODELS
    assert ("qwen/qwen3.7-max", "") not in models.OPENROUTER_MODELS
    assert "qwen/qwen3.8-max" in models._PROVIDER_MODELS["nous"]
    assert "qwen/qwen3.7-max" not in models._PROVIDER_MODELS["nous"]


def test_opencode_go_gpt_models_use_responses_api():
    assert models.opencode_model_api_mode("opencode-go", "gpt-5.6-luna") == "codex_responses"
    assert models.opencode_model_api_mode(
        "opencode-go", "opencode-go/gpt-5.6-luna"
    ) == "codex_responses"


@pytest.fixture(autouse=True)
def _clear_nous_model_caches():
    models._free_tier_cache = None
    models._free_tier_cache_by_profile.clear()
    models._nous_recommended_cache.clear()
    yield
    models._free_tier_cache = None
    models._free_tier_cache_by_profile.clear()
    models._nous_recommended_cache.clear()


@pytest.mark.asyncio
async def test_check_nous_free_tier_caches_result():
    account = NousPortalAccountInfo(
        logged_in=True,
        source="jwt",
        fresh=False,
        paid_service_access=False,
    )
    with patch(
        "hermes_cli.nous_account.get_nous_portal_account_info",
        new=AsyncMock(return_value=account),
    ) as get_account_info:
        assert await models.check_nous_free_tier() is True
        assert await models.check_nous_free_tier() is True

    get_account_info.assert_awaited_once_with(force_fresh=False)


@pytest.mark.asyncio
async def test_check_nous_free_tier_force_fresh_bypasses_cache():
    account = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        paid_service_access=True,
    )
    with patch(
        "hermes_cli.nous_account.get_nous_portal_account_info",
        new=AsyncMock(return_value=account),
    ) as get_account_info:
        assert await models.check_nous_free_tier() is False
        assert await models.check_nous_free_tier(force_fresh=True) is False

    assert get_account_info.await_count == 2
    get_account_info.assert_awaited_with(force_fresh=True)


@pytest.mark.asyncio
async def test_check_nous_free_tier_isolates_concurrent_profiles(
    tmp_path,
):
    both_started = asyncio.Event()
    starts = 0
    calls: dict[str, int] = {"profile-a": 0, "profile-b": 0}

    async def get_account_info(*, force_fresh=False):
        nonlocal starts
        assert force_fresh is False
        profile = get_hermes_home().name
        calls[profile] += 1
        starts += 1
        if starts == 2:
            both_started.set()
        await both_started.wait()
        return NousPortalAccountInfo(
            logged_in=True,
            source="jwt",
            fresh=False,
            paid_service_access=profile == "profile-b",
        )

    async def check(profile: str) -> tuple[bool, bool]:
        token = set_hermes_home_override(tmp_path / profile)
        try:
            first = await models.check_nous_free_tier()
            second = await models.check_nous_free_tier()
            return first, second
        finally:
            reset_hermes_home_override(token)

    with patch(
        "hermes_cli.nous_account.get_nous_portal_account_info",
        new=get_account_info,
    ):
        free_result, paid_result = await asyncio.gather(
            check("profile-a"),
            check("profile-b"),
        )

    assert free_result == (True, True)
    assert paid_result == (False, False)
    assert calls == {"profile-a": 1, "profile-b": 1}


@pytest.mark.asyncio
async def test_get_nous_recommended_aux_model_preserves_tier_preference():
    payload = {
        "paidRecommendedCompactionModel": {"modelName": "anthropic/claude-opus-4.7"},
        "freeRecommendedCompactionModel": {
            "modelName": "google/gemini-3-flash-preview"
        },
        "paidRecommendedVisionModel": {"modelName": "openai/gpt-5.4"},
        "freeRecommendedVisionModel": {"modelName": "google/gemini-3-flash-preview"},
    }
    with patch(
        "hermes_cli.models.fetch_nous_recommended_models",
        new=AsyncMock(return_value=payload),
    ):
        text = await models.get_nous_recommended_aux_model(
            vision=False,
            free_tier=False,
        )
        vision = await models.get_nous_recommended_aux_model(
            vision=True,
            free_tier=False,
        )

    assert text == "anthropic/claude-opus-4.7"
    assert vision == "openai/gpt-5.4"


@pytest.mark.asyncio
async def test_get_nous_recommended_aux_model_defaults_to_paid_on_tier_error():
    payload = {
        "paidRecommendedCompactionModel": {"modelName": "paid-model"},
        "freeRecommendedCompactionModel": {"modelName": "free-model"},
    }
    with (
        patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            new=AsyncMock(return_value=payload),
        ),
        patch(
            "hermes_cli.models.check_nous_free_tier",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        assert await models.get_nous_recommended_aux_model(vision=False) == "paid-model"


@pytest.mark.asyncio
async def test_fetch_nous_recommended_models_caches_per_portal():
    payload = {
        "freeRecommendedCompactionModel": {"modelName": "free-model"},
    }
    response = AsyncMock()
    response.json = lambda: payload
    response.raise_for_status = lambda: None
    client = AsyncMock()
    client.get.return_value = response
    client.__aenter__.return_value = client

    with (
        patch("hermes_cli.models.httpx.AsyncClient", return_value=client),
        patch(
            "hermes_cli.models._write_nous_recommended_disk",
            new=AsyncMock(),
        ) as write_cache,
    ):
        first = await models.fetch_nous_recommended_models("https://portal.example.com")
        second = await models.fetch_nous_recommended_models(
            "https://portal.example.com"
        )

    assert first == payload
    assert second == payload
    client.get.assert_awaited_once_with(
        "https://portal.example.com/api/nous/recommended-models",
        headers={"Accept": "application/json"},
    )
    write_cache.assert_awaited_once_with("https://portal.example.com", payload)
