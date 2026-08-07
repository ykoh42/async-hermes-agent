"""Tests for retained runtime behavior in :mod:`hermes_cli.models`."""

from unittest.mock import AsyncMock, patch

import pytest

import hermes_cli.models as models
from hermes_cli.nous_account import NousPortalAccountInfo


@pytest.fixture(autouse=True)
def _clear_nous_model_caches():
    models._free_tier_cache = None
    models._nous_recommended_cache.clear()
    yield
    models._free_tier_cache = None
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
