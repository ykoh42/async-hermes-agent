"""Tests for Hindsight's declared config surface."""

import pytest

from plugins.memory.config_schema import (
    KIND_SECRET,
    KIND_SELECT,
    get_provider_config_schema,
)

pytestmark = pytest.mark.asyncio


async def test_hindsight_is_declared():
    provider = await get_provider_config_schema("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    assert {field.key for field in provider.fields} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
    }


async def test_fields_are_all_inline():
    provider = await get_provider_config_schema("hindsight")
    assert provider is not None
    assert all(field.inline for field in provider.fields)


async def test_mode_gating_is_expressed_as_select_options():
    provider = await get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    assert "local_embedded" not in mode.allowed_values()


async def test_api_key_is_a_secret_bound_to_env():
    provider = await get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"
