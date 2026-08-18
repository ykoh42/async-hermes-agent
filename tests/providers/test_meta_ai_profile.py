"""Behavior contract for the bundled Meta Model API provider profile."""

import pytest

from providers import get_provider_profile


async def _profile():
    profile = await get_provider_profile("meta-ai")
    assert profile is not None
    return profile


@pytest.mark.asyncio
async def test_meta_profile_registered():
    profile = await _profile()
    assert profile.name == "meta-ai"
    assert profile.base_url == "https://api.meta.ai/v1"
    assert profile.auth_type == "api_key"
    assert profile.api_mode == "codex_responses"
    assert "MODEL_API_KEY" in profile.env_vars
    assert profile.supports_vision is True
    assert profile.default_aux_model == "muse-spark-1.2-contributor"
    assert profile.default_max_tokens == 16384
    assert set(profile.fallback_models) == {
        "muse-spark-1.2",
        "muse-spark-1.2-contributor",
    }


@pytest.mark.asyncio
async def test_meta_aliases_resolve_to_same_profile():
    profile = await _profile()
    for alias in ("meta", "muse", "muse-spark", "model-api", "msl"):
        assert await get_provider_profile(alias) is profile


@pytest.mark.asyncio
async def test_meta_reasoning_effort_is_top_level_and_meta_safe():
    profile = await _profile()
    assert profile.build_api_kwargs_extras(reasoning_config=None) == (
        {},
        {"reasoning_effort": "medium"},
    )
    assert profile.build_api_kwargs_extras(reasoning_config={"enabled": False}) == (
        {},
        {"reasoning_effort": "minimal"},
    )
    assert profile.build_api_kwargs_extras(reasoning_config={"effort": "high"}) == (
        {},
        {"reasoning_effort": "high"},
    )
    assert profile.build_api_kwargs_extras(reasoning_config={"effort": "none"}) == (
        {},
        {"reasoning_effort": "minimal"},
    )
    assert profile.build_api_kwargs_extras(reasoning_config={"effort": "ultra"}) == (
        {},
        {"reasoning_effort": "xhigh"},
    )
