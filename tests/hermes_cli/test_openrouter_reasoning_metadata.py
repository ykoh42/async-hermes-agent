"""OpenRouter reasoning metadata parity tests."""

from unittest.mock import AsyncMock

import pytest

import hermes_cli.models as models


@pytest.fixture(autouse=True)
def _reset_reasoning_cache():
    previous_cache = models._openrouter_reasoning_caps_cache
    previous_failure = models._openrouter_reasoning_caps_failed_at
    models._openrouter_reasoning_caps_cache = None
    models._openrouter_reasoning_caps_failed_at = None
    yield
    models._openrouter_reasoning_caps_cache = previous_cache
    models._openrouter_reasoning_caps_failed_at = previous_failure


def test_parse_reasoning_capabilities_is_tri_state():
    assert models.parse_openrouter_reasoning_capabilities({}) is None
    assert models.parse_openrouter_reasoning_capabilities(
        {"supported_parameters": ["tools"]}
    ) == {"supports_reasoning": False}
    assert models.parse_openrouter_reasoning_capabilities(
        {
            "supported_parameters": ["reasoning"],
            "reasoning": {
                "mandatory": True,
                "supported_efforts": ["LOW", "high", "low"],
            },
        }
    ) == {
        "supports_reasoning": True,
        "supported_efforts": ["low", "high"],
        "mandatory": True,
    }


@pytest.mark.parametrize(
    "item, expected",
    [
        (
            {
                "supported_parameters": ["temperature", "tools", "reasoning"],
                "reasoning": {
                    "mandatory": False,
                    "supported_efforts": ["low", "medium", "high"],
                },
            },
            {
                "supports_reasoning": True,
                "supported_efforts": ["low", "medium", "high"],
                "mandatory": False,
            },
        ),
        (
            {"supported_parameters": ["reasoning"], "reasoning": {}},
            {
                "supports_reasoning": True,
                "supported_efforts": None,
                "mandatory": False,
            },
        ),
        (
            {"supported_parameters": ["reasoning"]},
            {
                "supports_reasoning": True,
                "supported_efforts": None,
                "mandatory": False,
            },
        ),
    ],
)
def test_parse_reasoning_capabilities_matches_upstream_cases(item, expected):
    assert models.parse_openrouter_reasoning_capabilities(item) == expected


def test_parse_reasoning_capabilities_rejects_untrusted_object():
    assert models.parse_openrouter_reasoning_capabilities(
        {
            "supported_parameters": ["temperature", "tools"],
            "reasoning": {"supported_efforts": ["high"]},
        }
    ) == {"supports_reasoning": False}


def test_parse_reasoning_capabilities_normalizes_and_deduplicates_efforts():
    capabilities = models.parse_openrouter_reasoning_capabilities(
        {
            "supported_parameters": ["reasoning"],
            "reasoning": {"supported_efforts": [" High ", "high", "LOW", "", 3]},
        }
    )
    assert capabilities["supported_efforts"] == ["high", "low", "3"]


@pytest.mark.parametrize(
    ("requested", "supported", "expected"),
    [
        ("high", ["low", "medium", "high"], "high"),
        ("ultra", None, "ultra"),
        ("ultra", [], "ultra"),
        ("ultra", ["low", "medium", "high"], "high"),
        ("xhigh", ["low", "medium", "high"], "high"),
        ("max", ["minimal", "low"], "low"),
        ("minimal", ["low", "medium"], "low"),
        ("none", ["low", "high"], "low"),
        ("high", ["low", "high"], "high"),
        ("custom", ["low", "high"], "custom"),
        ("banana", ["low", "high"], "banana"),
        (None, ["low"], None),
        ("", ["low"], ""),
    ],
)
def test_reasoning_effort_clamps_without_escalating(
    requested, supported, expected
):
    assert (
        models.clamp_reasoning_effort_to_supported(requested, supported)
        == expected
    )


def test_reasoning_effort_clamp_never_escalates():
    assert (
        models.clamp_reasoning_effort_to_supported(
            "medium", ["low", "xhigh"]
        )
        == "low"
    )


@pytest.mark.asyncio
async def test_reasoning_capabilities_fetches_natively_and_caches(monkeypatch):
    fetch = AsyncMock(
        return_value={
            "openai/gpt-5.5": {
                "supports_reasoning": True,
                "supported_efforts": ["low", "high"],
                "mandatory": False,
            }
        }
    )
    monkeypatch.setattr(models, "_fetch_openrouter_reasoning_caps", fetch)

    first = await models.openrouter_model_reasoning_capabilities(
        "openai/gpt-5.5", allow_fetch=True
    )
    second = await models.openrouter_model_reasoning_capabilities(
        "openai/gpt-5.5", allow_fetch=True
    )
    assert first == second
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_reasoning_capabilities_cache_only_default_never_fetches(monkeypatch):
    fetch = AsyncMock(side_effect=AssertionError("cold cache must not fetch"))
    monkeypatch.setattr(models, "_fetch_openrouter_reasoning_caps", fetch)
    assert (
        await models.openrouter_model_reasoning_capabilities("a/b") is None
    )
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_reasoning_capabilities_failure_is_rate_limited(monkeypatch):
    client_factory = AsyncMock(side_effect=OSError("offline"))
    monkeypatch.setattr(models, "_create_httpx_client", client_factory)
    assert (
        await models.openrouter_model_reasoning_capabilities("a/b", allow_fetch=True)
        is None
    )
    assert (
        await models.openrouter_model_reasoning_capabilities("a/b", allow_fetch=True)
        is None
    )
    # The built-in fetcher enforces a 60-second failure throttle.
    client_factory.assert_awaited_once()


@pytest.mark.asyncio
async def test_openrouter_profile_clamps_cached_effort():
    from providers import get_provider_profile

    openrouter = await get_provider_profile("openrouter")
    assert openrouter is not None
    models._openrouter_reasoning_caps_cache = {
        "openai/gpt-5.5": {
            "supports_reasoning": True,
            "supported_efforts": ["low", "medium", "high"],
            "mandatory": False,
        }
    }
    body, _ = openrouter.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "ultra"},
        supports_reasoning=True,
        model="openai/gpt-5.5",
    )
    assert body["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_agent_reasoning_gate_prefers_catalog_and_keeps_static_fallback(
    monkeypatch,
):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._base_url_lower = "https://openrouter.ai/api/v1"
    agent.provider = "openrouter"
    agent.model = "nvidia/nemotron-3-super-120b-a12b"
    lookup = AsyncMock(return_value={"supports_reasoning": False})
    monkeypatch.setattr(models, "openrouter_model_reasoning_capabilities", lookup)
    assert await agent._supports_reasoning_extra_body() is False
    lookup.assert_awaited_once_with(agent.model, allow_fetch=True)

    lookup.return_value = None
    agent.model = "deepseek/deepseek-v4"
    assert await agent._supports_reasoning_extra_body() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "capabilities", "expected"),
    [
        (
            "nvidia/nemotron-3-ultra",
            {"supports_reasoning": True, "supported_efforts": ["low"]},
            True,
        ),
        (
            "openai/gpt-4o-mini",
            {"supports_reasoning": False},
            False,
        ),
        ("deepseek/deepseek-chat", None, True),
        ("someveryunknown/model-x", None, False),
    ],
)
async def test_agent_reasoning_gate_matches_upstream_metadata_cases(
    monkeypatch, model, capabilities, expected
):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._base_url_lower = "https://openrouter.ai/api/v1"
    agent.provider = "openrouter"
    agent.model = model
    lookup = AsyncMock(return_value=capabilities)
    monkeypatch.setattr(models, "openrouter_model_reasoning_capabilities", lookup)
    assert await agent._supports_reasoning_extra_body() is expected


@pytest.mark.asyncio
async def test_openrouter_profile_unknown_capability_passthrough():
    from providers import get_provider_profile

    profile = await get_provider_profile("openrouter")
    assert profile is not None
    models._openrouter_reasoning_caps_cache = {"a/b": None}
    body, _ = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "ultra"},
        supports_reasoning=True,
        model="unlisted/model",
    )
    assert body["reasoning"]["effort"] == "ultra"
