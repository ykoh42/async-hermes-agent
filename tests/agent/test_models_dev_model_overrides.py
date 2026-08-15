"""Behavior-level tests for the upstream model_overrides contract."""

import pytest

import agent.models_dev as models_dev


pytestmark = pytest.mark.asyncio


@pytest.fixture
def overrides(monkeypatch):
    value = {
        "openai": {
            "known": {
                "context_window": 123456,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "model_family": "manual-family",
            },
            "_default": {"context_window": 77777},
        },
        "_default": {"context_window": 55555},
    }

    async def load():
        return value

    monkeypatch.setattr(models_dev, "_load_model_overrides", load)
    return value


async def test_explicit_override_wins_and_unknown_model_uses_safe_defaults(
    monkeypatch, overrides
):
    async def catalog():
        return {
            "openai": {
                "models": {
                    "known": {
                        "tool_call": False,
                        "modalities": {"input": []},
                        "limit": {"context": 8192, "output": 2048},
                        "family": "catalog-family",
                    }
                }
            }
        }

    monkeypatch.setattr(models_dev, "fetch_models_dev", catalog)

    caps = await models_dev.get_model_capabilities("openai", "known")
    assert caps.context_window == 123456
    assert caps.max_output_tokens == 4096
    assert caps.supports_tools is False
    assert caps.supports_vision is True
    assert caps.model_family == "manual-family"

    unknown = await models_dev.get_model_capabilities("openai", "missing")
    assert unknown is not None
    assert unknown.context_window == 77777

    info = await models_dev.get_model_info("openai", "known")
    assert info is not None
    assert info.context_window == 123456
    assert info.max_output == 4096
    assert info.input_modalities == ("image",)


async def test_fill_gap_defaults_apply_only_to_catalog_misses(monkeypatch, overrides):
    async def catalog():
        return {"openai": {"models": {}}}

    monkeypatch.setattr(models_dev, "fetch_models_dev", catalog)

    assert await models_dev.lookup_models_dev_context("openai", "missing") == 77777
    caps = await models_dev.get_model_capabilities("openai", "missing")
    assert caps is not None
    assert caps.context_window == 77777

    info = await models_dev.get_model_info("openai", "missing")
    assert info is not None
    assert info.context_window == 77777


async def test_global_fill_gap_default_supports_unknown_provider(monkeypatch, overrides):
    async def catalog():
        return {}

    monkeypatch.setattr(models_dev, "fetch_models_dev", catalog)
    assert await models_dev.lookup_models_dev_context("not-in-catalog", "missing") == 55555
