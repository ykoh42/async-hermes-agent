"""Tests for OSS provider definitions and validation."""

from plugins.memory.mem0._oss_providers import (
    EMBEDDER_PROVIDERS,
    KNOWN_DIMS,
    LLM_PROVIDERS,
    VECTOR_PROVIDERS,
    validate_oss_config,
)


def test_provider_definitions_have_required_keys():
    for provider in LLM_PROVIDERS.values():
        assert {"label", "needs_key", "default_model"} <= provider.keys()
    for provider in EMBEDDER_PROVIDERS.values():
        assert {"label", "needs_key", "default_model", "dims"} <= provider.keys()
    for provider in VECTOR_PROVIDERS.values():
        assert {"label", "default_config"} <= provider.keys()
    for provider in EMBEDDER_PROVIDERS.values():
        assert provider["default_model"] in KNOWN_DIMS


def test_oss_validation_accepts_complete_config_and_rejects_missing_fields():
    valid = {
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
        "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/test"}},
    }
    assert validate_oss_config(valid) == []
    assert any(
        "llm" in error.lower()
        for error in validate_oss_config(
            {
                "embedder": valid["embedder"],
                "vector_store": valid["vector_store"],
            }
        )
    )
