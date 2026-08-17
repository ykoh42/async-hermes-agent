"""Provider-qualified model defaults are flattened at shared boundaries."""

from types import SimpleNamespace

from hermes_cli.config import split_model_config_default


def test_split_model_config_default_preserves_model_and_provider():
    assert split_model_config_default(
        {"provider": "custom", "model": "publisher/model"}
    ) == ("publisher/model", "custom")


def test_split_model_config_default_accepts_scalar_and_fallback_key():
    assert split_model_config_default("publisher/model") == ("publisher/model", "")
    assert split_model_config_default({"default": "publisher/model"}) == (
        "publisher/model",
        "",
    )


def test_anthropic_cache_policy_never_calls_lower_on_dict_model():
    from agent.agent_runtime_helpers import anthropic_prompt_cache_policy

    agent = SimpleNamespace(
        _cache_disabled=False,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        model={"provider": "openrouter", "model": "anthropic/claude-sonnet"},
        _runtime_config_snapshot={},
    )
    enabled, native = anthropic_prompt_cache_policy(agent)
    assert enabled is True
    assert native is False
