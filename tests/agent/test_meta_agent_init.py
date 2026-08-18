"""AIAgent Meta host mandate contracts."""


def _agent(**kwargs):
    from run_agent import AIAgent

    defaults = {
        "provider": "meta",
        "base_url": "https://api.meta.ai/v1",
        "api_key": "sk-test-meta",
        "model": "muse-spark-1.2",
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
    }
    defaults.update(kwargs)
    return AIAgent(**defaults)


def test_meta_url_selects_responses_and_preserves_provider():
    agent = _agent()
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "meta"


def test_explicit_chat_mode_wins_over_host_mandate():
    agent = _agent(api_mode="chat_completions")
    assert agent.api_mode == "chat_completions"


def test_meta_host_is_case_insensitive_and_provider_none_is_supported():
    assert _agent(base_url="https://API.META.AI/v1").api_mode == "codex_responses"
    assert _agent(provider=None).api_mode == "codex_responses"


def test_meta_provider_custom_endpoint_stays_chat_completions():
    assert _agent(base_url="https://custom.meta.endpoint/v1").api_mode == (
        "chat_completions"
    )
