"""Meta Responses usage cache accounting."""

from types import SimpleNamespace

from agent.usage_pricing import normalize_usage


def test_meta_cached_tokens_flow_to_cache_read():
    usage = SimpleNamespace(
        input_tokens=4000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(
            cached_tokens=3920, cache_creation_tokens=0
        ),
        output_tokens_details=None,
        prompt_tokens=None,
        completion_tokens=None,
    )
    normalized = normalize_usage(usage, api_mode="codex_responses")
    assert normalized.cache_read_tokens == 3920
    assert normalized.input_tokens == 80


def test_meta_cache_write_tokens_are_reported():
    usage = SimpleNamespace(
        input_tokens=4000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(
            cached_tokens=0, cache_creation_tokens=4000
        ),
    )
    normalized = normalize_usage(usage, api_mode="codex_responses")
    assert normalized.cache_write_tokens == 4000
