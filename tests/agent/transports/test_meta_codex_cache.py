"""Meta Responses prompt-cache behavior."""

from agent.transports import get_transport
from agent.transports.codex import _default_prompt_cache_retention_for_request


def test_meta_retention_applies_independent_of_model_name():
    transport = get_transport("codex_responses")
    for model in ("muse-spark-1.2", "muse-spark", "gpt-5.4", ""):
        kwargs = transport.build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://api.meta.ai/v1",
            session_id="sid",
        )
        assert kwargs["prompt_cache_retention"] == "24h"


def test_meta_retention_override_wins():
    kwargs = get_transport("codex_responses").build_kwargs(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        base_url="https://api.meta.ai/v1",
        request_overrides={"prompt_cache_retention": "in_memory"},
    )
    assert kwargs["prompt_cache_retention"] == "in_memory"
    assert (
        _default_prompt_cache_retention_for_request(
            "muse-spark-1.2", "https://API.META.AI:443/v1"
        )
        == "24h"
    )
