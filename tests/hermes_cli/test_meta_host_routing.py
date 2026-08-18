"""Meta host routing and cache-retention contracts."""

from agent.transports.codex import _default_prompt_cache_retention_for_request
from hermes_cli.providers import host_mandated_api_mode
from hermes_cli.runtime_provider import _detect_api_mode_for_url


def test_meta_host_requires_responses_wire():
    assert host_mandated_api_mode("https://api.meta.ai/v1") == "codex_responses"
    assert _detect_api_mode_for_url("https://api.meta.ai/v1") == "codex_responses"
    assert host_mandated_api_mode("https://api.meta.ai.attacker.test/v1") is None


def test_meta_host_gets_extended_prompt_cache_retention():
    assert (
        _default_prompt_cache_retention_for_request(
            "muse-spark-1.2", "https://api.meta.ai/v1"
        )
        == "24h"
    )
    assert (
        _default_prompt_cache_retention_for_request(
            "muse-spark-1.2", "https://meta.example/v1"
        )
        is None
    )
