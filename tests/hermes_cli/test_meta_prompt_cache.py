"""Meta host mandate and provider routing contracts."""

from hermes_cli import runtime_provider
from hermes_cli.providers import determine_api_mode, host_mandated_api_mode


def test_meta_exact_host_mandates_responses():
    for url in (
        "https://api.meta.ai/v1",
        "https://API.META.AI/v1/",
        "https://api.meta.ai:443/v1",
    ):
        assert host_mandated_api_mode(url) == "codex_responses"
        assert runtime_provider._detect_api_mode_for_url(url) == "codex_responses"


def test_meta_host_spoofs_are_not_mandated():
    for url in (
        "https://api.meta.ai.attacker.test/v1",
        "https://proxy.test/api.meta.ai/v1",
        "https://meta.ai/v1",
    ):
        assert host_mandated_api_mode(url) is None
        assert runtime_provider._detect_api_mode_for_url(url) is None


def test_meta_provider_and_custom_endpoint_routing():
    assert determine_api_mode("meta", "https://api.meta.ai/v1", "muse-spark") == (
        "codex_responses"
    )
    assert determine_api_mode("custom", "https://api.meta.ai/v1", "muse-spark") == (
        "codex_responses"
    )
    assert (
        determine_api_mode("meta", "https://custom.meta.endpoint/v1", "muse-spark")
        == "chat_completions"
    )
