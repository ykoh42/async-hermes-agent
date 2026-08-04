"""Unit tests for find_custom_provider_identity (base_url → custom:<name>).

Reverse lookup used by tui_gateway session persistence to recover a named
``providers:`` / ``custom_providers:`` entry from the only durable fact the
session row keeps once the provider has been resolved to the literal string
"custom": the endpoint URL. See
tests/tui_gateway/test_custom_provider_session_persistence.py for the
end-to-end persist/resume round-trip.
"""

import hermes_cli.runtime_provider as rp


def test_matches_legacy_custom_providers_list():
    config = {
        "custom_providers": [
            {"name": "MiMo v2.5 Pro", "base_url": "https://api.mimo.example/v1"}
        ]
    }
    assert (
        rp.find_custom_provider_identity("https://api.mimo.example/v1", config=config)
        == "custom:mimo-v2.5-pro"
    )


def test_matches_providers_dict_by_key():
    config = {"providers": {"local": {"api": "http://127.0.0.1:8000/v1"}}}
    assert (
        rp.find_custom_provider_identity(
            "http://127.0.0.1:8000/v1", config=config
        )
        == "custom:local"
    )


def test_matches_providers_dict_by_stable_key_not_display_name():
    config = {
        "providers": {
            "local-127.0.0.1:8000": {
                "name": "Local Ollama",
                "api": "http://127.0.0.1:8000/v1",
            }
        }
    }
    slug = rp.find_custom_provider_identity(
        "http://127.0.0.1:8000/v1", config=config
    )
    assert slug == "custom:local-127.0.0.1:8000"

    entry = rp._get_named_custom_provider(slug, config=config)
    assert entry is not None
    assert entry["name"] == "Local Ollama"


def test_match_ignores_trailing_slash_and_case():
    config = {
        "custom_providers": [
            {"name": "local", "base_url": "http://Localhost:8000/v1/"}
        ]
    }
    assert (
        rp.find_custom_provider_identity("http://localhost:8000/v1", config=config)
        == "custom:local"
    )


def test_no_match_returns_none():
    config = {
        "custom_providers": [
            {"name": "other", "base_url": "https://elsewhere.example/v1"}
        ]
    }
    assert rp.find_custom_provider_identity(
        "https://api.mimo.example/v1", config=config
    ) is None


def test_empty_base_url_returns_none():
    config = {"custom_providers": [{"name": "x"}]}
    assert rp.find_custom_provider_identity("", config=config) is None
    assert rp.find_custom_provider_identity(None, config=config) is None


def test_identity_resolves_back_through_named_lookup():
    """The returned slug must be accepted by _get_named_custom_provider —
    that is the whole point of persisting it."""
    config = {
        "custom_providers": [
            {
                "name": "mimo-v2.5-pro",
                "base_url": "https://api.mimo.example/v1",
                "api_key": "sk-entry",
            }
        ]
    }
    slug = rp.find_custom_provider_identity(
        "https://api.mimo.example/v1", config=config
    )
    assert slug == "custom:mimo-v2.5-pro"

    entry = rp._get_named_custom_provider(slug, config=config)
    assert entry is not None
    assert entry["base_url"] == "https://api.mimo.example/v1"
    assert entry["api_key"] == "sk-entry"
