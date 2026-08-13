"""Regression coverage for opt-in live E2E credential passthrough."""

from pathlib import Path

from tests import conftest


LIVE_E2E = conftest.PROJECT_ROOT / "tests" / "e2e" / "test_live_probe.py"


def test_ordinary_tests_never_inherit_live_provider_credentials() -> None:
    environment = {
        "HERMES_LIVE_TESTS": "1",
        "HERMES_LIVE_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "test-only-secret",
    }

    assert conftest._live_e2e_credential_names(
        environment, Path(__file__)
    ) == frozenset()


def test_openrouter_live_e2e_preserves_only_its_explicit_key() -> None:
    environment = {
        "HERMES_LIVE_TESTS": "1",
        "HERMES_LIVE_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "test-only-secret",
        "ANTHROPIC_API_KEY": "must-remain-hermetic",
    }

    assert conftest._live_e2e_credential_names(environment, LIVE_E2E) == frozenset(
        {"OPENROUTER_API_KEY"}
    )


def test_copilot_live_e2e_preserves_supported_token_sources() -> None:
    environment = {
        "HERMES_LIVE_TESTS": "1",
        "HERMES_LIVE_PROVIDER": "copilot",
    }

    assert conftest._live_e2e_credential_names(environment, LIVE_E2E) == frozenset(
        {"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
    )


def test_reasoning_live_e2e_preserves_explicit_and_openrouter_keys() -> None:
    environment = {
        "HERMES_LIVE_REASONING_TESTS": "1",
        "HERMES_LIVE_REASONING_PROVIDER": "openrouter",
    }

    assert conftest._live_e2e_credential_names(environment, LIVE_E2E) == frozenset(
        {"HERMES_LIVE_REASONING_API_KEY", "OPENROUTER_API_KEY"}
    )
