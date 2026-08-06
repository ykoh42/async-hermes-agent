"""Copilot ACP deprecated CLI detection regressions."""

import pytest

from agent.copilot_acp_client import _is_gh_copilot_deprecation_message


@pytest.mark.parametrize(
    "stderr_text",
    [
        "The gh-copilot extension has been deprecated.",
        "gh-copilot: no commands will be executed.",
        "The GH-Copilot Extension HAS BEEN DEPRECATED.",
    ],
)
def test_genuine_deprecation_variants_match(stderr_text: str):
    assert _is_gh_copilot_deprecation_message(stderr_text)


@pytest.mark.parametrize(
    "stderr_text",
    [
        "Error: connection refused",
        "",
        "copilot-cli: failed to authenticate with the API",
        "warning: the --foo flag is scheduled for deprecation in v3",
        "See https://github.com/github/copilot-cli/issues for support",
        "gh-copilot: command not found",
        "extension has been deprecated (some other extension)",
    ],
)
def test_new_cli_and_generic_errors_do_not_match(stderr_text: str):
    assert not _is_gh_copilot_deprecation_message(stderr_text)


def test_github_models_azure_url_maps_to_copilot():
    from agent.model_metadata import _URL_TO_PROVIDER

    assert _URL_TO_PROVIDER.get("models.inference.ai.azure.com") == "copilot"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://models.inference.ai.azure.com",
        "https://models.inference.ai.azure.com/v1/chat",
        "https://models.github.ai/inference",
    ],
)
def test_github_models_base_urls_are_recognized(base_url: str):
    from hermes_cli.models import _is_github_models_base_url

    assert _is_github_models_base_url(base_url)
