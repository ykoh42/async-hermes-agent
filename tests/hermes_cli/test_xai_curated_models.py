"""Regression tests for xAI curated model list (OAuth picker)."""

from hermes_cli.models import _PROVIDER_MODELS


def test_xai_oauth_includes_grok_composer_2_5_fast():
    models = _PROVIDER_MODELS["xai-oauth"]
    assert "grok-composer-2.5-fast" in models
