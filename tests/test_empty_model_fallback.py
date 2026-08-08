"""Tests for provider default-model resolution."""


class TestGetDefaultModelForProvider:
    """Unit tests for hermes_cli.models.get_default_model_for_provider."""

    def test_known_provider_returns_first_model(self):
        from hermes_cli.models import get_default_model_for_provider

        result = get_default_model_for_provider("openai-codex")
        assert result
        assert isinstance(result, str)
