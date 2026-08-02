"""Provider configuration persistence tests retained by the library runtime."""

from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal string-format config."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: some-old-model\n")
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in (
        "HERMES_MODEL",
        "LLM_MODEL",
        "HERMES_INFERENCE_PROVIDER",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "STEPFUN_API_KEY",
        "STEPFUN_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


class TestSaveModelChoiceAlwaysDict:
    def test_string_model_becomes_dict(self, config_home):
        """The reusable auth helper normalizes the legacy string form."""
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("kimi-k2.5")

        import yaml

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        assert config["model"]["default"] == "kimi-k2.5"


class TestProviderPersistsAfterModelSave:
    def test_update_config_for_provider_uses_atomic_yaml_write(self, config_home):
        """Provider writes stay atomic independently of the deleted CLI flow."""
        from hermes_cli.auth import _update_config_for_provider

        config_path = config_home / "config.yaml"
        original_text = config_path.read_text(encoding="utf-8")

        def fail_write(path, data, **kwargs):
            assert path == config_path
            assert data["model"]["provider"] == "nous"
            assert data["model"]["base_url"] == "https://inference.example.com/v1"
            assert data["model"]["default"] == "some-old-model"
            assert kwargs["sort_keys"] is False
            raise OSError("simulated atomic write failure")

        with patch("hermes_cli.auth.atomic_yaml_write", side_effect=fail_write):
            with pytest.raises(OSError, match="simulated atomic write failure"):
                _update_config_for_provider(
                    "nous",
                    "https://inference.example.com/v1/",
                    default_model="llama-3.3",
                )

        assert config_path.read_text(encoding="utf-8") == original_text


class TestZaiEndpointPicker:
    def test_current_endpoint_is_default_choice(self, config_home):
        """The reusable endpoint picker selects the currently active URL."""
        from hermes_cli.auth import ZAI_ENDPOINTS
        from hermes_cli.model_setup_flows import _select_zai_endpoint

        coding_url = ZAI_ENDPOINTS[2][1]
        captured = {}

        def choose(choices, *, default=0, title=""):
            captured["default"] = default
            captured["choices"] = choices
            return default

        with patch("hermes_cli.main._prompt_provider_choice", side_effect=choose):
            result = _select_zai_endpoint(coding_url)

        assert captured["default"] == 2
        assert result == coding_url
