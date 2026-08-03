from tools.environments.local import (
    _is_hermes_internal_secret,
    _sanitize_subprocess_env,
)


def test_provider_credentials_are_removed():
    result = _sanitize_subprocess_env(
        {"PATH": "/bin", "OPENAI_API_KEY": "secret", "USER_VALUE": "ok"}
    )
    assert result["PATH"] == "/bin"
    assert result["USER_VALUE"] == "ok"
    assert "OPENAI_API_KEY" not in result


def test_dynamic_auxiliary_and_relay_secrets_are_removed():
    assert _is_hermes_internal_secret("AUXILIARY_VISION_API_KEY")
    assert _is_hermes_internal_secret("GATEWAY_RELAY_MAIN_SECRET")
    result = _sanitize_subprocess_env(
        {
            "AUXILIARY_VISION_API_KEY": "secret",
            "GATEWAY_RELAY_MAIN_SECRET": "secret",
            "GATEWAY_RELAY_URL": "https://example.test",
        }
    )
    assert "AUXILIARY_VISION_API_KEY" not in result
    assert "GATEWAY_RELAY_MAIN_SECRET" not in result
    assert result["GATEWAY_RELAY_URL"] == "https://example.test"
