"""Fallback entry API-key resolution parity tests."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import resolve_entry_api_key


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {
            "provider": "custom",
            "api_key": "inline-key",
            "key_env": "FB_KEY",
        }
        assert resolve_entry_api_key(entry) == "inline-key"

    def test_no_key_fields_returns_none(self):
        assert (
            resolve_entry_api_key({"provider": "openrouter", "model": "glm"})
            is None
        )

    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_resolves_from_active_secret_scope_not_raw_env(
        self, monkeypatch
    ):
        monkeypatch.setenv("FB_KEY", "fake-other-profile-key")
        token = set_secret_scope({"FB_KEY": "fake-active-profile-key"})
        try:
            assert (
                resolve_entry_api_key({"key_env": "FB_KEY"})
                == "fake-active-profile-key"
            )
        finally:
            reset_secret_scope(token)

    def test_key_env_falls_back_to_env_when_no_active_scope(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"
