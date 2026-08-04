"""Secret-scope invariants for retained provider execution boundaries."""

import pytest

from agent import secret_scope as ss
from agent.auxiliary_client import _scoped_key_env


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class _Scope:
    def __init__(self, mapping):
        self.mapping = mapping
        self.token = None

    def __enter__(self):
        self.token = ss.set_secret_scope(self.mapping)
        return self

    def __exit__(self, *_exc):
        ss.reset_secret_scope(self.token)


def test_auxiliary_scoped_value_wins(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    ss.set_multiplex_active(True)
    with _Scope({"OPENROUTER_API_KEY": "sk-scoped"}):
        assert _scoped_key_env("OPENROUTER_API_KEY") == "sk-scoped"


def test_auxiliary_scoped_miss_does_not_borrow(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
    ss.set_multiplex_active(True)
    with _Scope({"UNRELATED": "x"}):
        assert _scoped_key_env("OPENAI_API_KEY") == ""


def test_auxiliary_unscoped_startup_keeps_legacy_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-own-env")
    ss.set_multiplex_active(True)
    assert _scoped_key_env("OPENAI_API_KEY") == "sk-own-env"


def test_auxiliary_empty_name_is_empty():
    assert _scoped_key_env("") == ""
