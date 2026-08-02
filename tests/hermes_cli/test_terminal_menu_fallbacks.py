"""Regression tests for numbered fallbacks when the interactive curses menu
cannot initialize (e.g. non-TTY, curses unavailable, terminal error)."""

import subprocess
from types import SimpleNamespace

from hermes_cli.config import load_config, save_config


def _raise_menu(*args, **kwargs):
    # Mimic curses_radiolist hitting an unrecoverable terminal error so the
    # caller's except clause routes to the numbered-input fallback.
    raise subprocess.CalledProcessError(2, ["tput", "clear"])




def test_prompt_model_selection_requires_expensive_confirmation(monkeypatch, capsys):
    from hermes_cli.auth import _prompt_model_selection

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *_args, **_kwargs: SimpleNamespace(message="EXPENSIVE MODEL WARNING"),
    )
    responses = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    selected = _prompt_model_selection(
        ["openai/gpt-5.5-pro"],
        confirm_provider="nous",
    )

    out = capsys.readouterr().out
    assert selected is None
    assert "EXPENSIVE MODEL WARNING" in out
