"""Tests for profile identity detection used by the agent runtime."""

from pathlib import Path

from hermes_cli.profiles import get_active_profile_name


def test_default_profile(tmp_path: Path, monkeypatch) -> None:
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    assert get_active_profile_name() == "default"


def test_named_profile(tmp_path: Path, monkeypatch) -> None:
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert get_active_profile_name() == "coder"


def test_custom_deployment_is_its_default_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_home = tmp_path / "deployment"
    custom_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setenv("HERMES_HOME", str(custom_home))

    assert get_active_profile_name() == "default"
