"""Unit tests for hermes_cli.managed_scope (resolver + loaders + key helpers)."""
import textwrap
from pathlib import Path

import pytest
from blockbuster import BlockBuster


# ── Directory resolver ───────────────────────────────────────────────────────






# ── Loaders + key helpers ────────────────────────────────────────────────────


def _write_managed(tmp_path, monkeypatch, *, config=None, env=None):
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir(exist_ok=True)
    if config is not None:
        (managed / "config.yaml").write_text(textwrap.dedent(config), encoding="utf-8")
    if env is not None:
        (managed / ".env").write_text(textwrap.dedent(env), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    return managed








def test_load_managed_env_and_is_env_managed(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    _write_managed(
        tmp_path, monkeypatch, env="OPENAI_API_BASE=https://org.example/v1\n"
    )
    assert managed_scope.load_managed_env() == {
        "OPENAI_API_BASE": "https://org.example/v1"
    }
    assert managed_scope.is_env_managed("OPENAI_API_BASE") is True
    assert managed_scope.is_env_managed("OTHER") is False




def test_managed_dir_env_scrubbed_by_default():
    """conftest must scrub HERMES_MANAGED_DIR so a dev-shell value can't leak in."""
    import os

    assert "HERMES_MANAGED_DIR" not in os.environ


@pytest.mark.asyncio
async def test_readonly_config_load_avoids_sync_managed_directory_probe(
    tmp_path, monkeypatch
):
    from hermes_cli.config import load_config_readonly

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "approvals:\n  deny:\n    - 'curl *'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("load_config_readonly must not probe directories synchronously")
        ),
    )

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        config = await load_config_readonly()
    finally:
        blockbuster.deactivate()

    assert config["approvals"]["deny"] == ["curl *"]
