from __future__ import annotations

import asyncio

import pytest
from blockbuster import BlockBuster

from hermes_cli import config as config_module


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_readonly_cache_preserves_identity_and_env_invalidation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ASYNC_HERMES_TEST_KEY", "first")
    _write_config(tmp_path, "model:\n  default: ${ASYNC_HERMES_TEST_KEY}\n")

    first = await config_module.load_config_readonly()
    second = await config_module.load_config_readonly()
    assert first is second
    assert first["model"]["default"] == "first"

    monkeypatch.setenv("ASYNC_HERMES_TEST_KEY", "second")
    third = await config_module.load_config_readonly()
    assert third is not first
    assert third["model"]["default"] == "second"


@pytest.mark.asyncio
async def test_concurrent_first_load_returns_one_cached_object(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, "model:\n  default: concurrent/model\n")

    loaded = await asyncio.gather(
        *(config_module.load_config_readonly() for _ in range(8))
    )

    assert all(config is loaded[0] for config in loaded)


@pytest.mark.asyncio
async def test_readonly_cache_retains_last_known_good_without_blocking(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, "model:\n  default: known-good\n")
    expected = await config_module.load_config_readonly()

    _write_config(tmp_path, "model: [broken\n")
    blocker = BlockBuster()
    blocker.activate()
    try:
        retained = await config_module.load_config_readonly()
    finally:
        blocker.deactivate()

    assert retained["model"]["default"] == expected["model"]["default"]
    assert list(tmp_path.glob("config.yaml.corrupt.*.bak"))


@pytest.mark.asyncio
async def test_non_mapping_yaml_uses_upstream_parse_failure_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, "model:\n  default: known-good\n")
    expected = await config_module.load_config_readonly()

    _write_config(tmp_path, "- not\n- a\n- mapping\n")
    retained = await config_module.load_config_readonly()

    assert retained["model"]["default"] == expected["model"]["default"]
