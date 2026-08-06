"""Tests for `hermes chat --safe-mode` isolation."""

from __future__ import annotations

import os
import sys
import types

import pytest


_VARS = ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG", "HERMES_IGNORE_RULES")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in _VARS:
        os.environ.pop(var, None)


@pytest.mark.asyncio
async def test_plugin_discovery_skipped(monkeypatch):
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    from hermes_cli.plugins import PluginManager

    mgr = PluginManager()
    called = []
    monkeypatch.setattr(mgr, "_discover_and_load_inner", lambda: called.append(True))

    await mgr.discover_and_load()

    assert called == []
    assert mgr._discovered is True
    assert mgr._plugins == {}








