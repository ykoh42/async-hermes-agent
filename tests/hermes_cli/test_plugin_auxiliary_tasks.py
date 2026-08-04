"""Tests for the plugin auxiliary-task registration API.

Covers:
  - PluginContext.register_auxiliary_task() validation
  - PluginManager._aux_tasks storage + force-rediscovery clearing
  - get_plugin_auxiliary_tasks() module-level helper
  - _all_aux_tasks() merge of built-in + plugin tasks
  - _reset_aux_to_auto() includes plugin tasks
  - _get_auxiliary_task_config() layers plugin defaults under user config
"""

from __future__ import annotations

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_auxiliary_tasks,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_ctx(name: str = "test_plugin") -> tuple[PluginContext, PluginManager]:
    """Build a PluginContext + fresh PluginManager wired together.

    The manager skips discovery (no plugins.yaml, no scan) so the test
    can exercise registration paths directly.
    """
    manager = PluginManager()
    manager._discovered = True  # skip auto-discovery on lookup
    manifest = PluginManifest(name=name)
    ctx = PluginContext(manifest, manager)
    return ctx, manager


@pytest.fixture
def patched_manager(monkeypatch):
    """Replace the module-level singleton with a fresh manager for the test.

    Restored automatically after the test by monkeypatch.
    """
    from hermes_cli import plugins as plugins_mod

    fresh = PluginManager()
    fresh._discovered = True
    monkeypatch.setattr(plugins_mod, "_PLUGIN_MANAGER", fresh, raising=False)

    def _stub_get_manager() -> PluginManager:
        return fresh

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _stub_get_manager)
    monkeypatch.setattr(plugins_mod, "_ensure_plugins_discovered", _stub_get_manager)
    yield fresh


# ── PluginContext.register_auxiliary_task ────────────────────────────────────


def test_register_auxiliary_task_basic():
    ctx, manager = _make_ctx("my_plugin")
    ctx.register_auxiliary_task(
        key="my_task",
        display_name="My task",
        description="a custom side task",
    )
    assert "my_task" in manager._aux_tasks
    entry = manager._aux_tasks["my_task"]
    assert entry["key"] == "my_task"
    assert entry["display_name"] == "My task"
    assert entry["description"] == "a custom side task"
    assert entry["plugin"] == "my_plugin"
    # Routing defaults populated
    assert entry["defaults"]["provider"] == "auto"
    assert entry["defaults"]["model"] == ""
    assert entry["defaults"]["timeout"] == 60




# ── PluginManager state lifecycle ────────────────────────────────────────────




# ── Module-level helper ──────────────────────────────────────────────────────




# ── _all_aux_tasks merges built-in + plugin ──────────────────────────────────


# ── auxiliary_client._get_auxiliary_task_config defaults layering ────────────



