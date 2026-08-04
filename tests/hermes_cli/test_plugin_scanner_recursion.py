"""Tests for plugin scanner recursion, kinds, and path-derived keys.

Covers ``_scan_directory`` recursion into category namespaces, ``kind``
parsing, path-derived registry keys, and the gate logic (user backends still
opt in; exclusive kind skipped; unknown kinds → standalone warning).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hermes_cli.plugins import PluginManager


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_plugin(
    root: Path,
    segments: list[str],
    *,
    manifest_extra: Dict[str, Any] | None = None,
    register_body: str = "pass",
) -> Path:
    """Create a plugin dir at ``root/<segments...>/`` with plugin.yaml + __init__.py.

    ``segments`` lets tests build both flat (``["my-plugin"]``) and
    category-namespaced (``["category", "name"]``) layouts.
    """
    plugin_dir = root
    for seg in segments:
        plugin_dir = plugin_dir / seg
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": segments[-1],
        "version": "0.1.0",
        "description": f"Test plugin {'/'.join(segments)}",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )
    return plugin_dir


def _enable(hermes_home: Path, name: str) -> None:
    """Append ``name`` to ``plugins.enabled`` in ``<hermes_home>/config.yaml``."""
    cfg_path = hermes_home / "config.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            cfg = {}
    plugins_cfg = cfg.setdefault("plugins", {})
    enabled = plugins_cfg.setdefault("enabled", [])
    if isinstance(enabled, list) and name not in enabled:
        enabled.append(name)
    cfg_path.write_text(yaml.safe_dump(cfg))


# ── Scanner recursion ──────────────────────────────────────────────────────


class TestCategoryNamespaceRecursion:
    def test_category_namespace_discovered(self, tmp_path, monkeypatch):
        """A nested ``<root>/category/name/plugin.yaml`` is discovered with
        a path-derived key when the category parent has no manifest."""
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        user_plugins = hermes_home / "plugins"

        _write_plugin(user_plugins, ["category", "name"])
        _enable(hermes_home, "category/name")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "category/name" in mgr._plugins
        loaded = mgr._plugins["category/name"]
        assert loaded.manifest.key == "category/name"
        assert loaded.manifest.name == "name"
        assert loaded.enabled is True


    def test_depth_cap_two(self, tmp_path, monkeypatch):
        """Plugins nested three levels deep are not discovered.

        ``<root>/a/b/c/plugin.yaml`` should NOT be picked up — cap is
        two segments.
        """
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        user_plugins = hermes_home / "plugins"

        _write_plugin(user_plugins, ["a", "b", "c"])

        mgr = PluginManager()
        mgr.discover_and_load()

        non_bundled = [
            k for k, p in mgr._plugins.items()
            if p.manifest.source != "bundled"
        ]
        assert non_bundled == []



# ── Kind parsing ───────────────────────────────────────────────────────────


class TestKindField:
    def test_default_kind_is_standalone(self, tmp_path, monkeypatch):
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        _write_plugin(hermes_home / "plugins", ["p1"])
        _enable(hermes_home, "p1")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert mgr._plugins["p1"].manifest.kind == "standalone"

    @pytest.mark.parametrize("kind", ["backend", "exclusive", "standalone"])
    def test_valid_kinds_parsed(self, kind, tmp_path, monkeypatch):
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        _write_plugin(
            hermes_home / "plugins",
            ["p1"],
            manifest_extra={"kind": kind},
        )
        # Not all kinds auto-load, but manifest should parse.
        _enable(hermes_home, "p1")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "p1" in mgr._plugins
        assert mgr._plugins["p1"].manifest.kind == kind

    def test_unknown_kind_falls_back_to_standalone(self, tmp_path, monkeypatch, caplog):
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        _write_plugin(
            hermes_home / "plugins",
            ["p1"],
            manifest_extra={"kind": "bogus"},
        )
        _enable(hermes_home, "p1")

        with caplog.at_level("WARNING"):
            mgr = PluginManager()
            mgr.discover_and_load()

        assert mgr._plugins["p1"].manifest.kind == "standalone"
        assert any(
            "unknown kind" in rec.getMessage() for rec in caplog.records
        )


# ── Gate logic ─────────────────────────────────────────────────────────────


class TestBackendGate:
    def test_user_backend_still_gated_by_enabled(self, tmp_path, monkeypatch):
        """User-installed ``kind: backend`` plugins still require opt-in —
        they're not trusted by default."""
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        user_plugins = hermes_home / "plugins"

        _write_plugin(
            user_plugins,
            ["category", "fancy"],
            manifest_extra={"kind": "backend"},
        )
        # Do NOT opt in.

        mgr = PluginManager()
        mgr.discover_and_load()

        loaded = mgr._plugins["category/fancy"]
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "")


    def test_exclusive_kind_skipped(self, tmp_path, monkeypatch):
        """``kind: exclusive`` plugins are recorded but not loaded — the
        category's own discovery system handles them (memory today)."""
        import os
        hermes_home = Path(os.environ["HERMES_HOME"])  # set by hermetic conftest fixture
        _write_plugin(
            hermes_home / "plugins",
            ["some-backend"],
            manifest_extra={"kind": "exclusive"},
        )
        _enable(hermes_home, "some-backend")

        mgr = PluginManager()
        mgr.discover_and_load()

        loaded = mgr._plugins["some-backend"]
        assert loaded.enabled is False
        assert "exclusive" in (loaded.error or "")
