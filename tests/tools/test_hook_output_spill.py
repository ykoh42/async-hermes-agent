"""Tests for native-async plugin hook output spilling."""

from __future__ import annotations

import json

import pytest

from tools import hook_output_spill as spill

pytestmark = pytest.mark.asyncio


def _config(root, **overrides):
    config = {
        "enabled": True,
        "max_chars": 100,
        "preview_head": 20,
        "preview_tail": 20,
        "directory": str(root),
    }
    config.update(overrides)
    return config


async def test_under_cap_and_disabled_are_unchanged(tmp_path):
    assert await spill.spill_if_oversized(
        "small", config=_config(tmp_path)
    ) == "small"
    text = "x" * 200
    assert await spill.spill_if_oversized(
        text, config=_config(tmp_path, enabled=False)
    ) == text


async def test_oversized_text_is_saved_with_bounded_preview(tmp_path):
    text = "head" + "x" * 200 + "tail"
    preview = await spill.spill_if_oversized(
        text,
        session_id="session/unsafe",
        source="plugin hook",
        config=_config(tmp_path),
    )
    output_dir = tmp_path / "session_unsafe"
    files = list(output_dir.iterdir())
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == text + "\n"
    assert "plugin hook output truncated" in preview
    assert str(files[0]) in preview
    assert text[:20] in preview
    assert text[-20:] in preview
    assert "x" * 100 not in preview


async def test_write_failure_still_returns_bounded_preview(monkeypatch, tmp_path):
    async def fail_makedirs(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(spill.aiofiles.os, "makedirs", fail_makedirs)
    text = "x" * 200
    preview = await spill.spill_if_oversized(
        text,
        config=_config(tmp_path),
    )
    assert "spill write failed" in preview
    assert len(preview) < len(text)


async def test_config_uses_async_loader(monkeypatch):
    async def load_config():
        return {
            "hooks": {
                "output_spill": {
                    "max_chars": 42,
                    "preview_head": 3,
                    "preview_tail": 4,
                    "directory": "/tmp/hooks",
                }
            }
        }

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", load_config)
    config = await spill.get_spill_config()
    assert json.dumps(config, sort_keys=True)
    assert config["max_chars"] == 42
    assert config["preview_head"] == 3
    assert config["preview_tail"] == 4
