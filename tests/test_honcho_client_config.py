"""Tests for Honcho client configuration."""

import json
import os
import stat
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho import HonchoMemoryProvider


class TestHonchoClientConfigAutoEnable:
    """Test auto-enable behavior when API key is present."""

    async def test_auto_enables_when_api_key_present_no_explicit_enabled(self, tmp_path):
        """When API key exists and enabled is not set, should auto-enable."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            # Note: no "enabled" field
        }))

        cfg = await HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is True  # Auto-enabled because API key exists

    async def test_respects_explicit_enabled_false(self, tmp_path):
        """When enabled is explicitly False, should stay disabled even with API key."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            "enabled": False,  # Explicitly disabled
        }))

        cfg = await HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is False  # Respects explicit setting


    async def test_disabled_when_no_api_key_and_no_explicit_enabled(self, tmp_path):
        """When no API key and enabled not set, should be disabled."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "workspace": "test",
            # No apiKey, no enabled
        }))

        # Clear env var if set
        env_key = os.environ.pop("HONCHO_API_KEY", None)
        try:
            cfg = await HonchoClientConfig.from_global_config(config_path=config_path)
            assert cfg.api_key is None
            assert cfg.enabled is False  # No API key = not enabled
        finally:
            if env_key:
                os.environ["HONCHO_API_KEY"] = env_key


    async def test_from_env_always_enabled(self, monkeypatch):
        """from_env() should always set enabled=True."""
        monkeypatch.setenv("HONCHO_API_KEY", "env-test-key")

        cfg = await HonchoClientConfig.from_env()

        assert cfg.api_key == "env-test-key"
        assert cfg.enabled is True



@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
@pytest.mark.asyncio
async def test_save_config_sets_owner_only_permissions(tmp_path):
    """honcho.json is created atomically with 0o600, not chmod-after-write."""
    provider = HonchoMemoryProvider()
    await provider.save_config({"api_key": "hc-test-key"}, str(tmp_path))
    config_file = tmp_path / "honcho.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"


class TestLatencyFlagResolution:

    async def test_host_block_wins(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_BASE_URL', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'queryRewrite': False,
            'firstTurnBaseWait': 3,
            'hosts': {'hermes': {
                'queryRewrite': True,
                'firstTurnBaseWait': 0,
                'firstTurnDialecticWait': 0.5,
            }},
        }))
        cfg = await HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.query_rewrite is True
        assert cfg.first_turn_base_wait == 0.0
        assert cfg.first_turn_dialectic_wait == 0.5

    async def test_per_host_timeout_wins_over_global(self, tmp_path, monkeypatch):
        monkeypatch.delenv('HONCHO_TIMEOUT', raising=False)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps({
            'apiKey': 'k',
            'timeout': 30,
            'hosts': {'hermes': {'timeout': 5}},
        }))
        cfg = await HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.timeout == 5.0
