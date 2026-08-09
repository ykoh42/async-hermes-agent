"""Real-process tests for native-async bundled secret sources."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path

import pytest

from agent.secret_sources._cache import CachedFetch, DiskCache
from agent.secret_sources.base import FetchResult, SecretSource
from agent.secret_sources.bitwarden import (
    _encrypted_disk_cache_path,
    _platform_asset_name,
    _read_encrypted_disk_cache,
    _write_encrypted_disk_cache,
    fetch_bitwarden_secrets,
)
from agent.secret_sources.onepassword import fetch_onepassword_secrets
from agent.secret_sources import registry
from hermes_cli.plugins import PluginContext, PluginManifest


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="these fake helper scripts are POSIX shell scripts"
)


def _write_executable(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.mark.asyncio
async def test_onepassword_fetch_uses_real_async_subprocess(tmp_path, monkeypatch):
    helper = _write_executable(
        tmp_path,
        "op",
        "sleep 0.05\n"
        'case "$5" in\n'
        "  op://Vault/OpenAI/key) printf 'sk-openai' ;;\n"
        "  op://Vault/Anthropic/key) printf 'sk-anthropic' ;;\n"
        "  *) exit 9 ;;\n"
        "esac",
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *args, **kwargs: pytest.fail("asyncio.to_thread was called"),
    )

    secrets, warnings = await fetch_onepassword_secrets(
        references={
            "OPENAI_API_KEY": "op://Vault/OpenAI/key",
            "ANTHROPIC_API_KEY": "op://Vault/Anthropic/key",
        },
        account="test-account",
        binary=helper,
        use_cache=False,
        home_path=tmp_path,
    )

    assert warnings == []
    assert secrets == {
        "ANTHROPIC_API_KEY": "sk-anthropic",
        "OPENAI_API_KEY": "sk-openai",
    }


@pytest.mark.asyncio
async def test_bitwarden_fetch_uses_real_async_subprocess(tmp_path, monkeypatch):
    helper = _write_executable(
        tmp_path,
        "bws",
        "sleep 0.05\n"
        "[ \"$BWS_ACCESS_TOKEN\" = 'test-token' ] || exit 8\n"
        "[ \"$BWS_SERVER_URL\" = 'https://vault.example' ] || exit 9\n"
        'printf \'%s\' \'[{"key":"OPENAI_API_KEY","value":"sk-bws"}]\'',
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *args, **kwargs: pytest.fail("asyncio.to_thread was called"),
    )

    secrets, warnings = await fetch_bitwarden_secrets(
        access_token="test-token",
        project_id="project-id",
        binary=helper,
        use_cache=False,
        server_url="https://vault.example",
        home_path=tmp_path,
    )

    assert warnings == []
    assert secrets == {"OPENAI_API_KEY": "sk-bws"}


@pytest.mark.asyncio
async def test_bitwarden_platform_probe_survives_repeated_cancellation(monkeypatch):
    from agent.secret_sources import bitwarden

    communicate_started = asyncio.Event()
    release_communicate = asyncio.Event()
    communicate_completed = asyncio.Event()

    class BlockingProcess:
        returncode = None
        killed = False

        async def communicate(self):
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(bitwarden.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bitwarden.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(_platform_asset_name())
    await communicate_started.wait()
    task.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_communicate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert communicate_completed.is_set()


@pytest.mark.asyncio
async def test_disk_cache_round_trip_is_awaitable_and_private(tmp_path):
    cache = DiskCache("test.json", key_serializer=str)
    entry = CachedFetch(secrets={"API_KEY": "secret"}, fetched_at=10**10)

    await cache.write("key", entry, 300, tmp_path)
    restored = await cache.read("key", 300, tmp_path)

    assert restored == entry
    assert stat.S_IMODE(cache.path(tmp_path).stat().st_mode) == 0o600
    await cache.clear(tmp_path)
    assert not cache.path(tmp_path).exists()


@pytest.mark.asyncio
async def test_bitwarden_encrypted_cache_round_trip(tmp_path):
    cache_key = ("fingerprint", "project", "https://vault.example")
    entry = CachedFetch(secrets={"OPENAI_API_KEY": "secret"}, fetched_at=time.time())
    await _write_encrypted_disk_cache(
        cache_key=cache_key,
        access_token="bootstrap-token",
        entry=entry,
        home_path=tmp_path,
    )

    restored = await _read_encrypted_disk_cache(
        cache_key=cache_key,
        access_token="bootstrap-token",
        max_age_seconds=300,
        home_path=tmp_path,
    )

    assert restored == entry
    cache_path = _encrypted_disk_cache_path(tmp_path)
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
    assert b"secret" not in cache_path.read_bytes()


def test_plugin_context_registers_native_async_secret_source(monkeypatch):
    registry._reset_registry_for_tests()
    monkeypatch.setattr(registry, "_ensure_builtin_sources", lambda: None)

    class PluginSource(SecretSource):
        name = "plugin_source"
        label = "Plugin source"

        async def fetch(self, cfg, home_path):
            return FetchResult(secrets={"PLUGIN_TOKEN": "value"})

    context = PluginContext(PluginManifest(name="test-plugin"), None)
    context.register_secret_source(PluginSource())
    assert registry.get_source("plugin_source") is not None
