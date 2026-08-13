"""Unit tests for the on-disk MCP schema cache (tools/mcp_schema_cache.py).

The module landed in #56832's extraction without its tests; these cover the
fingerprint keying, read/write round-trip, and invalidation behavior.
"""

import asyncio
import gc
import weakref

import pytest

import tools.mcp_schema_cache as msc


class TestConfigFingerprint:
    def test_stable_for_same_config(self):
        cfg = {"command": "npx", "args": ["-y", "@playwright/mcp"]}
        assert msc.config_fingerprint(cfg) == msc.config_fingerprint(dict(cfg))

    def test_changes_when_connection_config_changes(self):
        base = {"command": "npx", "args": ["-y", "@playwright/mcp"]}
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "args": ["-y", "@playwright/mcp", "--headless"]}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "command": "uvx"}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "tools": {"include": ["a"]}}
        )

    def test_ignores_non_connection_keys(self):
        base = {"command": "npx", "args": []}
        assert msc.config_fingerprint(base) == msc.config_fingerprint(
            {**base, "timeout": 5, "enabled": True, "lazy": True}
        )


class TestCacheRoundTrip:
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")

    @pytest.mark.asyncio
    async def test_write_then_read_with_matching_fingerprint(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        tools = [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]
        await msc.write_cache_entry("srv", "fp1", tools=tools, utility_tools=[])
        entry = await msc.get_cached_entry("srv", "fp1")
        assert entry is not None
        assert msc.tools_from_cache_entry(entry) == tools
        assert msc.utility_tools_from_cache_entry(entry) == []
        assert await msc.has_cached_entry("srv", "fp1")

    @pytest.mark.asyncio
    async def test_fingerprint_mismatch_returns_none(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        await msc.write_cache_entry("srv", "fp1", tools=[], utility_tools=[])
        assert await msc.get_cached_entry("srv", "OTHER") is None
        assert not await msc.has_cached_entry("srv", "OTHER")

    @pytest.mark.asyncio
    async def test_missing_server_returns_none(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        assert await msc.get_cached_entry("nope", "fp") is None

    @pytest.mark.asyncio
    async def test_clear_cache_entry(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        await msc.write_cache_entry("srv", "fp1", tools=[], utility_tools=[])
        await msc.clear_cache_entry("srv")
        assert await msc.get_cached_entry("srv", "fp1") is None

    @pytest.mark.asyncio
    async def test_corrupt_cache_file_is_tolerated(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / "cache.json").write_text("{not json", encoding="utf-8")
        assert await msc.get_cached_entry("srv", "fp") is None
        # And writes recover the file.
        await msc.write_cache_entry("srv", "fp", tools=[], utility_tools=[])
        assert await msc.has_cached_entry("srv", "fp")

    def test_malformed_entry_shapes_are_tolerated(self):
        assert msc.tools_from_cache_entry({"tools": "nope"}) == []
        assert msc.utility_tools_from_cache_entry({}) == []


class TestCacheFileLocation:
    @pytest.mark.asyncio
    async def test_cache_lives_under_hermes_home_cache_dir_with_0600(
        self, monkeypatch, tmp_path
    ):
        # Real path (no _cache_path monkeypatch): HERMES_HOME/cache/…, 0o600,
        # matching the discovery-cache precedent in tools/registry.py.
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        path = msc._cache_path()
        assert path == tmp_path / "cache" / "mcp_schema_cache.json"
        await msc.write_cache_entry("srv", "fp", tools=[], utility_tools=[])
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600


class TestWriteSkip:
    @pytest.mark.asyncio
    async def test_identical_payload_skips_rewrite(self, monkeypatch, tmp_path):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        saves = []
        real_save = msc._save_all

        async def _counting_save(data):
            saves.append(1)
            await real_save(data)

        monkeypatch.setattr(msc, "_save_all", _counting_save)
        tools = [{"name": "t1", "description": "d", "inputSchema": {}}]
        await msc.write_cache_entry("srv", "fp1", tools=tools, utility_tools=[])
        assert len(saves) == 1
        # Identical payload (reconnect / list_changed refresh) → no rewrite.
        await msc.write_cache_entry("srv", "fp1", tools=list(tools), utility_tools=[])
        assert len(saves) == 1
        # Changed payload → rewrite.
        await msc.write_cache_entry("srv", "fp2", tools=tools, utility_tools=[])
        assert len(saves) == 2


class TestCacheLockLifecycle:
    def test_sequential_event_loops_do_not_reuse_lock(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        loop_refs = []
        lock_refs = []

        async def use_lock():
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            lock = await msc._cache_lock()
            lock_refs.append(weakref.ref(lock))
            async with lock:
                await asyncio.sleep(0)

        asyncio.run(use_lock())
        asyncio.run(use_lock())
        gc.collect()

        assert loop_refs[0]() is None
        assert lock_refs[0]() is None

    @pytest.mark.asyncio
    async def test_symlink_aliases_share_one_cache_lock(
        self, monkeypatch, tmp_path
    ):
        real_home = tmp_path / "real"
        alias_home = tmp_path / "alias"
        real_home.mkdir()
        alias_home.symlink_to(real_home, target_is_directory=True)
        active_path = real_home / "cache.json"
        monkeypatch.setattr(msc, "_cache_path", lambda: active_path)

        first = await msc._cache_lock()
        active_path = alias_home / "cache.json"
        second = await msc._cache_lock()

        assert first is second
