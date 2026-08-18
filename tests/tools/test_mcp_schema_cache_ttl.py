"""SEP-2549 schema-cache TTL expiry (tools/mcp_schema_cache.py)."""

import time
import asyncio

import pytest

from tools import mcp_schema_cache as sc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_cache_path", lambda: tmp_path / "cache.json")
    yield


@pytest.mark.asyncio
async def test_entry_without_ttl_never_expires():
    await sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}])
    assert await sc.get_cached_entry("srv", "fp") is not None


@pytest.mark.asyncio
async def test_entry_within_ttl_served():
    await sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    entry = await sc.get_cached_entry("srv", "fp")
    assert entry is not None
    assert entry["ttl_ms"] == 60_000
    assert "written_at" in entry


@pytest.mark.asyncio
async def test_entry_past_ttl_is_a_miss(monkeypatch):
    await sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=1_000)
    real_time = time.time
    monkeypatch.setattr(sc.time, "time", lambda: real_time() + 2.0)
    assert await sc.get_cached_entry("srv", "fp") is None


@pytest.mark.asyncio
async def test_ttl_rewrite_advances_written_at():
    await sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    first = (await sc.get_cached_entry("srv", "fp"))["written_at"]
    await asyncio.sleep(0.01)
    # Identical payload would previously short-circuit; TTL'd entries must
    # rewrite so written_at advances on every live reconfirmation.
    await sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    second = (await sc.get_cached_entry("srv", "fp"))["written_at"]
    assert second > first


@pytest.mark.asyncio
async def test_cache_scope_round_trips():
    await sc.write_cache_entry(
        "srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000, cache_scope="private"
    )
    assert (await sc.get_cached_entry("srv", "fp"))["cache_scope"] == "private"
