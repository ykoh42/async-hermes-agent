"""Regression test for #44037 — holographic provider leaked its SQLite
connection to GC on shutdown instead of closing it.

The corruption-mechanism framing in #44037 (TLS bytes written into the DB via
an fd-recycle race) was not reproducible from the code: dropping a sqlite
connection flushes valid pages through SQLite's own VFS, never TLS framing, and
the provider is at most a *releaser* of DB fds, not the TLS-flushing owner.

But the underlying resource-hygiene bug is real and is what this test pins:
``HolographicMemoryProvider.shutdown()`` must call ``MemoryStore.close()`` so
the ``check_same_thread=False`` connection's fd is released deterministically
on shutdown, rather than at a non-deterministic GC time on an arbitrary thread.
"""

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


async def _make_provider(tmp_path):
    db_path = str(tmp_path / "memory_store.db")
    provider = HolographicMemoryProvider(config={"db_path": db_path, "hrr_dim": 64})
    await provider.initialize(session_id="test-session")
    return provider


@pytest.mark.asyncio
async def test_shutdown_closes_store_connection(tmp_path):
    provider = await _make_provider(tmp_path)
    store = provider._store
    assert store is not None
    conn = store._conn

    # Connection is live before shutdown.
    assert (await (await conn.execute("SELECT 1")).fetchone())[0] == 1

    await provider.shutdown()

    # References are dropped...
    assert provider._store is None
    assert provider._retriever is None

    # ...AND the underlying connection was actually closed (not left to GC).
    with pytest.raises(ValueError, match="no active connection"):
        await conn.execute("SELECT 1")
