"""Tests for the holographic MemoryStore shared-connection registry.

MemoryStore instances pointing at the same database file must share one
loop-local SQLite connection and one async lock. Multiple providers coexist in
a single process (the main agent plus delegate_task subagents); independent
writers must never race into ``database is locked`` failures.
"""

import asyncio

import pytest
import pytest_asyncio

from plugins.memory.holographic.store import MemoryStore

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _clean_shared_registry():
    """Each test starts and ends with an empty shared-connection registry."""
    for entry in list(MemoryStore._shared.values()):
        await entry["conn"].close()
    MemoryStore._shared.clear()
    yield
    leaked = list(MemoryStore._shared)
    for entry in list(MemoryStore._shared.values()):
        await entry["conn"].close()
    MemoryStore._shared.clear()
    assert not leaked, f"test leaked shared connections: {leaked}"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory_store.db"


class TestSharedConnection:
    async def test_same_path_shares_one_connection(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        await asyncio.gather(a._initialize(), b._initialize())
        try:
            assert a._conn is b._conn
            assert a._lock is b._lock
            assert len(MemoryStore._shared) == 1
            assert MemoryStore._shared[a._registry_key]["refs"] == 2
        finally:
            await a.close()
            await b.close()

    async def test_different_paths_get_distinct_connections(self, tmp_path):
        a = MemoryStore(tmp_path / "one.db")
        b = MemoryStore(tmp_path / "two.db")
        await asyncio.gather(a._initialize(), b._initialize())
        try:
            assert a._conn is not b._conn
            assert len(MemoryStore._shared) == 2
        finally:
            await a.close()
            await b.close()

    async def test_symlinked_path_shares_connection(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)

        a = MemoryStore(real_dir / "memory_store.db")
        b = MemoryStore(link_dir / "memory_store.db")
        await asyncio.gather(a._initialize(), b._initialize())
        try:
            assert a._conn is b._conn
            assert len(MemoryStore._shared) == 1
        finally:
            await a.close()
            await b.close()

    async def test_writes_visible_across_instances(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        await asyncio.gather(a._initialize(), b._initialize())
        try:
            fact_id = await a.add_fact(
                "Hermes likes shared connections",
                category="test",
            )
            facts = await b.list_facts(category="test")
            assert [fact["fact_id"] for fact in facts] == [fact_id]
        finally:
            await a.close()
            await b.close()

    async def test_schema_initialised_once_per_connection(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        await a._initialize()
        await b._initialize()
        try:
            assert MemoryStore._shared[a._registry_key]["ready"] is True
            await b.add_fact("schema still works")
        finally:
            await a.close()
            await b.close()


class TestCloseSemantics:
    async def test_closing_one_instance_keeps_sibling_alive(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        await asyncio.gather(a._initialize(), b._initialize())
        await a.close()
        try:
            assert await b.add_fact("survivor write") > 0
        finally:
            await b.close()

    async def test_last_close_releases_connection(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        await asyncio.gather(a._initialize(), b._initialize())
        conn = a._conn
        await a.close()
        await b.close()
        assert MemoryStore._shared == {}
        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("SELECT 1")

    async def test_close_is_idempotent(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        await asyncio.gather(a._initialize(), b._initialize())
        await a.close()
        await a.close()
        try:
            await b.add_fact("still alive after double close")
            assert MemoryStore._shared[b._registry_key]["refs"] == 1
        finally:
            await b.close()

    async def test_context_manager_releases_reference(self, db_path):
        async with MemoryStore(db_path) as store:
            await store.add_fact("context managed")
        assert MemoryStore._shared == {}

    async def test_reopen_after_full_close(self, db_path):
        async with MemoryStore(db_path) as store:
            await store.add_fact("first lifetime")
        async with MemoryStore(db_path) as store:
            facts = await store.list_facts()
        assert [fact["content"] for fact in facts] == ["first lifetime"]


class TestConcurrency:
    async def test_concurrent_multi_instance_writers(self, db_path):
        """Concurrent instances share one FIFO lock and preserve all writes."""
        writer_count, fact_count = 8, 15

        async def writer(index: int) -> None:
            store = MemoryStore(db_path)
            await store._initialize()
            try:
                for sequence in range(fact_count):
                    await store.add_fact(
                        f"fact writer={index} seq={sequence}",
                        category="load",
                    )
            finally:
                await store.close()

        await asyncio.gather(*(writer(index) for index in range(writer_count)))

        async with MemoryStore(db_path) as store:
            facts = await store.list_facts(category="load", limit=500)
        assert len(facts) == writer_count * fact_count
        assert MemoryStore._shared == {}

    async def test_failed_write_does_not_pin_write_lock(self, db_path, monkeypatch):
        broken = MemoryStore(db_path)
        sibling = MemoryStore(db_path)
        await asyncio.gather(broken._initialize(), sibling._initialize())

        async def fail_rebuild(self, conn, category):
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        try:
            monkeypatch.setattr(MemoryStore, "_rebuild_bank", fail_rebuild)
            with pytest.raises(RuntimeError, match="boom"):
                await broken.add_fact("write that fails after the INSERT")
            monkeypatch.undo()

            assert broken._conn.in_transaction is False
            await sibling.add_fact("sibling write right after the failure")
        finally:
            await broken.close()
            await sibling.close()


class TestProviderShutdown:
    async def test_shutdown_releases_shared_connection(self, db_path):
        from plugins.memory.holographic import HolographicMemoryProvider

        provider = HolographicMemoryProvider(config={"db_path": str(db_path)})
        await provider.initialize("session-shutdown")
        assert MemoryStore._shared[provider._store._registry_key]["refs"] == 1

        await provider.shutdown()

        assert provider._store is None
        assert MemoryStore._shared == {}


class TestStoreBehavior:
    async def test_duplicate_add_returns_existing_id_without_modifying_row(
        self,
        db_path,
    ):
        async with MemoryStore(db_path, default_trust=0.6) as store:
            first = await store.add_fact(
                "Alice Example uses Python.",
                category="project",
                tags="python",
            )
            duplicate = await store.add_fact(
                "  Alice Example uses Python.  ",
                category="tool",
                tags="changed",
            )
            facts = await store.list_facts()

            assert duplicate == first
            assert store._current_fact_count() == 1
            assert len(facts) == 1
            assert facts[0]["category"] == "project"
            assert facts[0]["tags"] == "python"
            assert facts[0]["trust_score"] == 0.6

    async def test_empty_add_and_missing_mutations_preserve_store(self, db_path):
        async with MemoryStore(db_path) as store:
            with pytest.raises(ValueError, match="content must not be empty"):
                await store.add_fact("   ")
            assert await store.update_fact(999, content="missing") is False
            assert await store.remove_fact(999) is False
            with pytest.raises(KeyError, match="fact_id 999 not found"):
                await store.record_feedback(999, helpful=True)
            assert store._current_fact_count() == 0

    async def test_search_filters_and_increments_retrieval_count(self, db_path):
        async with MemoryStore(db_path) as store:
            project_id = await store.add_fact(
                "Project Atlas deploys with Kubernetes.",
                category="project",
            )
            await store.add_fact(
                "Kubernetes shell aliases are configured.",
                category="tool",
            )
            results = await store.search_facts(
                "Kubernetes",
                category="project",
            )
            assert [result["fact_id"] for result in results] == [project_id]
            assert results[0]["retrieval_count"] == 0

            listed = await store.list_facts(category="project")
            assert listed[0]["retrieval_count"] == 1
            assert await store.search_facts("") == []

    async def test_update_feedback_and_trust_clamping(self, db_path):
        async with MemoryStore(db_path, default_trust=0.5) as store:
            fact_id = await store.add_fact(
                "Alice Example uses Vim.",
                category="user_pref",
            )
            assert await store.update_fact(
                fact_id,
                content="Alice Example uses Neovim.",
                trust_delta=2.0,
                tags="editor",
                category="tool",
            ) is True
            fact = (await store.list_facts())[0]
            assert fact["content"] == "Alice Example uses Neovim."
            assert fact["category"] == "tool"
            assert fact["tags"] == "editor"
            assert fact["trust_score"] == 1.0

            helpful = await store.record_feedback(fact_id, helpful=True)
            assert helpful == {
                "fact_id": fact_id,
                "old_trust": 1.0,
                "new_trust": 1.0,
                "helpful_count": 1,
            }
            unhelpful = await store.record_feedback(fact_id, helpful=False)
            assert unhelpful == {
                "fact_id": fact_id,
                "old_trust": 1.0,
                "new_trust": 0.9,
                "helpful_count": 1,
            }

    async def test_remove_cleans_links_and_updates_shared_count(self, db_path):
        async with MemoryStore(db_path) as store:
            fact_id = await store.add_fact("Alice Example works on Project Atlas.")
            conn, lock = await store._ready()
            async with lock:
                links_before = await conn.execute_fetchall(
                    "SELECT * FROM fact_entities WHERE fact_id = ?",
                    (fact_id,),
                )
            assert links_before

            assert await store.remove_fact(fact_id) is True
            assert store._current_fact_count() == 0
            async with lock:
                links_after = await conn.execute_fetchall(
                    "SELECT * FROM fact_entities WHERE fact_id = ?",
                    (fact_id,),
                )
            assert links_after == []
