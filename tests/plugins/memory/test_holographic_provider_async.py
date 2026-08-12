"""Native-async lifecycle and integration coverage for holographic memory."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import stat

import pytest
import pytest_asyncio
import yaml
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.memory_manager import MemoryManager
from plugins.memory import load_memory_provider
from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.store import MemoryStore

pytestmark = pytest.mark.asyncio


async def test_public_method_signatures_match_upstream_with_async_only():
    expected_parameters = {
        HolographicMemoryProvider: {
            "is_available": ["self"],
            "save_config": ["self", "values", "hermes_home"],
            "get_config_schema": ["self"],
            "initialize": ["self", "session_id", "kwargs"],
            "system_prompt_block": ["self"],
            "prefetch": ["self", "query", "session_id"],
            "sync_turn": [
                "self",
                "user_content",
                "assistant_content",
                "session_id",
            ],
            "get_tool_schemas": ["self"],
            "handle_tool_call": ["self", "tool_name", "args", "kwargs"],
            "on_session_end": ["self", "messages"],
            "on_memory_write": ["self", "action", "target", "content"],
            "shutdown": ["self"],
        },
        MemoryStore: {
            "add_fact": ["self", "content", "category", "tags"],
            "search_facts": [
                "self",
                "query",
                "category",
                "min_trust",
                "limit",
            ],
            "update_fact": [
                "self",
                "fact_id",
                "content",
                "trust_delta",
                "tags",
                "category",
            ],
            "remove_fact": ["self", "fact_id"],
            "list_facts": ["self", "category", "min_trust", "limit"],
            "record_feedback": ["self", "fact_id", "helpful"],
            "rebuild_all_vectors": ["self", "dim"],
            "close": ["self"],
        },
    }

    for owner, methods in expected_parameters.items():
        for name, parameters in methods.items():
            method = getattr(owner, name)
            expected_async = name not in {
                "system_prompt_block",
                "get_config_schema",
                "get_tool_schemas",
            }
            assert inspect.iscoroutinefunction(method) is expected_async
            assert list(inspect.signature(method).parameters) == parameters


@pytest_asyncio.fixture(autouse=True)
async def _clean_shared_registry():
    for entry in list(MemoryStore._shared.values()):
        await entry["conn"].close()
    MemoryStore._shared.clear()
    yield
    leaked = list(MemoryStore._shared)
    for entry in list(MemoryStore._shared.values()):
        await entry["conn"].close()
    MemoryStore._shared.clear()
    assert not leaked, f"test leaked shared connections: {leaked}"


async def test_provider_discovery_manager_tools_and_restart(tmp_path):
    loaded = await load_memory_provider("holographic")
    assert loaded is not None
    assert loaded.name == "holographic"

    manager = MemoryManager()
    manager.add_provider(loaded)
    assert manager.has_tool("fact_store")
    assert manager.has_tool("fact_feedback")

    await manager.initialize_all(
        "session-one",
        hermes_home=str(tmp_path),
        platform="service",
    )
    expected_empty_prompt = (
        "# Holographic Memory\n"
        "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
        "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
        "Use fact_feedback to rate facts after using them (trains trust scores)."
    )
    assert manager.build_system_prompt() == expected_empty_prompt

    added = json.loads(
        await manager.handle_tool_call(
            "fact_store",
            {
                "action": "add",
                "content": "Alice Example prefers GraphQL for Project Atlas.",
                "category": "user_pref",
                "tags": "alice,graphql",
            },
        )
    )
    assert added == {"fact_id": 1, "status": "added"}
    assert "Active. 1 facts stored" in manager.build_system_prompt()

    recalled = await manager.prefetch_all("What does Alice prefer?")
    assert "Alice Example prefers GraphQL" in recalled

    feedback = json.loads(
        await manager.handle_tool_call(
            "fact_feedback",
            {"action": "helpful", "fact_id": added["fact_id"]},
        )
    )
    assert feedback["old_trust"] == 0.5
    assert feedback["new_trust"] == 0.55
    await manager.shutdown_all()

    resumed = HolographicMemoryProvider(config={})
    await resumed.initialize("session-two", hermes_home=str(tmp_path))
    try:
        assert "Active. 1 facts stored" in resumed.system_prompt_block()
        listed = json.loads(
            await resumed.handle_tool_call(
                "fact_store",
                {"action": "list", "limit": 10},
            )
        )
        assert listed["count"] == 1
        assert listed["facts"][0]["content"] == (
            "Alice Example prefers GraphQL for Project Atlas."
        )
        removed = json.loads(
            await resumed.handle_tool_call(
                "fact_store",
                {"action": "remove", "fact_id": added["fact_id"]},
            )
        )
        assert removed == {"removed": True}
        assert resumed.system_prompt_block() == expected_empty_prompt
    finally:
        await resumed.shutdown()


async def test_shared_store_updates_are_visible_in_both_provider_prompts(tmp_path):
    config = {"db_path": str(tmp_path / "shared.db")}
    first = HolographicMemoryProvider(config=config)
    second = HolographicMemoryProvider(config=config)
    await asyncio.gather(
        first.initialize("first", hermes_home=str(tmp_path)),
        second.initialize("second", hermes_home=str(tmp_path)),
    )
    try:
        added = json.loads(
            await first.handle_tool_call(
                "fact_store",
                {"action": "add", "content": "Shared prompt count fact."},
            )
        )
        assert "Active. 1 facts stored" in first.system_prompt_block()
        assert "Active. 1 facts stored" in second.system_prompt_block()

        await second.handle_tool_call(
            "fact_store",
            {"action": "remove", "fact_id": added["fact_id"]},
        )
        assert "Empty fact store" in first.system_prompt_block()
        assert "Empty fact store" in second.system_prompt_block()
    finally:
        await asyncio.gather(first.shutdown(), second.shutdown())


async def test_concurrent_profiles_load_their_own_config_and_database(tmp_path):
    alpha_home = tmp_path / "alpha"
    beta_home = tmp_path / "beta"
    alpha_home.mkdir()
    beta_home.mkdir()
    (alpha_home / "config.yaml").write_text(
        "plugins:\n  hermes-memory-store:\n    default_trust: 0.7\n",
        encoding="utf-8",
    )
    (beta_home / "config.yaml").write_text(
        "plugins:\n  hermes-memory-store:\n    default_trust: 0.2\n",
        encoding="utf-8",
    )

    alpha = HolographicMemoryProvider()
    beta = HolographicMemoryProvider()
    await asyncio.gather(
        alpha.initialize("alpha", hermes_home=str(alpha_home)),
        beta.initialize("beta", hermes_home=str(beta_home)),
    )
    try:
        assert alpha._store.default_trust == 0.7
        assert beta._store.default_trust == 0.2
        assert alpha._store.db_path == alpha_home / "memory_store.db"
        assert beta._store.db_path == beta_home / "memory_store.db"

        await asyncio.gather(
            alpha.handle_tool_call(
                "fact_store",
                {"action": "add", "content": "Alpha profile fact."},
            ),
            beta.handle_tool_call(
                "fact_store",
                {"action": "add", "content": "Beta profile fact."},
            ),
        )
        alpha_facts, beta_facts = await asyncio.gather(
            alpha._store.list_facts(),
            beta._store.list_facts(),
        )
        assert [fact["content"] for fact in alpha_facts] == [
            "Alpha profile fact."
        ]
        assert [fact["content"] for fact in beta_facts] == [
            "Beta profile fact."
        ]
    finally:
        await asyncio.gather(alpha.shutdown(), beta.shutdown())


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode bits")
async def test_save_config_preserves_symlink_mode_and_unrelated_values(tmp_path):
    managed = tmp_path / "managed"
    profile = tmp_path / "profile"
    managed.mkdir()
    profile.mkdir()
    target = managed / "config.yaml"
    target.write_text("model:\n  default: existing/model\n", encoding="utf-8")
    target.chmod(0o640)
    link = profile / "config.yaml"
    link.symlink_to(target)

    provider = HolographicMemoryProvider(config={})
    await provider.save_config(
        {"default_trust": "0.75", "auto_extract": "true"},
        str(profile),
    )

    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {
        "model": {"default": "existing/model"},
        "plugins": {
            "hermes-memory-store": {
                "auto_extract": "true",
                "default_trust": "0.75",
            }
        },
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits")
async def test_save_config_creates_owner_only_file(tmp_path):
    provider = HolographicMemoryProvider(config={})
    await provider.save_config({"default_trust": "0.5"}, str(tmp_path))
    assert stat.S_IMODE((tmp_path / "config.yaml").stat().st_mode) == 0o600


async def test_save_config_cancellation_preserves_original_and_cleans_temp(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "config.yaml"
    original = "model:\n  default: existing/model\n"
    config_path.write_text(original, encoding="utf-8")
    replace_started = asyncio.Event()
    never_release = asyncio.Event()

    async def paused_replace(source, target):
        replace_started.set()
        await never_release.wait()

    monkeypatch.setattr("plugins.memory.holographic.aiofiles.os.replace", paused_replace)
    provider = HolographicMemoryProvider(config={})
    task = asyncio.create_task(
        provider.save_config({"default_trust": "0.8"}, str(tmp_path))
    )
    await replace_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert config_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".config_*.tmp")) == []


async def test_initialize_cancellation_closes_connection_and_registry(
    monkeypatch,
    tmp_path,
):
    started = asyncio.Event()
    never_release = asyncio.Event()

    async def paused_init(self):
        started.set()
        await never_release.wait()

    monkeypatch.setattr(MemoryStore, "_init_db", paused_init)
    store = MemoryStore(tmp_path / "cancel.db")
    task = asyncio.create_task(store._initialize())
    await started.wait()
    connection = next(iter(MemoryStore._shared.values()))["conn"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert MemoryStore._shared == {}
    assert store._entry is None
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("SELECT 1")


async def test_cancelled_write_finishes_before_reraising(
    monkeypatch,
    tmp_path,
):
    store = MemoryStore(tmp_path / "write-cancel.db", hrr_dim=64)
    await store._initialize()
    rebuild_started = asyncio.Event()
    release_rebuild = asyncio.Event()
    rebuild_completed = asyncio.Event()
    original_rebuild = MemoryStore._rebuild_bank

    async def paused_rebuild(self, connection, category):
        rebuild_started.set()
        await release_rebuild.wait()
        await original_rebuild(self, connection, category)
        rebuild_completed.set()

    monkeypatch.setattr(MemoryStore, "_rebuild_bank", paused_rebuild)
    task = asyncio.create_task(store.add_fact("Cancellation-safe fact."))
    await rebuild_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release_rebuild.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert rebuild_completed.is_set()
    assert store._current_fact_count() == 1
    assert [fact["content"] for fact in await store.list_facts()] == [
        "Cancellation-safe fact."
    ]
    await store.close()


async def test_close_finishes_owned_connection_before_reraising_cancellation(
    monkeypatch,
    tmp_path,
):
    store = MemoryStore(tmp_path / "close-cancel.db")
    await store._initialize()
    connection = store._conn
    original_close = connection.close
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def paused_close():
        close_started.set()
        await release_close.wait()
        await original_close()

    monkeypatch.setattr(connection, "close", paused_close)
    task = asyncio.create_task(store.close())
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert MemoryStore._shared == {}
    assert store._entry is None
    assert store._conn is None
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("SELECT 1")


async def test_native_lifecycle_does_not_block_or_leak_tasks(tmp_path):
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        provider = HolographicMemoryProvider(config={"hrr_dim": 64})
        await provider.initialize("service-session", hermes_home=str(tmp_path))
        added = json.loads(
            await provider.handle_tool_call(
                "fact_store",
                {
                    "action": "add",
                    "content": "Service lifecycle remembers Project Aurora.",
                },
            )
        )
        await provider.prefetch("Project Aurora", session_id="service-session")
        await provider.on_memory_write(
            "add",
            "user",
            "The user prefers concise lifecycle reports.",
        )
        await provider.handle_tool_call(
            "fact_feedback",
            {"action": "helpful", "fact_id": added["fact_id"]},
        )
        await provider.shutdown()
