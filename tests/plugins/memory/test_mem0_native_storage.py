"""Parity tests for the native-async Mem0 OSS SQLite state."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from plugins.memory.mem0._native_oss import SQLiteManager


@pytest.mark.asyncio
async def test_sqlite_manager_initializes_lazily(tmp_path):
    path = tmp_path / "history.db"
    manager = SQLiteManager(str(path))

    assert path.exists() is False

    assert await manager.get_history("missing") == []
    assert path.exists() is True
    await manager.close()


@pytest.mark.asyncio
async def test_sqlite_manager_preserves_history_result_shape(tmp_path):
    manager = SQLiteManager(str(tmp_path / "history.db"))
    await manager.add_history(
        "m1",
        None,
        "likes tea",
        "ADD",
        created_at="2026-08-09T00:00:00+00:00",
        actor_id="alice",
        role="user",
    )
    await manager.add_history(
        "m1",
        "likes tea",
        "likes coffee",
        "UPDATE",
        created_at="2026-08-09T00:00:01+00:00",
        updated_at="2026-08-09T00:00:02+00:00",
        is_deleted=1,
    )

    rows = await manager.get_history("m1")

    assert [{key: row[key] for key in row if key != "id"} for row in rows] == [
        {
            "memory_id": "m1",
            "old_memory": None,
            "new_memory": "likes tea",
            "event": "ADD",
            "created_at": "2026-08-09T00:00:00+00:00",
            "updated_at": None,
            "is_deleted": False,
            "actor_id": "alice",
            "role": "user",
        },
        {
            "memory_id": "m1",
            "old_memory": "likes tea",
            "new_memory": "likes coffee",
            "event": "UPDATE",
            "created_at": "2026-08-09T00:00:01+00:00",
            "updated_at": "2026-08-09T00:00:02+00:00",
            "is_deleted": True,
            "actor_id": None,
            "role": None,
        },
    ]
    assert all(isinstance(row["id"], str) and row["id"] for row in rows)
    await manager.close()


@pytest.mark.asyncio
async def test_sqlite_manager_batch_history_matches_single_insert(tmp_path):
    manager = SQLiteManager(str(tmp_path / "history.db"))
    await manager.batch_add_history(
        [
            {
                "memory_id": "m1",
                "old_memory": None,
                "new_memory": "one",
                "event": "ADD",
                "created_at": "2026-08-09T00:00:00+00:00",
            },
            {
                "memory_id": "m2",
                "old_memory": None,
                "new_memory": "two",
                "event": "ADD",
                "created_at": "2026-08-09T00:00:01+00:00",
                "is_deleted": 1,
            },
        ]
    )

    assert (await manager.get_history("m1"))[0]["new_memory"] == "one"
    assert (await manager.get_history("m2"))[0]["is_deleted"] is True
    await manager.close()


@pytest.mark.asyncio
async def test_sqlite_manager_keeps_only_ten_latest_messages(tmp_path):
    manager = SQLiteManager(str(tmp_path / "history.db"))
    for index in range(12):
        await manager.save_messages(
            [{"role": "user", "content": f"message-{index}", "name": "alice"}],
            "user:u1|agent:a1",
        )

    rows = await manager.get_last_messages("user:u1|agent:a1", 10)

    assert len(rows) == 10
    assert {row["content"] for row in rows} == {
        f"message-{index}" for index in range(2, 12)
    }
    assert all(row["role"] == "user" for row in rows)
    assert all(row["name"] == "alice" for row in rows)
    assert all(row["created_at"] for row in rows)
    await manager.close()


@pytest.mark.asyncio
async def test_sqlite_manager_migrates_legacy_history_table(tmp_path):
    path = tmp_path / "history.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE history (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            old_memory TEXT,
            new_memory TEXT,
            event TEXT,
            created_at DATETIME,
            conversation_id TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("h1", "m1", None, "fact", "ADD", "2026-08-09", "legacy"),
    )
    connection.commit()
    connection.close()

    manager = SQLiteManager(str(path))
    rows = await manager.get_history("m1")

    assert rows[0]["id"] == "h1"
    assert rows[0]["new_memory"] == "fact"
    await manager.close()

    connection = sqlite3.connect(path)
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(history)").fetchall()
    }
    connection.close()
    assert columns == {
        "id",
        "memory_id",
        "old_memory",
        "new_memory",
        "event",
        "created_at",
        "updated_at",
        "is_deleted",
        "actor_id",
        "role",
    }


@pytest.mark.asyncio
async def test_sqlite_manager_reset_and_close_are_deterministic(tmp_path):
    path = tmp_path / "history.db"
    manager = SQLiteManager(str(path))
    await manager.add_history("m1", None, "fact", "ADD")

    await manager.reset()
    connection = sqlite3.connect(path)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    connection.close()
    assert "history" not in tables
    assert "messages" not in tables

    await manager.close()
    await manager.close()

    with pytest.raises(RuntimeError, match="closed SQLiteManager"):
        await manager.reset()


@pytest.mark.asyncio
async def test_sqlite_manager_close_finishes_cleanup_before_reraising_cancellation():
    entered = asyncio.Event()
    release = asyncio.Event()

    class FakeConnection:
        def __init__(self):
            self.closed = False

        async def close(self):
            entered.set()
            await release.wait()
            self.closed = True

    connection = FakeConnection()
    manager = SQLiteManager(":memory:")
    manager.connection = connection
    task = asyncio.create_task(manager.close())
    await entered.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert connection.closed is True
    assert manager.connection is None
