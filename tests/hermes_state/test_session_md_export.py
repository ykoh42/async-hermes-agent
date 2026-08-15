import time

import pytest

import hermes_state
from hermes_state import SessionDB

pytestmark = pytest.mark.asyncio


async def test_export_candidates_via_prune_filters_ended_old_sessions(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_state.time, "time", lambda: 2_000_000.0)
    try:
        await db.create_session("old_cli", source="cli")
        await db.end_session("old_cli", "done")
        connection = await db._get_connection()
        await connection.execute("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_000_000.0, 1_000_010.0, "old_cli"))
        await connection.commit()

        await db.create_session("new_cli", source="cli")
        await db.end_session("new_cli", "done")
        await connection.execute("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_990_000.0, 1_990_010.0, "new_cli"))
        await connection.commit()

        await db.create_session("old_active", source="cli")
        await connection.execute("UPDATE sessions SET started_at=? WHERE id=?", (1_000_000.0, "old_active"))
        await connection.commit()

        # Export uses the shared prune/archive candidate selection.
        candidates = await db.list_prune_candidates(
            started_before=2_000_000.0 - 5 * 86400, archived=None
        )
        assert [c["id"] for c in candidates] == ["old_cli"]
    finally:
        await db.close()




async def test_get_compression_lineage_returns_only_compression_chain(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await db.create_session("root", source="cli")
        await db.end_session("root", "compression")
        await db.create_session("child", source="cli", parent_session_id="root")
        await db.end_session("child", "compression")
        await db.create_session("tip", source="cli", parent_session_id="child")
        await db.create_session("branch", source="cli", parent_session_id="root", model_config={"_branched_from": "root"})
        await db.create_session("delegate", source="delegate", parent_session_id="child", model_config={"_delegate_from": "child"})
        await db.create_session("tool", source="tool", parent_session_id="child")

        assert await db.get_compression_lineage("tip") == ["root", "child", "tip"]
        assert await db.get_compression_lineage("branch") == ["branch"]
        assert await db.get_compression_lineage("delegate") == ["delegate"]
        assert await db.get_compression_lineage("tool") == ["tool"]
    finally:
        await db.close()


async def test_fork_children_created_before_continuation_do_not_hijack_lineage(tmp_path):
    # Regression: the forward walk used to accept any non-branch child as the
    # compression continuation. A delegate/tool child spawned BEFORE the real
    # continuation row (the common runtime ordering — the subagent exists
    # before compression rotates the session) was picked as the successor,
    # so lineage and session .md export followed the subagent's transcript.
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await db.create_session("root", source="cli")
        await db.append_message("root", role="user", content="root msg")
        await db.create_session(
            "delegate",
            source="delegate",
            parent_session_id="root",
            model_config={"_delegate_from": "root"},
        )
        await db.append_message("delegate", role="user", content="delegate private msg")
        await db.end_session("root", "compression")
        await db.create_session("continuation", source="cli", parent_session_id="root")
        await db.append_message("continuation", role="user", content="continuation msg")

        await db.create_session("root2", source="cli")
        await db.create_session("toolchild", source="tool", parent_session_id="root2")
        await db.end_session("root2", "compression")
        await db.create_session("cont2", source="cli", parent_session_id="root2")

        assert await db.get_compression_lineage("root") == ["root", "continuation"]
        assert await db.get_compression_lineage("continuation") == ["root", "continuation"]
        assert await db.get_compression_lineage("root2") == ["root2", "cont2"]

        exported = await db.export_session_lineage("root")
        assert exported is not None
        assert exported["lineage_session_ids"] == ["root", "continuation"]
        contents = [
            m.get("content")
            for seg in exported["segments"]
            for m in (seg.get("messages") or [])
        ]
        assert contents == ["root msg", "continuation msg"]
    finally:
        await db.close()
