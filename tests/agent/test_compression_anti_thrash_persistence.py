"""Anti-thrash state must survive process restarts (#54923).

The guard in ``_automatic_compression_blocked_locally()`` trips only after two
consecutive compactions that fail to bring the real prompt under the
threshold. Historically ``_ineffective_compression_count`` was in-memory only:
a fresh compressor bound to a resumed (already-compacted) session started at
``compression_count == 0`` with a disarmed guard, so a near-threshold session
could legally re-compact once per process restart, forever — the exact
residual @lanyusea identified in #54923.

The counter now round-trips the durable session-state channel exactly like
``compression_failure_cooldown_until`` (#54465) and the fallback streak
(af7dceaf7):

* every compression boundary persists the latest verdict to the session row,
* the async turn prologue hydrates a resumed compressor with an armed (1) or
  tripped (2) guard,
* the reset semantics are unchanged — a real provider reading below the
  threshold still clears the counter (update_from_response), and that clear
  is durable too.
"""
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import (
    _hydrate_persisted_compression_guards,
    _persist_compression_guards,
)
from hermes_state import SessionDB


async def _compressor(
    db: SessionDB | None = None, session_id: str = ""
) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_static_context_length",
        return_value=100_000,
    ):
        cc = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    if db is not None:
        cc.bind_session_state(db, session_id)
        await _hydrate_persisted_compression_guards(cc, db, session_id)
    return cc


def _db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


class TestCounterRoundTripsBindSessionState:
    @pytest.mark.asyncio
    async def test_fresh_compressor_inherits_tripped_guard_after_restart(
        self, tmp_path
    ):
        """A restart must not disarm a tripped anti-thrash breaker."""
        db = _db(tmp_path)
        await db.create_session("s1", source="cli")

        first = await _compressor(db, "s1")
        # Two compactions that failed to clear the threshold — judged on the
        # provider's real prompt counts, exactly as conversation_loop drives it.
        for _ in range(2):
            first._verify_compaction_cleared_threshold = True
            first.update_from_response({"prompt_tokens": first.threshold_tokens + 1})
        await _persist_compression_guards(first, db, "s1")
        assert first._ineffective_compression_count == 2
        assert first.should_compress(10**9) is False

        # Process restart: a brand-new compressor binds the same session.
        second = await _compressor(db, "s1")
        assert second.compression_count == 0  # the #54923 precondition
        assert second._ineffective_compression_count == 2
        assert second.should_compress(10**9) is False, (
            "a fresh compressor on a resumed session must inherit the "
            "tripped anti-thrash guard instead of re-compacting"
        )
        await db.close()


    @pytest.mark.asyncio
    async def test_rebind_to_other_session_does_not_leak_counter(self, tmp_path):
        """The counter is per-session: switching sessions must not carry it."""
        db = _db(tmp_path)
        await db.create_session("hot", source="cli")
        await db.create_session("cold", source="cli")
        await db.set_compression_ineffective_count("hot", 2)

        cc = await _compressor(db, "hot")
        assert cc._ineffective_compression_count == 2

        cc.bind_session_state(db, "cold")
        await _hydrate_persisted_compression_guards(cc, db, "cold")
        assert cc._ineffective_compression_count == 0
        await db.close()



class TestResetSemanticsPreserved:
    @pytest.mark.asyncio
    async def test_real_dip_below_threshold_clears_counter_durably(self, tmp_path):
        """The L1466-1474 contract survives: any real provider reading below
        the threshold clears the latch — and now clears the durable copy, so
        a restart cannot resurrect voided strikes."""
        db = _db(tmp_path)
        await db.create_session("s1", source="cli")

        cc = await _compressor(db, "s1")
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        assert cc._ineffective_compression_count == 1
        await _persist_compression_guards(cc, db, "s1")
        assert await db.get_compression_ineffective_count("s1") == 1

        # An ordinary fitting response (not post-compaction) clears the latch.
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens - 1})
        assert cc._ineffective_compression_count == 0
        await _persist_compression_guards(cc, db, "s1")
        assert await db.get_compression_ineffective_count("s1") == 0

        # And a restart sees the cleared state.
        fresh = await _compressor(db, "s1")
        assert fresh._ineffective_compression_count == 0
        assert fresh.should_compress(10**9) is True
        await db.close()

    @pytest.mark.asyncio
    async def test_post_compaction_clearing_reading_resets_durably(self, tmp_path):
        """The post-compaction success verdict (real tokens under threshold)
        also zeroes the durable strike count."""
        db = _db(tmp_path)
        await db.create_session("s1", source="cli")

        cc = await _compressor(db, "s1")
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        await _persist_compression_guards(cc, db, "s1")
        assert await db.get_compression_ineffective_count("s1") == 1

        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens - 1})
        assert cc._ineffective_compression_count == 0
        await _persist_compression_guards(cc, db, "s1")
        assert await db.get_compression_ineffective_count("s1") == 0
        await db.close()

    @pytest.mark.asyncio
    async def test_update_model_reset_persists_at_boundary(self, tmp_path):
        """update_model() voids strikes judged against the old threshold; the
        durable copy must not resurrect them on the next restart."""
        db = _db(tmp_path)
        await db.create_session("s1", source="cli")

        cc = await _compressor(db, "s1")
        await db.set_compression_ineffective_count("s1", 2)
        cc._ineffective_compression_count = 2

        cc.update_model("other/model", 100_000)
        await _persist_compression_guards(cc, db, "s1")

        assert cc._ineffective_compression_count == 0
        assert await db.get_compression_ineffective_count("s1") == 0
        await db.close()


class TestStrikesPersistAtAsyncBoundary:
    @pytest.mark.asyncio
    async def test_no_op_compaction_branch_persists_at_boundary(self, tmp_path):
        """The insufficient-messages no-op branch records its strike durably."""
        db = _db(tmp_path)
        await db.create_session("s1", source="cli")

        cc = await _compressor(db, "s1")
        # 3 tiny messages < minimum window → the #40803 no-op branch.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = await cc.compress(msgs, current_tokens=10**9)
        await _persist_compression_guards(cc, db, "s1")
        assert out == msgs
        assert cc._ineffective_compression_count == 1
        assert await db.get_compression_ineffective_count("s1") == 1
        await db.close()

    @pytest.mark.asyncio
    async def test_persist_failure_is_swallowed_and_memory_still_advances(
        self, tmp_path
    ):
        """A DB write failure must not break the in-memory guard."""
        db = _db(tmp_path)
        await db.create_session("s1", source="cli")
        cc = await _compressor(db, "s1")

        with patch.object(
            db,
            "set_compression_ineffective_count",
            new_callable=AsyncMock,
            side_effect=Exception("disk full"),
        ):
            cc._verify_compaction_cleared_threshold = True
            cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
            await _persist_compression_guards(cc, db, "s1")

        assert cc._ineffective_compression_count == 1
        assert await db.get_compression_ineffective_count("s1") == 0
        await db.close()


class TestCompressionBoundaryCarry:
    @pytest.mark.asyncio
    async def test_rotation_boundary_carries_counter_onto_child_row(self, tmp_path):
        """Session-id rotation must not launder an armed guard through the
        fresh child row (same carry contract as the fallback streak)."""
        db = _db(tmp_path)
        await db.create_session("parent", source="cli")
        cc = await _compressor(db, "parent")
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        await _persist_compression_guards(cc, db, "parent")
        assert await db.get_compression_ineffective_count("parent") == 1

        await db.create_session("child", source="cli", parent_session_id="parent")
        await cc.on_session_start(
            "child",
            boundary_reason="compression",
            old_session_id="parent",
            session_db=db,
        )
        await _persist_compression_guards(cc, db, "child")

        assert cc._session_id == "child"
        assert cc._ineffective_compression_count == 1
        # Persisted onto the child row so a restart right after rotation
        # still inherits the armed guard.
        assert await db.get_compression_ineffective_count("child") == 1
        await db.close()
