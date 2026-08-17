"""Tests for the single-shape session_search tool.

Three calling shapes:
  1. DISCOVERY — pass query → FTS5 + anchored window + bookends per hit
  2. SCROLL    — pass session_id + around_message_id → just the window
  3. BROWSE    — no args → recent sessions chronologically

All run zero LLM calls.
"""
import json
import time

import pytest
import pytest_asyncio

from hermes_state import SessionDB
from tools.session_search_tool import (
    SESSION_SEARCH_SCHEMA,
    _format_timestamp,
    _is_compacted_message,
    _is_compression_ended,
    _resolve_to_parent,
    _session_link,
    session_search,
)


@pytest_asyncio.fixture
async def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    await store.close()


async def _execute_sql(db, sql, params=()):
    connection = await db._get_connection()
    await connection.execute(sql, params)
    await connection.commit()


async def _seed_modpack_sessions(db):
    """Create three sessions about a modpack so FTS5 has hits to dedupe."""
    now = int(time.time())
    # Older session — modpack origin
    await db.create_session("s_oldest", source="cli")
    await _execute_sql(
        db,
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 30000, "Building the Modpack", "s_oldest"),
    )
    await db.append_message("s_oldest", role="user", content="Let's build a Minecraft modpack")
    await db.append_message("s_oldest", role="assistant", content="Great. Let me scaffold the modpack repo.")
    await db.append_message("s_oldest", role="user", content="Use NeoForge 1.21.1")
    await db.append_message("s_oldest", role="assistant", content="Done. Modpack repo created with NeoForge 1.21.1.")
    await db.append_message("s_oldest", role="assistant", content="Tier-0 mods installed; modpack smoke test passes.")

    # Middle session — modpack quest coverage
    await db.create_session("s_middle", source="cli")
    await _execute_sql(
        db,
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 15000, "Modpack Quest Coverage", "s_middle"),
    )
    await db.append_message("s_middle", role="user", content="Deep-dive every modpack reference quest guide")
    await db.append_message("s_middle", role="assistant", content="Surveying ATM10 questbook for modpack inspiration.")
    await db.append_message("s_middle", role="user", content="Update the modpack version too")
    await db.append_message("s_middle", role="assistant", content="Modpack version bumped 0.4 → 0.8.5; quest coverage page added.")

    # Newest session — modpack mob spawn fix
    await db.create_session("s_newest", source="cli")
    await _execute_sql(
        db,
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 1000, "Modpack Mob Spawn Fix", "s_newest"),
    )
    await db.append_message("s_newest", role="user", content="Fix the modpack mob spawning")
    await db.append_message("s_newest", role="assistant", content="Investigating elite mob gating in the modpack KubeJS.")
    await db.append_message("s_newest", role="assistant", content="Shipped commit b850442. Modpack alternator nerfed too.")


# =========================================================================
# Schema invariants
# =========================================================================

class TestSchema:
    def test_schema_params_cover_every_shape(self):
        params = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
        # Discovery shape
        assert "query" in params
        assert "limit" in params
        assert params["sort"]["enum"] == ["newest", "oldest"]
        # Scroll shape
        assert "session_id" in params
        assert "around_message_id" in params
        assert "window" in params
        # Shared
        assert "role_filter" in params
        # Mode is inferred from which args are set — no explicit mode param
        assert "mode" not in params


class TestFormatTimestamp:
    def test_formats_unix_and_passes_through_the_rest(self):
        assert "2023" in _format_timestamp(1700000000)
        assert _format_timestamp(None) == "unknown"
        assert _format_timestamp("not-a-number-string") == "not-a-number-string"


# =========================================================================
# Browse shape (no args)
# =========================================================================

class TestBrowseShape:
    @pytest.mark.asyncio
    async def test_no_args_returns_recent_sessions(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(await session_search(db=db))
        assert result["success"] is True
        assert result["mode"] == "browse"
        assert result["count"] >= 3

    @pytest.mark.asyncio
    async def test_browse_excludes_current_session(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(await session_search(db=db, current_session_id="s_newest"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids

    @pytest.mark.asyncio
    async def test_closes_database_created_for_default_lookup(self, monkeypatch):
        import hermes_state

        class OwnedDatabase:
            closed = False

            def __init__(self, _path):
                pass

            async def list_sessions_rich(self, **_kwargs):
                return []

            async def close(self):
                self.closed = True

        database = OwnedDatabase("unused")
        monkeypatch.setattr(hermes_state, "SessionDB", lambda _path: database)

        result = json.loads(await session_search())

        assert result["success"] is True
        assert database.closed is True

    @pytest.mark.asyncio
    async def test_closes_cross_profile_database_but_not_borrowed_database(
        self, monkeypatch
    ):
        from tools import session_search_tool

        class Database:
            def __init__(self, *, session=None):
                self.closed = False
                self.session = session

            async def get_session(self, _session_id):
                return self.session

            async def get_messages(self, _session_id):
                return []

            async def close(self):
                self.closed = True

        borrowed = Database()
        profile_database = Database(
            session={"source": "cli", "model": "test", "title": "Profile session"}
        )

        async def resolve_profile_database(_profile):
            return profile_database

        monkeypatch.setattr(
            session_search_tool, "_resolve_profile_db", resolve_profile_database
        )

        result = json.loads(
            await session_search(db=borrowed, profile="work", session_id="profile-session")
        )

        assert result["success"] is True
        assert profile_database.closed is True
        assert borrowed.closed is False


# =========================================================================
# Discovery shape (with query)
# =========================================================================

class TestDiscoveryShape:
    @pytest.mark.asyncio
    async def test_discovery_result_has_bookends_and_window(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(await session_search(query="modpack", limit=3, db=db))
        assert result["success"] is True
        assert result["mode"] == "discover"
        assert result["count"] >= 1
        for hit in result["results"]:
            assert "bookend_start" in hit
            assert "messages" in hit
            assert "bookend_end" in hit
            assert "match_message_id" in hit
            assert "snippet" in hit
            assert "messages_before" in hit
            assert "messages_after" in hit

    @pytest.mark.asyncio
    async def test_full_detail_hydrates_every_hit_and_adaptive_marks_lower_hits(self, db):
        await _seed_modpack_sessions(db)
        full = json.loads(
            await session_search(query="modpack", limit=3, detail="full", db=db)
        )
        adaptive = json.loads(
            await session_search(query="modpack", limit=3, db=db)
        )

        assert full["detail"] == "full"
        assert all(hit["detail"] == "full" for hit in full["results"])
        assert adaptive["detail"] == "adaptive"
        assert adaptive["results"][0]["detail"] == "full"
        for hit in adaptive["results"][1:]:
            assert hit["detail"] == "compact"
            assert len(hit["messages"]) == 1
            assert hit["bookend_start"] == []
            assert hit["bookend_end"] == []


    @pytest.mark.asyncio
    async def test_current_session_filtered_out(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(
            await session_search(
                query="modpack", db=db, current_session_id="s_newest"
            )
        )
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids


class TestDiscoverySort:
    @pytest.mark.asyncio
    async def test_sort_newest_orders_by_recency(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(
            await session_search(
                query="modpack", limit=3, sort="newest", db=db
            )
        )
        # First result should be the most recent session
        first = result["results"][0]
        assert first["session_id"] == "s_newest" or "Newest" in (first.get("title") or "")

    @pytest.mark.asyncio
    async def test_sort_oldest_orders_by_age(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(
            await session_search(
                query="modpack", limit=3, sort="oldest", db=db
            )
        )
        first = result["results"][0]
        assert first["session_id"] == "s_oldest"


# =========================================================================
# Scroll shape (session_id + around_message_id)
# =========================================================================

class TestScrollShape:
    @pytest.mark.asyncio
    async def test_scroll_returns_anchored_window_without_bookends(self, db):
        await _seed_modpack_sessions(db)
        # Get an anchor first via discovery
        disc = json.loads(await session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]

        # Now scroll
        result = json.loads(
            await session_search(
                session_id=anchor_sid,
                around_message_id=anchor_mid,
                window=2,
                db=db,
            )
        )
        assert result["success"] is True
        assert result["mode"] == "scroll"
        # Scroll shape has no bookends
        assert "bookend_start" not in result
        assert "bookend_end" not in result
        # The anchor is in the window and flagged
        anchor_in_window = [m for m in result["messages"] if m["id"] == anchor_mid]
        assert len(anchor_in_window) == 1
        assert anchor_in_window[0].get("anchor") is True

    @pytest.mark.asyncio
    async def test_scroll_window_clamped_to_20(self, db):
        await _seed_modpack_sessions(db)
        disc = json.loads(await session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(
            await session_search(
                session_id=anchor_sid,
                around_message_id=anchor_mid,
                window=999,
                db=db,
            )
        )
        assert result["window"] == 20


    @pytest.mark.asyncio
    async def test_scroll_rejects_active_delegation_child_in_current_lineage(self, db):
        await db.create_session("s_current", source="cli")
        await db.create_session(
            "s_delegate", source="delegate", parent_session_id="s_current"
        )
        mid = await db.append_message(
            "s_delegate", role="assistant", content="live delegated result"
        )

        result = json.loads(
            await session_search(
                session_id="s_delegate",
                around_message_id=mid,
                db=db,
                current_session_id="s_current",
            )
        )

        assert result["success"] is False
        assert "current session" in result.get("error", "").lower()


class TestScrollPattern:
    """The forward/backward scroll loop using tool output."""

    @pytest.mark.asyncio
    async def test_scroll_forward_from_last_id(self, db):
        # Long session
        await db.create_session("s_long", source="cli")
        ids = []
        for i in range(20):
            ids.append(
                await db.append_message(
                    "s_long",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"long session msg {i}",
                )
            )

        v1 = json.loads(
            await session_search(
                session_id="s_long", around_message_id=ids[5], window=3, db=db
            )
        )
        last_id = v1["messages"][-1]["id"]
        v2 = json.loads(
            await session_search(
                session_id="s_long", around_message_id=last_id, window=3, db=db
            )
        )
        # Forward scroll: v2 should reach further than v1
        assert max(m["id"] for m in v2["messages"]) > max(m["id"] for m in v1["messages"])
        # Boundary id appears in both
        assert last_id in [m["id"] for m in v1["messages"]]
        assert last_id in [m["id"] for m in v2["messages"]]


# =========================================================================
# Shape precedence
# =========================================================================

class TestShapePrecedence:
    @pytest.mark.asyncio
    async def test_scroll_args_beat_query(self, db):
        await _seed_modpack_sessions(db)
        disc = json.loads(await session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        # Pass both query and scroll args — scroll should win
        result = json.loads(
            await session_search(
                query="modpack",  # would normally trigger discovery
                session_id=anchor_sid,
                around_message_id=anchor_mid,
                db=db,
            )
        )
        assert result["mode"] == "scroll"


    @pytest.mark.asyncio
    async def test_session_id_without_anchor_reads(self, db):
        await _seed_modpack_sessions(db)
        # session_id alone (no anchor, no query) → read shape, not browse.
        result = json.loads(await session_search(session_id="s_oldest", db=db))
        assert result["mode"] == "read"


# =========================================================================
# Read shape — dump a whole session by id (serves @session links)
# =========================================================================

class TestReadShape:
    @pytest.mark.asyncio
    async def test_read_returns_full_session(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(await session_search(session_id="s_oldest", db=db))
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["session_id"] == "s_oldest"
        assert result["message_count"] == 5
        assert result["truncated"] is False
        assert len(result["messages"]) == 5
        assert result["session_meta"]["title"] == "Building the Modpack"

    @pytest.mark.asyncio
    async def test_read_truncates_large_session(self, db):
        await db.create_session("s_big", source="cli")
        for i in range(50):
            await db.append_message(
                "s_big", role="user" if i % 2 == 0 else "assistant", content=f"m{i}"
            )
        result = json.loads(await session_search(session_id="s_big", db=db))
        assert result["mode"] == "read"
        assert result["message_count"] == 50
        assert result["truncated"] is True
        assert len(result["messages"]) == 30  # head 20 + tail 10


# =========================================================================
# Session links — the value the agent writes to point the user at a session
# =========================================================================

def _linked_session_id(link: str) -> str:
    """Recover the session id from an `@session:[<profile>/]<id>` value."""
    assert link.startswith("@session:"), link
    value = link[len("@session:"):]

    return value.rsplit("/", 1)[-1]


class TestSessionLink:
    @pytest.mark.asyncio
    async def test_link_carries_the_named_profile(self):
        assert await _session_link("s_oldest", "work") == "@session:work/s_oldest"


    @pytest.mark.asyncio
    async def test_every_discovery_result_links_to_its_own_session(self, db):
        await _seed_modpack_sessions(db)
        result = json.loads(await session_search(query="modpack", limit=5, db=db))

        assert result["results"]
        for entry in result["results"]:
            assert _linked_session_id(entry["link"]) == entry["session_id"]


# =========================================================================
# Cross-profile read — `profile` swaps in another profile's DB (read-only)
# =========================================================================

class TestCrossProfileRead:
    def _patch_profiles(self, monkeypatch, home):
        from hermes_cli import profiles as profiles_mod

        async def get_profile_dir(_name):
            return home

        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", get_profile_dir)

    @pytest.mark.asyncio
    async def test_bare_id_locates_across_profiles(self, db, tmp_path, monkeypatch):
        # The real-world failure: model dropped the owning profile and passed a
        # bare id. The tool must scan profiles and find it anyway.
        profiles_root = tmp_path / "profiles"
        other_home = profiles_root / "asdf"
        other_home.mkdir(parents=True)
        other = SessionDB(other_home / "state.db")
        await other.create_session("s_far", source="cli")
        await other.append_message("s_far", role="user", content="hi")
        await other.close()

        from hermes_cli import profiles as profiles_mod

        async def get_profiles_root():
            return profiles_root

        async def get_profile_dir(name):
            return other_home if name == "asdf" else tmp_path / "default_home"

        monkeypatch.setattr(profiles_mod, "_get_profiles_root", get_profiles_root)
        monkeypatch.setattr(
            profiles_mod,
            "get_profile_dir",
            get_profile_dir,
        )

        # `db` (current profile) lacks s_far; no profile passed → scan finds it.
        result = json.loads(await session_search(session_id="s_far", db=db))
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["profile"] == "asdf"


    @pytest.mark.asyncio
    async def test_combined_value_autosplits(self, db, tmp_path, monkeypatch):
        # Agent passed the raw "@session:<profile>/<id>" value as session_id with
        # no separate profile — the tool should recover both.
        other_home = tmp_path / "other_home"
        other_home.mkdir()
        other = SessionDB(other_home / "state.db")
        await other.create_session("s_other", source="cli")
        await other.append_message("s_other", role="user", content="hi")
        await other.close()

        self._patch_profiles(monkeypatch, other_home)

        # Every permutation the model might send must resolve to (asdf, s_other).
        for kwargs in (
            {"session_id": "asdf/s_other"},                    # full value, no profile
            {"session_id": "asdf/s_other", "profile": "asdf"},  # full value AND profile
            {"session_id": "s_other", "profile": "asdf"},       # bare id + profile
        ):
            result = json.loads(await session_search(db=db, **kwargs))
            assert result["success"] is True, kwargs
            assert result["mode"] == "read"
            assert result["session_id"] == "s_other"


# =========================================================================
# Cron demotion in discover ranking (#19434)
# =========================================================================

class TestCronDemotion:
    async def _seed_cron_and_interactive(self, db):
        """One interactive (telegram) session and several cron sessions, all
        matching the same query. Cron rows accumulate repetitive vocabulary
        and out-number the user's single interactive session — the live-data
        symptom in #19434.
        """
        now = int(time.time())
        # Interactive user session — older, so it loses on bare recency too.
        await db.create_session("s_user", source="telegram")
        await _execute_sql(
            db,
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (now - 90000, "s_user"),
        )
        await db.append_message("s_user", role="user", content="how is the venom project going")
        await db.append_message("s_user", role="assistant", content="The venom project shipped its first milestone.")
        # Several cron sessions, all newer and all stuffed with the same terms.
        for i in range(8):
            sid = f"cron_{i}"
            await db.create_session(sid, source="cron")
            await _execute_sql(
                db,
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (now - 1000 - i, sid),
            )
            await db.append_message(sid, role="user", content="venom project daily status")
            await db.append_message(sid, role="assistant", content="venom project venom project venom summary")

    @pytest.mark.asyncio
    async def test_interactive_session_surfaces_above_cron(self, db):
        await self._seed_cron_and_interactive(db)
        result = json.loads(
            await session_search(query="venom project", limit=1, db=db)
        )
        assert result["success"] is True
        assert result["count"] == 1
        # With cron drowning FTS, bare BM25/recency would return a cron_* hit.
        # Demotion must put the user's interactive session first.
        assert result["results"][0]["source"] == "telegram"
        assert result["results"][0]["session_id"] == "s_user"

    @pytest.mark.asyncio
    async def test_cron_still_reachable_when_only_match(self, db):
        """Demotion must not exclude cron — when only cron matches, it still
        comes back."""
        now = int(time.time())
        await db.create_session("cron_only", source="cron")
        await _execute_sql(
            db,
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (now - 500, "cron_only"),
        )
        await db.append_message("cron_only", role="user", content="quarterly archive sweep")
        await db.append_message("cron_only", role="assistant", content="Archive sweep complete.")
        result = json.loads(await session_search(query="archive sweep", db=db))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["source"] == "cron"


# =========================================================================
# Compaction summary filtering (#43175)
# =========================================================================

class TestCompactionSummaryFiltering:
    """session_search discovery must exclude compaction handoffs from bookends."""

    def test_is_compaction_summary_detects_prefix(self):
        from tools.session_search_tool import _is_compaction_summary
        assert _is_compaction_summary("[CONTEXT COMPACTION — REFERENCE ONLY] foo")
        assert _is_compaction_summary("[CONTEXT SUMMARY]: old summary")
        assert not _is_compaction_summary("Hello, how can I help?")
        assert not _is_compaction_summary("")
        assert not _is_compaction_summary(None)

    @pytest.mark.asyncio
    async def test_compaction_summary_excluded_from_bookend_start(self, db):
        """Compaction handoff in bookend_start position must be filtered out."""
        await db.create_session("s_compact", source="cli")
        # First message: a compaction handoff (should be filtered)
        await db.append_message(
            "s_compact",
            role="user",
            content="[CONTEXT COMPACTION — REFERENCE ONLY] "
            "Earlier turns were compacted into the summary below. "
            + "x" * 50000,
        )
        # Second message: normal user message
        await db.append_message("s_compact", role="user", content="Fix the zorgblat rendering bug")
        # Padding messages to push window away from session start (so bookend has room)
        for i in range(10):
            await db.append_message("s_compact", role="user", content=f"setup step {i}")
            await db.append_message("s_compact", role="assistant", content=f"setup done {i}")
        # Match target: uses a unique term so FTS5 anchors here, not at the start
        await db.append_message("s_compact", role="user", content="investigate the frobnitz mob spawning in KubeJS")
        await db.append_message("s_compact", role="assistant", content="I'll look into the frobnitz mob spawning issue.")
        # Tail messages
        for i in range(5):
            await db.append_message("s_compact", role="user", content=f"tail {i}")
            await db.append_message("s_compact", role="assistant", content=f"done tail {i}")

        result = json.loads(
            await session_search(query="frobnitz mob spawning", db=db, limit=1)
        )
        assert result["success"] is True
        assert len(result["results"]) >= 1
        entry = result["results"][0]
        # bookend_start must NOT contain the compaction handoff
        for msg in entry.get("bookend_start", []):
            assert "[CONTEXT COMPACTION" not in (msg.get("content") or "")
        # The normal message should still be present in bookend_start
        bookend_contents = [m.get("content", "") for m in entry.get("bookend_start", [])]
        assert any("zorgblat" in c for c in bookend_contents)


# =========================================================================
# Compression-aware discovery (#6256)
#
# After compression (in-place compaction or legacy rotation), pre-compaction
# content is no longer in the live context but MUST stay discoverable via
# session_search. The old code skipped any FTS hit on the current session or
# lineage, creating a "memory black hole". Delegation children must STAY
# excluded — their content is still visible to the parent agent.
# =========================================================================

class TestResolveToParent:
    """Unit tests for _resolve_to_parent's compression-aware tuple return."""

    @pytest.mark.asyncio
    async def test_legacy_rotation_detects_compression(self, db):
        """Parent ended with end_reason='compression', child has parent_session_id."""
        await db.create_session("s_parent", source="cli")
        await db.end_session("s_parent", "compression")
        await db.create_session("s_child", source="cli", parent_session_id="s_parent")
        root, has_compression = await _resolve_to_parent(db, "s_child")
        assert root == "s_parent"
        assert has_compression is True


    @pytest.mark.asyncio
    async def test_chain_with_mixed_edges(self, db):
        """Compression grandparent → parent → child (no end_reason on parent)."""
        await db.create_session("s_gp", source="cli")
        await db.end_session("s_gp", "compression")
        await db.create_session("s_p", source="cli", parent_session_id="s_gp")
        # s_p does NOT end with compression — but ancestor s_gp does
        await db.create_session("s_c", source="cli", parent_session_id="s_p")
        root, has_compression = await _resolve_to_parent(db, "s_c")
        assert root == "s_gp"
        assert has_compression is True


class TestIsCompactedMessage:
    """Unit tests for the _is_compacted_message helper."""

    @pytest.mark.asyncio
    async def test_active_message_returns_false(self, db):
        await db.create_session("s1", source="cli")
        mid = await db.append_message("s1", role="user", content="hello")
        assert await _is_compacted_message(db, mid) is False

    @pytest.mark.asyncio
    async def test_compacted_message_returns_true(self, db):
        await db.create_session("s1", source="cli")
        mid = await db.append_message("s1", role="user", content="archived content")
        await db.archive_and_compact("s1", [
            {"role": "assistant", "content": "compacted summary"},
        ])
        # mid is now active=0, compacted=1
        assert await _is_compacted_message(db, mid) is True


class TestInPlaceCompactionDiscovery:
    """In-place compaction: archived turns on the SAME session_id must be
    discoverable from the current session."""

    @pytest.mark.asyncio
    async def test_archived_content_discoverable_after_compaction(self, db):
        """The core regression: pre-compaction content on the current session
        must surface in discovery even though raw_sid == current_session_id."""
        await db.create_session("s_compact", source="cli")
        await db.append_message("s_compact", role="user",
                                content="The spectral phoenix only spawns during full moons")
        await db.append_message("s_compact", role="assistant",
                                content="Spectral phoenix requires moonstone bait")
        await db.archive_and_compact("s_compact", [
            {"role": "user", "content": "Summary: spectral phoenix discussed"},
            {"role": "assistant", "content": "Acknowledged spectral phoenix info"},
        ])

        result = json.loads(
            await session_search(
                query="spectral phoenix", db=db, current_session_id="s_compact"
            )
        )
        assert result["success"] is True
        assert result["count"] >= 1
        # The hit should be from the same session (archived rows)
        hit = result["results"][0]
        assert hit["session_id"] == "s_compact"

    @pytest.mark.asyncio
    async def test_live_content_still_filtered_on_current_session(self, db):
        """Non-compacted (active) content on the current session stays filtered."""
        await db.create_session("s_live", source="cli")
        await db.append_message("s_live", role="user", content="crystal golem farming route")
        result = json.loads(
            await session_search(
                query="crystal golem", db=db, current_session_id="s_live"
            )
        )
        assert result["count"] == 0


class TestLegacyRotationDiscovery:
    """Legacy rotation: parent session ended with end_reason='compression',
    child session created. Parent's pre-compaction content must be discoverable
    from the child."""

    @pytest.mark.asyncio
    async def test_compression_parent_discoverable_from_child(self, db):
        await db.create_session("s_parent", source="cli")
        await db.append_message("s_parent", role="user",
                                content="The void crystal mining requires diamond pickaxe")
        await db.append_message("s_parent", role="assistant",
                                content="Void crystal found in the deep caverns")
        await db.end_session("s_parent", "compression")

        await db.create_session("s_child", source="cli", parent_session_id="s_parent")
        await db.append_message("s_child", role="user", content="Continue void crystal work")

        result = json.loads(
            await session_search(
                query="void crystal", db=db, current_session_id="s_child"
            )
        )
        assert result["success"] is True
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_parent" in sids


class TestDelegationExclusion:
    """Delegation children (delegate_task) must STAY excluded — their content
    is still visible to the parent agent. parent_session_id is set but the
    parent does NOT have end_reason='compression'."""

    @pytest.mark.asyncio
    async def test_delegation_parent_excluded_from_child(self, db):
        """Child can see its own content but parent's live content stays
        excluded (it's in context via delegation)."""
        await db.create_session("s_parent", source="cli")
        await db.append_message("s_parent", role="user",
                                content="nebula deployment infrastructure setup")
        await db.append_message("s_parent", role="assistant",
                                content="Nebula deployment configured successfully")

        await db.create_session("s_child", source="cli", parent_session_id="s_parent")
        await db.append_message("s_child", role="user",
                                content="delegated nebula deployment subtask")

        result = json.loads(
            await session_search(
                query="nebula deployment", db=db, current_session_id="s_child"
            )
        )
        assert result["count"] == 0


# =========================================================================
# Both layers together: discovery scope (#63144) × bookend bounding (#69334)
#
# Compaction touches two independent layers of session_search:
#   1. Discovery scope — compaction-archived rows on the current session must
#      surface in discovery (this PR).
#   2. Content bounding — bookends must exclude generated compaction handoff
#      summaries and cap message content length (#43175 / #69334).
# A compacted session exercises both at once: its archived content is the FTS
# hit, while the compaction summary row it produced sits at the session tail,
# exactly where bookend_end is sampled.
# =========================================================================

class TestCompactionDiscoveryBothLayers:
    """Compacted-session content is discoverable AND its bookends still
    exclude compaction summaries / cap content length."""

    async def _seed_compacted_session(self, db):
        await db.create_session("s_both", source="cli")
        # Long normal opening — exercises the 1200-char bookend cap.
        await db.append_message("s_both", role="user",
                                content="Kick off the obsidian gateway migration. " + "o" * 5000)
        await db.append_message("s_both", role="assistant",
                                content="Starting the obsidian gateway migration plan.")
        # Padding so the anchored window doesn't swallow the bookends.
        for i in range(10):
            await db.append_message("s_both", role="user", content=f"migration step {i}")
            await db.append_message("s_both", role="assistant", content=f"migration step {i} done")
        # The FTS match target — will be archived by compaction below.
        await db.append_message("s_both", role="user",
                                content="the obsidian gateway needs a quartz keystone to activate")
        await db.append_message("s_both", role="assistant",
                                content="Noted: quartz keystone required for the obsidian gateway.")
        for i in range(5):
            await db.append_message("s_both", role="user", content=f"wrap-up {i}")
            await db.append_message("s_both", role="assistant", content=f"wrapped {i}")
        # Compact in place: everything above becomes active=0/compacted=1 and
        # the handoff summary is inserted as the new live tail.
        await db.archive_and_compact("s_both", [
            {"role": "user",
             "content": "[CONTEXT COMPACTION — REFERENCE ONLY] "
                        "Earlier turns were compacted into this summary. " + "s" * 50000},
            {"role": "assistant", "content": "Continuing after compaction."},
        ])

    @pytest.mark.asyncio
    async def test_archived_hit_surfaces_with_bounded_summary_free_bookends(self, db):
        await self._seed_compacted_session(db)

        result = json.loads(
            await session_search(
                query="quartz keystone", db=db, current_session_id="s_both"
            )
        )

        # Layer 1 — discovery scope: the archived (active=0, compacted=1)
        # content on the CURRENT session must surface.
        assert result["success"] is True
        assert result["count"] >= 1
        entry = result["results"][0]
        assert entry["session_id"] == "s_both"

        # Layer 2a — summary exclusion: the compaction handoff row sits at the
        # session tail (freshly inserted by archive_and_compact), exactly where
        # bookend_end samples — it must be filtered out.
        for msg in entry.get("bookend_start", []) + entry.get("bookend_end", []):
            assert "[CONTEXT COMPACTION" not in (msg.get("content") or "")

        # Layer 2b — content caps: bookends ≤1200 chars, window ≤4000 chars.
        for msg in entry.get("bookend_start", []) + entry.get("bookend_end", []):
            assert len(msg.get("content") or "") <= 1210
        for msg in entry.get("messages", []):
            assert len(msg.get("content") or "") <= 4010

        # The long-but-legitimate opening survives (capped, not dropped).
        bookend_contents = [m.get("content") or "" for m in entry.get("bookend_start", [])]
        assert any("obsidian gateway migration" in c for c in bookend_contents)


# =========================================================================
# Teknium review round 2: rewind exclusion + delegation-under-compression
# =========================================================================

class TestRewindExclusion:
    """Rewind/undo rows (active=0, compacted=0) must STAY hidden — only
    compaction archives (active=0, compacted=1) should surface."""

    @pytest.mark.asyncio
    async def test_compacted_messages_still_surface_alongside_rewind(self, db):
        """On the same session: compacted rows surface, rewind rows don't."""
        await db.create_session("s_mixed", source="cli")
        # Message that will be compacted
        await db.append_message("s_mixed", role="user",
                                content="compaction archived content beta")
        await db.archive_and_compact("s_mixed", [
            {"role": "assistant", "content": "Summary of beta"},
        ])
        # Now add a post-compaction message and rewind it
        mid2 = await db.append_message("s_mixed", role="user",
                                       content="rewound content gamma")
        await _execute_sql(
            db,
            "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?",
            (mid2,),
        )

        # Compacted content should be discoverable
        result_compact = json.loads(
            await session_search(
                query="compaction archived content beta",
                db=db,
                current_session_id="s_mixed",
            )
        )
        assert result_compact["count"] >= 1

        # Rewound content should NOT be discoverable
        result_rewind = json.loads(
            await session_search(
                query="rewound content gamma",
                db=db,
                current_session_id="s_mixed",
            )
        )
        assert result_rewind["count"] == 0


class TestCompressionEndedHelper:
    """Unit tests for _is_compression_ended."""

    @pytest.mark.asyncio
    async def test_compression_ended_session(self, db):
        await db.create_session("s1", source="cli")
        await db.end_session("s1", "compression")
        assert await _is_compression_ended(db, "s1") is True

    @pytest.mark.asyncio
    async def test_delegation_child_not_ended(self, db):
        """A delegation child under a compression continuation does NOT have
        end_reason='compression' itself."""
        await db.create_session("s_parent", source="cli")
        await db.end_session("s_parent", "compression")
        await db.create_session("s_continuation", source="cli", parent_session_id="s_parent")
        await db.create_session("s_delegate_child", source="cli", parent_session_id="s_continuation")
        assert await _is_compression_ended(db, "s_delegate_child") is False


class TestLegacyContinuationPlusDelegation:
    """Regression: a delegation child created under a compression continuation
    must stay excluded — its content is still live to the parent agent.
    Only the compression-ended ancestor's content should surface."""

    @pytest.mark.asyncio
    async def test_compression_parent_surfaces_but_delegate_child_excluded(self, db):
        """Setup: grandparent (compression) → parent (compression) → child
        (active, current session). A delegation grandchild is created under
        the parent. Searching from the child should find grandparent/parent
        content but NOT the delegation grandchild's content."""
        # Grandparent: compression-ended, has searchable content
        await db.create_session("s_gp", source="cli")
        await db.append_message("s_gp", role="user",
                                content="grandparent cosmic anomaly research data")
        await db.end_session("s_gp", "compression")

        # Parent: compression-ended continuation
        await db.create_session("s_p", source="cli", parent_session_id="s_gp")
        await db.append_message("s_p", role="user",
                                content="parent cosmic anomaly follow-up notes")
        await db.end_session("s_p", "compression")

        # Current session: active child
        await db.create_session("s_current", source="cli", parent_session_id="s_p")

        # Delegation child under s_p (not compression-ended)
        await db.create_session("s_delegate", source="cli", parent_session_id="s_p")
        await db.append_message("s_delegate", role="assistant",
                                content="delegated cosmic anomaly subtask results")

        result = json.loads(
            await session_search(
                query="cosmic anomaly", db=db, current_session_id="s_current"
            )
        )

        # Compression-ended ancestors should be discoverable
        sids = [r["session_id"] for r in result["results"]]
        assert "s_gp" in sids or "s_p" in sids

        # Delegation child must NOT appear
        assert "s_delegate" not in sids


@pytest.mark.asyncio
async def test_new_reset_predecessor_is_searchable_in_current_lineage(db):
    """A reset child shares lineage but does not carry the predecessor text."""
    await db.create_session("s_old", source="cli")
    await db.append_message(
        "s_old", role="user", content="legacy reset recovery phrase"
    )
    await db.end_session("s_old", "session_reset")
    await db.create_session(
        "s_new", source="cli", parent_session_id="s_old",
        model_config={"_reset_from": "s_old"},
    )

    result = json.loads(
        await session_search(
            query="legacy reset recovery", db=db, current_session_id="s_new"
        )
    )
    assert any(hit["session_id"] == "s_old" for hit in result["results"])


@pytest.mark.asyncio
async def test_branched_parent_remains_hidden_from_current_lineage(db):
    """A branch copies the transcript, so the parent remains in-context."""
    await db.create_session("s_parent", source="cli")
    await db.append_message(
        "s_parent", role="user", content="branched in context phrase"
    )
    await db.end_session("s_parent", "branched")
    await db.create_session(
        "s_branch", source="cli", parent_session_id="s_parent",
        model_config={"_branched_from": "s_parent"},
    )
    await db.append_message(
        "s_branch", role="user", content="branched in context phrase"
    )

    result = json.loads(
        await session_search(
            query="branched in context", db=db, current_session_id="s_branch"
        )
    )
    assert all(hit["session_id"] != "s_parent" for hit in result["results"])
