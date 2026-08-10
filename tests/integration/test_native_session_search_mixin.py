"""Native-async and structural parity coverage for SessionSearchMixin."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB
from hermes_state_search import SessionSearchMixin


_UPSTREAM_METHOD_ORDER = (
    "_search_message_fields",
    "_try_incremental_merge_fts",
    "fts_rebuild_status",
    "_fts_rebuild_finish",
    "_fts_teardown_trash_step",
    "fts_rebuild_step",
    "fts_cjk_rebuild_status",
    "fts_cjk_rebuild_step",
    "_fts_cjk_rebuild_finish",
    "_fts_cjk_reset_if_stale",
    "_fts_external_index_empty_with_messages",
    "_fts_index_known_empty",
    "_reset_fts_index_to_empty",
    "_seed_fts_rebuild_markers",
    "_repair_optimize_bookkeeping",
    "fts_optimize_available",
    "_demote_legacy_fts_to_trash",
    "optimize_fts_storage",
    "get_anchored_view",
    "list_recent_user_messages",
    "_sanitize_fts5_query",
    "_is_cjk_codepoint",
    "_contains_cjk",
    "_count_cjk",
    "_has_lone_cjk_run",
    "_trigram_eligible_tokens",
    "_run_trigram_search",
    "search_messages",
    "_describe_search_path",
    "_search_messages_impl",
    "_search_unindexed_gap",
    "search_sessions_by_id",
    "_fts_table_exists",
    "optimize_fts",
    "rebuild_fts",
    "_merge_fts_incrementally",
)

_SYNC_METHODS = {
    "_search_message_fields",
    "_sanitize_fts5_query",
    "_is_cjk_codepoint",
    "_contains_cjk",
    "_count_cjk",
    "_has_lone_cjk_run",
    "_trigram_eligible_tokens",
    "_describe_search_path",
}


def _defined_methods(cls):
    return tuple(
        name
        for name, value in cls.__dict__.items()
        if inspect.isfunction(value)
        or isinstance(value, (classmethod, staticmethod))
    )


def test_search_mixin_preserves_upstream_structure_and_api_shape():
    assert SessionDB.__mro__[1] is SessionSearchMixin
    assert _defined_methods(SessionSearchMixin) == _UPSTREAM_METHOD_ORDER
    assert not set(_UPSTREAM_METHOD_ORDER).intersection(SessionDB.__dict__)

    for name in _UPSTREAM_METHOD_ORDER:
        is_async = inspect.iscoroutinefunction(getattr(SessionSearchMixin, name))
        assert is_async is (name not in _SYNC_METHODS), name

    namespace = SessionSearchMixin.__dict__
    assert isinstance(namespace["_search_message_fields"], classmethod)
    assert isinstance(namespace["_is_cjk_codepoint"], staticmethod)
    assert isinstance(namespace["_contains_cjk"], staticmethod)
    assert isinstance(namespace["_count_cjk"], classmethod)
    assert isinstance(namespace["_has_lone_cjk_run"], classmethod)
    assert isinstance(namespace["_trigram_eligible_tokens"], staticmethod)

    assert tuple(
        inspect.signature(namespace["_fts_external_index_empty_with_messages"])
        .parameters
    ) == ("self", "conn")
    assert tuple(
        inspect.signature(namespace["_seed_fts_rebuild_markers"]).parameters
    ) == ("self", "conn", "force")
    assert tuple(
        inspect.signature(namespace["_is_cjk_codepoint"].__func__).parameters
    ) == ("cp",)


@pytest.mark.asyncio
async def test_search_mixin_real_sqlite_path_does_not_block_or_leak(tmp_path):
    database = SessionDB(tmp_path / "search.db")
    await database.create_session("session", source="cli")
    message_id = await database.append_message(
        "session",
        role="user",
        content="saffron nebula search marker",
    )
    if not database._fts_enabled:
        await database.close()
        pytest.skip("SQLite build has no FTS5")

    blocker = BlockBuster()
    async with no_task_leaks(action=LeakAction.RAISE):
        blocker.activate()
        try:
            results = await database.search_messages(
                "saffron nebula",
                fields=("snippet", "role", "session_id", "id"),
            )
            assert len(results) == 1
            assert list(results[0]) == ["id", "session_id", "role", "snippet"]
            assert results[0]["id"] == message_id
            assert results[0]["session_id"] == "session"
            assert results[0]["role"] == "user"

            view = await database.get_anchored_view(
                "session",
                message_id,
                window=1,
                bookend=1,
            )
            assert any(row["id"] == message_id for row in view["window"])
        finally:
            await database.close()
            blocker.deactivate()


@pytest.mark.asyncio
async def test_cancelled_search_propagates_and_instance_remains_usable(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "cancelled-search.db")
    await database.create_session("session", source="cli")
    await database.append_message(
        "session",
        role="user",
        content="cancel propagation marker",
    )
    if not database._fts_enabled:
        await database.close()
        pytest.skip("SQLite build has no FTS5")

    read_started = asyncio.Event()
    original_read_fetchall = database._read_fetchall

    async def _stalled_read(sql, params=()):
        read_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(database, "_read_fetchall", _stalled_read)
    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(database.search_messages("cancel marker"))
        await asyncio.wait_for(read_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(database, "_read_fetchall", original_read_fetchall)
    try:
        results = await database.search_messages("cancel marker")
        assert len(results) == 1
        assert results[0]["session_id"] == "session"
    finally:
        await database.close()
