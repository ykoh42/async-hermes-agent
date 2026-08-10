"""E2E parity coverage for native-async SessionDB token accounting."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB


def test_token_pipeline_preserves_upstream_method_shape():
    for name in (
        "_execute_write",
        "_sleep_before_write_retry",
        "_insert_session_row",
        "_token_writer_loop",
        "_apply_token_batch",
        "_stop_token_writer",
        "_drain_token_queue_at_exit",
        "_record_model_usage",
    ):
        assert inspect.iscoroutinefunction(getattr(SessionDB, name)), name
    assert not inspect.iscoroutinefunction(SessionDB._coalesce_token_deltas)
    assert not hasattr(SessionDB, "_write")
    assert SessionDB._WRITE_PATIENCE_S == 20.0
    assert SessionDB._TRANSCRIPT_WRITE_PATIENCE_S == 60.0


def test_coalesce_preserves_routes_costs_and_absolute_barriers(tmp_path):
    database = SessionDB(tmp_path / "coalesce.db")
    route = {
        "model": "model-a",
        "billing_provider": "provider-a",
        "billing_base_url": "https://provider-a.test/v1",
        "billing_mode": "api_key",
        "cost_status": "estimated",
        "cost_source": "catalog",
        "pricing_version": "v1",
    }
    batch = [
        (
            "session",
            {
                **route,
                "input_tokens": 2,
                "output_tokens": 1,
                "estimated_cost_usd": None,
                "api_call_count": 1,
            },
        ),
        (
            "session",
            {
                **route,
                "input_tokens": 3,
                "output_tokens": 4,
                "estimated_cost_usd": 0.25,
                "api_call_count": 1,
            },
        ),
        (
            "session",
            {
                **route,
                "absolute": True,
                "input_tokens": 100,
                "output_tokens": 20,
                "api_call_count": 1,
            },
        ),
        (
            "session",
            {
                **route,
                "input_tokens": 7,
                "output_tokens": 5,
                "estimated_cost_usd": None,
                "api_call_count": 1,
            },
        ),
    ]

    coalesced = database._coalesce_token_deltas(batch)

    assert len(coalesced) == 3
    assert coalesced[0][1]["input_tokens"] == 5
    assert coalesced[0][1]["output_tokens"] == 5
    assert coalesced[0][1]["api_call_count"] == 2
    assert coalesced[0][1]["estimated_cost_usd"] == 0.25
    assert coalesced[1][1]["absolute"] is True
    assert coalesced[2][1]["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_slow_write_retry_jitter_yields_to_the_event_loop(tmp_path):
    database = SessionDB(tmp_path / "retry-jitter.db")
    database._WRITE_RETRY_SLOW_AFTER_S = -1.0
    database._WRITE_RETRY_SLOW_MIN_S = 0.05
    database._WRITE_RETRY_SLOW_MAX_S = 0.05
    heartbeats = 0

    retry = asyncio.create_task(
        database._sleep_before_write_retry(
            asyncio.get_running_loop().time() + 1.0,
            1.0,
        )
    )
    while not retry.done():
        heartbeats += 1
        await asyncio.sleep(0.005)

    assert await retry is True
    assert heartbeats >= 5
    await database.close()


@pytest.mark.asyncio
async def test_queue_coalesces_before_native_writer_and_leaves_no_task(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "queued.db")
    applied = []

    async def _record(session_id, **kwargs):
        await asyncio.sleep(0)
        applied.append((session_id, kwargs))

    monkeypatch.setattr(database, "update_token_counts", _record)
    async with no_task_leaks(action=LeakAction.RAISE):
        await database.queue_token_counts(
            "session",
            model="model-a",
            input_tokens=2,
            api_call_count=1,
        )
        await database.queue_token_counts(
            "session",
            model="model-a",
            input_tokens=3,
            api_call_count=1,
        )
        assert await database.flush_token_counts() is True
        assert database._token_writer_task is None
        await database.queue_token_counts(
            "session",
            model="model-a",
            input_tokens=7,
            api_call_count=1,
        )
        assert await database.flush_token_counts() is True
        await database.close()

    assert len(applied) == 2
    assert applied[0][0] == "session"
    assert applied[0][1]["input_tokens"] == 5
    assert applied[0][1]["api_call_count"] == 2
    assert applied[1][1]["input_tokens"] == 7
    assert applied[1][1]["api_call_count"] == 1
    assert database._token_writer_task is None


@pytest.mark.asyncio
async def test_first_accounted_route_and_per_model_rows_match_upstream(tmp_path):
    database = SessionDB(tmp_path / "routes.db")
    await database.create_session(
        "session",
        source="library",
        model="requested-model",
    )
    await database.update_session_billing_route(
        "session",
        provider="requested-provider",
        base_url="https://requested.test/v1",
        billing_mode="api_key",
    )

    blocker = BlockBuster()
    async with no_task_leaks(action=LeakAction.RAISE):
        blocker.activate()
        try:
            await database.queue_token_counts(
                "session",
                model="fallback-model",
                billing_provider="fallback-provider",
                billing_base_url="https://fallback.test/v1",
                billing_mode="api_key",
                input_tokens=11,
                output_tokens=3,
                api_call_count=1,
            )
            await database.queue_token_counts(
                "session",
                model="fallback-model",
                billing_provider="fallback-provider",
                billing_base_url="https://fallback.test/v1",
                billing_mode="api_key",
                input_tokens=5,
                output_tokens=2,
                api_call_count=1,
            )
            assert await database.flush_token_counts() is True

            await database.queue_token_counts(
                "session",
                model="later-model",
                billing_provider="later-provider",
                billing_base_url="https://later.test/v1",
                billing_mode="oauth",
                input_tokens=7,
                output_tokens=1,
                api_call_count=1,
            )
            assert await database.flush_token_counts() is True

            session = await database.get_session("session")
            assert session["model"] == "fallback-model"
            assert session["billing_provider"] == "fallback-provider"
            assert session["input_tokens"] == 23
            assert session["output_tokens"] == 6
            assert session["api_call_count"] == 3

            rows = await database._read_fetchall(
                "SELECT model, billing_provider, api_call_count, "
                "input_tokens, output_tokens FROM session_model_usage "
                "WHERE session_id = ? AND task = '' ORDER BY model",
                ("session",),
            )
            assert [tuple(row) for row in rows] == [
                ("fallback-model", "fallback-provider", 2, 16, 5),
                ("later-model", "later-provider", 1, 7, 1),
            ]

            await database.create_session(
                "implicit-route",
                source="library",
                model="session-model",
            )
            await database.update_session_billing_route(
                "implicit-route",
                provider="session-provider",
                base_url="https://session-route.test/v1",
                billing_mode="api_key",
            )
            await database.update_token_counts(
                "implicit-route",
                input_tokens=4,
                output_tokens=2,
                api_call_count=1,
            )
            implicit_rows = await database._read_fetchall(
                "SELECT model, billing_provider, input_tokens, output_tokens "
                "FROM session_model_usage WHERE session_id = ? AND task = ''",
                ("implicit-route",),
            )
            assert [tuple(row) for row in implicit_rows] == [
                ("session-model", "session-provider", 4, 2)
            ]
        finally:
            await database.close()
            blocker.deactivate()


@pytest.mark.asyncio
async def test_flush_timeout_and_cancellation_do_not_cancel_writer(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "flush-cancel.db")
    original_update = database.update_token_counts
    update_started = asyncio.Event()
    release_update = asyncio.Event()

    async def _stalled_update(session_id, **kwargs):
        update_started.set()
        await release_update.wait()
        await original_update(session_id, **kwargs)

    monkeypatch.setattr(database, "update_token_counts", _stalled_update)
    await database.queue_token_counts(
        "session",
        input_tokens=9,
        api_call_count=1,
    )
    await asyncio.wait_for(update_started.wait(), timeout=1.0)

    async with no_task_leaks(action=LeakAction.RAISE):
        assert await database.flush_token_counts(timeout=0.01) is False
        flush_task = asyncio.create_task(database.flush_token_counts(timeout=1.0))
        await asyncio.sleep(0)
        flush_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flush_task

        assert database._token_writer_task is not None
        assert not database._token_writer_task.cancelled()
        release_update.set()
        assert await database.flush_token_counts(timeout=1.0) is True
        session = await database.get_session("session")
        assert session["input_tokens"] == 9
        assert session["api_call_count"] == 1
        await database.close()


@pytest.mark.asyncio
async def test_cancelled_close_drains_then_reraises_and_reopens_cleanly(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "close-cancel.db"
    database = SessionDB(path)
    original_update = database.update_token_counts
    update_started = asyncio.Event()
    release_update = asyncio.Event()

    async def _stalled_update(session_id, **kwargs):
        update_started.set()
        await release_update.wait()
        await original_update(session_id, **kwargs)

    monkeypatch.setattr(database, "update_token_counts", _stalled_update)
    await database.queue_token_counts(
        "session",
        input_tokens=13,
        output_tokens=4,
        api_call_count=1,
    )
    await asyncio.wait_for(update_started.wait(), timeout=1.0)

    async with no_task_leaks(action=LeakAction.RAISE):
        close_task = asyncio.create_task(database.close())
        await asyncio.sleep(0)
        close_task.cancel()
        release_update.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

    assert database._closed is True
    assert database._token_writer_task is None

    restored = SessionDB(path)
    try:
        session = await restored.get_session("session")
        assert session["input_tokens"] == 13
        assert session["output_tokens"] == 4
        assert session["api_call_count"] == 1
    finally:
        await restored.close()
