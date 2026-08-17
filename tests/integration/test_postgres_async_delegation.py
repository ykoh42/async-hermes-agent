"""PostgreSQL durability and session_search integration for async delegation."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
from pathlib import Path
import sys

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad


_EXPECTED_PUBLIC_SURFACE = (
    ("recover_abandoned_delegations", True),
    ("restore_undelivered_completions", True),
    ("mark_completion_delivered", True),
    ("claim_completion_delivery", True),
    ("claim_event_delivery", True),
    ("release_completion_delivery", True),
    ("drop_completion_delivery", True),
    ("complete_completion_delivery", True),
    ("complete_event_delivery", True),
    ("release_event_delivery", True),
    ("get_durable_delegation", True),
    ("active_count", False),
    ("active_for_session", False),
    ("active_task_count", False),
    ("has_live_for_session", False),
    ("dispatch_async_delegation", True),
    ("dispatch_async_delegation_batch", True),
    ("list_async_delegations", False),
    ("interrupt_all", True),
    ("interrupt_for_session", True),
)


def _postgres_dsn() -> str:
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    return dsn


def _surface(path: Path) -> list[tuple[str, bool]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.name, isinstance(node, ast.AsyncFunctionDef))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


def test_async_delegation_public_surface_matches_upstream_baseline():
    current = _surface(Path(__file__).parents[2] / "tools" / "async_delegation.py")
    assert current == list(_EXPECTED_PUBLIC_SURFACE)

    for name, expected_async in _EXPECTED_PUBLIC_SURFACE:
        function = getattr(ad, name)
        assert inspect.iscoroutinefunction(function) is expected_async, name


@pytest.mark.asyncio
async def test_explicit_sqlite_context_does_not_inherit_pg_scope_binding():
    class FakePostgresStore:
        async def _read(self, operation):
            return None

        async def _write(self, operation):
            return None

    await ad._reset_for_tests()
    store = FakePostgresStore()
    await ad._bind_session_db(store)
    assert ad._postgres_session_db() is not None
    await ad._bind_session_db(None)
    assert ad._postgres_session_db() is None

    class SQLiteStore:
        pass

    await ad._bind_session_db(SQLiteStore())
    assert ad._postgres_session_db() is None


@pytest.mark.asyncio
async def test_pg_binding_cannot_cross_profile_context(tmp_path):
    class FakePostgresStore:
        async def _read(self, operation):
            return None

        async def _write(self, operation):
            return None

    await ad._reset_for_tests()
    store = FakePostgresStore()
    first = tmp_path / "profile-a"
    second = tmp_path / "profile-b"
    first.mkdir()
    second.mkdir()
    token = set_hermes_home_override(first)
    try:
        await ad._bind_session_db(store)
        assert ad._postgres_session_db() is store
        reset_hermes_home_override(token)
        token = set_hermes_home_override(second)
        await ad._activate_scope_state()
        assert ad._postgres_session_db() is None
        reset_hermes_home_override(token)
        token = set_hermes_home_override(first)
        await ad._activate_scope_state()
        with pytest.raises(RuntimeError, match="shared by"):
            await ad._bind_session_db(FakePostgresStore())
    finally:
        await ad._reset_for_tests()
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_bound_postgres_delegation_never_creates_sqlite_sidecar(tmp_path):
    from hermes_state_postgres import SessionDB

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    await ad._reset_for_tests()
    try:
        await ad._bind_session_db(database)

        async def runner():
            return {
                "status": "completed",
                "summary": "delegated result",
                "api_calls": 1,
            }

        result = await ad.dispatch_async_delegation(
            goal="find the answer",
            context=None,
            toolsets=None,
            role="leaf",
            model="test-model",
            session_key="pg-session",
            runner=runner,
        )
        delegation_id = result["delegation_id"]
        while ad.active_count():
            await asyncio.sleep(0)

        durable = await ad.get_durable_delegation(delegation_id)
        assert durable is not None
        assert durable["state"] == "completed"
        assert durable["result"]["summary"] == "delegated result"
        assert not (tmp_path / "state.db").exists()

        from sqlalchemy import text

        await database._ensure_ready()
        async with database._engine.connect() as connection:
            row = await connection.execute(
                text(
                    "SELECT owner_instance_id, lease_expires_at "
                    "FROM async_delegations WHERE delegation_id = :id"
                ),
                {"id": delegation_id},
            )
            owner, lease = row.one()
        assert owner
        assert lease is None
    finally:
        await ad._reset_for_tests()
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_postgres_session_search_reads_injected_backend(tmp_path):
    from hermes_state_postgres import SessionDB
    from tools.session_search_tool import session_search

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    session_id = f"pg-search-{os.getpid()}"
    try:
        await database.create_session(session_id, "test")
        await database.append_message(
            session_id,
            "assistant",
            "delegated result persisted in PostgreSQL",
        )
        payload = await session_search(query="delegated result", db=database)
        assert session_id in payload
        assert "PostgreSQL" in payload
        assert not (tmp_path / "state.db").exists()
    finally:
        await database.delete_session(session_id)
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_postgres_session_search_browse_read_and_scroll_shapes(tmp_path):
    from hermes_state_postgres import SessionDB
    from tools.session_search_tool import session_search

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    session_id = f"pg-search-shapes-{os.getpid()}"
    try:
        await database.create_session(session_id, "api", title="PG search shapes")
        first = await database.append_message(
            session_id, "user", "search shape question"
        )
        second = await database.append_message(
            session_id, "assistant", "search shape answer"
        )

        browse = json.loads(await session_search(db=database))
        assert browse["success"] is True
        assert browse["mode"] == "browse"
        assert any(row["session_id"] == session_id for row in browse["results"])

        read = json.loads(await session_search(db=database, session_id=session_id))
        assert read["success"] is True
        assert read["mode"] == "read"
        assert [row["id"] for row in read["messages"]] == [first, second]

        scroll = json.loads(
            await session_search(
                db=database,
                session_id=session_id,
                around_message_id=second,
                window=1,
            )
        )
        assert scroll["success"] is True
        assert scroll["mode"] == "scroll"
        assert scroll["around_message_id"] == second
        assert any(row["id"] == second for row in scroll["messages"])
        assert not (tmp_path / "state.db").exists()
    finally:
        await database.delete_session(session_id)
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_postgres_delegation_delivery_claim_is_single_winner(tmp_path):
    from hermes_state_postgres import SessionDB

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    await ad._reset_for_tests()
    try:
        await ad._bind_session_db(database)
        record = {
            "delegation_id": f"claim-{os.getpid()}",
            "goal": "claim",
            "context": None,
            "toolsets": None,
            "role": "leaf",
            "model": "m",
            "session_key": "pg-session",
            "status": "running",
            "dispatched_at": 1.0,
        }
        await ad._persist_dispatch(record)
        event = {
            "type": "async_delegation",
            "delegation_id": record["delegation_id"],
            "session_key": "pg-session",
            "status": "completed",
            "summary": "one",
            "dispatched_at": 1.0,
            "completed_at": 2.0,
        }
        await ad._persist_completion(event, {"summary": "one"})
        claims = await asyncio.gather(
            ad.claim_completion_delivery(record["delegation_id"], "a"),
            ad.claim_completion_delivery(record["delegation_id"], "b"),
        )
        assert sorted(claims) == [False, True]
    finally:
        await ad._reset_for_tests()
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_postgres_delegation_restores_and_completes_delivery(tmp_path):
    from hermes_state_postgres import SessionDB

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    await ad._reset_for_tests()
    delegation_id = f"restore-{os.getpid()}"
    try:
        await ad._bind_session_db(database)
        await ad._persist_dispatch(
            {
                "delegation_id": delegation_id,
                "goal": "restore",
                "context": None,
                "toolsets": None,
                "role": "leaf",
                "model": "m",
                "session_key": "restore-session",
                "status": "running",
                "dispatched_at": 1.0,
            }
        )
        await ad._persist_completion(
            {
                "type": "async_delegation",
                "delegation_id": delegation_id,
                "session_key": "restore-session",
                "status": "completed",
                "completed_at": 2.0,
            },
            {"status": "completed", "summary": "restored"},
        )
        queue = asyncio.Queue()
        assert await ad.restore_undelivered_completions(queue) >= 1
        events = [queue.get_nowait() for _ in range(queue.qsize())]
        event = next(
            event
            for event in events
            if event.get("delegation_id") == delegation_id
        )
        assert event["restored"] is True
        claim = await ad.claim_event_delivery(event, "restore-worker")
        assert claim
        await ad.complete_event_delivery(event, claim)
        durable = await ad.get_durable_delegation(delegation_id)
        assert durable["delivery_state"] == "delivered"
    finally:
        await ad._reset_for_tests()
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_postgres_delegation_claim_is_single_winner_across_processes(tmp_path):
    from hermes_state_postgres import SessionDB

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    await ad._reset_for_tests()
    delegation_id = f"process-claim-{os.getpid()}"
    try:
        await ad._bind_session_db(database)
        record = {
            "delegation_id": delegation_id,
            "goal": "process claim",
            "context": None,
            "toolsets": None,
            "role": "leaf",
            "model": "m",
            "session_key": "pg-process-session",
            "status": "running",
            "dispatched_at": 1.0,
        }
        await ad._persist_dispatch(record)
        await ad._persist_completion(
            {
                "type": "async_delegation",
                "delegation_id": delegation_id,
                "session_key": "pg-process-session",
                "status": "completed",
                "completed_at": 2.0,
            },
            {"status": "completed"},
        )

        child = """
import asyncio
import os
import sys

from hermes_state_postgres import SessionDB
from tools import async_delegation as delegation


async def main():
    database = SessionDB(sys.argv[1])
    try:
        await delegation._bind_session_db(database)
        claimed = await delegation.claim_completion_delivery(sys.argv[2], sys.argv[3])
        print("true" if claimed else "false", flush=True)
    finally:
        await database.close()


asyncio.run(main())
"""
        commands = []
        for claim_id in ("worker-a", "worker-b", "worker-c", "worker-d"):
            worker_home = tmp_path / claim_id
            worker_home.mkdir()
            env = os.environ.copy()
            env["HERMES_HOME"] = str(worker_home)
            commands.append(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    child,
                    _postgres_dsn(),
                    delegation_id,
                    claim_id,
                    cwd=str(Path(__file__).parents[2]),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
        processes = await asyncio.gather(*commands)
        outputs = await asyncio.gather(*(process.communicate() for process in processes))
        assert all(process.returncode == 0 for process in processes), [
            stderr.decode() for _stdout, stderr in outputs
        ]
        claims = [stdout.decode().strip().splitlines()[-1] for stdout, _stderr in outputs]
        assert claims.count("true") == 1
        assert claims.count("false") == 3
    finally:
        await ad._reset_for_tests()
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_agent_first_await_binds_injected_postgres_store(tmp_path):
    from hermes_state_postgres import SessionDB
    from run_agent import AIAgent

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    agent = AIAgent(
        provider="custom",
        api_key="delegation-test-key",
        base_url="http://127.0.0.1:1/v1",
        model="delegation-test-model",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_db=database,
    )
    try:
        await agent._ensure_provider_runtime()
        state = ad._current_scope_state()
        assert state.session_db_ref is not None
        assert state.session_db_ref() is database
        assert not (tmp_path / "state.db").exists()
    finally:
        await agent.close()
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_postgres_storage_v2_migrates_delegations_atomically(tmp_path):
    import psycopg
    from hermes_state_postgres import SessionDB
    from sqlalchemy import text

    dsn = _postgres_dsn()
    schema = f"delegation_v2_{os.getpid()}_{id(tmp_path)}"
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    connect_args:\n"
        f"      options: '-c search_path={schema}'\n",
        encoding="utf-8",
    )
    admin_dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    admin = await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True)
    token = set_hermes_home_override(home)
    database = SessionDB(dsn)
    migrated = None
    try:
        async with admin.cursor() as cursor:
            await cursor.execute(f'CREATE SCHEMA "{schema}"')
        await database._ensure_ready()
        async with database._engine.begin() as connection:
            await connection.execute(text("DROP TABLE async_delegations"))
            await connection.execute(
                text(
                    "UPDATE state_meta SET value = '2' "
                    "WHERE key = 'postgres_storage_version'"
                )
            )
        await database.close()

        migrated = SessionDB(dsn)
        await migrated._ensure_ready()
        async with migrated._engine.connect() as connection:
            storage = (
                await connection.execute(
                    text(
                        "SELECT value FROM state_meta "
                        "WHERE key = 'postgres_storage_version'"
                    )
                )
            ).scalar_one()
            table = (
                await connection.execute(
                    text("SELECT to_regclass('async_delegations')")
                )
            ).scalar_one()
        assert storage == "4"
        assert table == "async_delegations"
    finally:
        if migrated is not None:
            await migrated.close()
        if not database._closed:
            await database.close()
        async with admin.cursor() as cursor:
            await cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_read_only_postgres_delegation_fails_before_runner_or_sqlite_fallback(
    tmp_path,
):
    from hermes_state_postgres import SessionDB

    database = SessionDB(_postgres_dsn(), read_only=True)
    home_token = set_hermes_home_override(tmp_path)
    await ad._reset_for_tests()
    started = False
    try:
        await ad._bind_session_db(database)

        async def runner():
            nonlocal started
            started = True
            return {"status": "completed"}

        with pytest.raises(PermissionError, match="read-only"):
            await ad.dispatch_async_delegation(
                goal="read only",
                context=None,
                toolsets=None,
                role="leaf",
                model="m",
                session_key="readonly",
                runner=runner,
            )
        assert started is False
        assert not (tmp_path / "state.db").exists()
    finally:
        await ad._reset_for_tests()
        await database.close()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_expired_postgres_owner_is_recovered_and_fenced(tmp_path):
    from hermes_state_postgres import SessionDB
    from sqlalchemy import text
    from tools.process_registry import process_registry

    database = SessionDB(_postgres_dsn())
    home_token = set_hermes_home_override(tmp_path)
    await ad._reset_for_tests()
    release = asyncio.Event()
    started = asyncio.Event()
    try:
        await ad._bind_session_db(database)

        async def runner():
            started.set()
            await release.wait()
            return {"status": "completed", "summary": "late"}

        result = await ad.dispatch_async_delegation(
            goal="fenced",
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="pg-session",
            runner=runner,
        )
        await started.wait()

        async def expire(connection):
            await connection.execute(
                text(
                    "UPDATE async_delegations SET lease_expires_at = 0 "
                    "WHERE delegation_id = :id"
                ),
                {"id": result["delegation_id"]},
            )

        await database._write(expire)
        assert await ad.recover_abandoned_delegations() == 1
        release.set()
        while ad.active_count():
            await asyncio.sleep(0)
        durable = await ad.get_durable_delegation(result["delegation_id"])
        assert durable["state"] == "unknown"
        assert durable["result"]["status"] == "unknown"
        assert process_registry.completion_queue.empty()
    finally:
        release.set()
        await ad._reset_for_tests()
        await database.close()
        reset_hermes_home_override(home_token)
