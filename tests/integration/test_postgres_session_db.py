"""TDD and integration coverage for the explicit PostgreSQL SessionDB backend."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import uuid

import pytest

from hermes_state import SessionDB as SQLiteSessionDB
from hermes_state_postgres import SCHEMA_VERSION
from hermes_state_postgres import SessionDB as PostgresSessionDB
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from agent.agent_init import _validate_async_session_db
from run_agent import AIAgent


def _public_methods(cls):
    return {
        name: getattr(cls, name)
        for name in dir(cls)
        if not name.startswith("_") and callable(getattr(cls, name, None))
    }


def _signature_contract(method):
    signature = inspect.signature(method)
    return tuple(
        (
            parameter.name,
            parameter.kind,
            parameter.default,
        )
        for parameter in signature.parameters.values()
    )


def test_postgres_surface_matches_sqlite_retained_contract():
    sqlite_methods = _public_methods(SQLiteSessionDB)
    postgres_methods = _public_methods(PostgresSessionDB)
    assert set(postgres_methods) == set(sqlite_methods)
    for name, sqlite_method in sqlite_methods.items():
        postgres_method = postgres_methods[name]
        assert inspect.iscoroutinefunction(postgres_method) is inspect.iscoroutinefunction(
            sqlite_method
        ), name
        assert _signature_contract(postgres_method) == _signature_contract(sqlite_method), name


def test_postgres_constructor_rejects_implicit_or_non_postgres_storage():
    with pytest.raises(ValueError, match="explicit"):
        PostgresSessionDB()
    with pytest.raises(ValueError, match=r"postgresql\+asyncpg"):
        PostgresSessionDB("state.db")
    with pytest.raises(ValueError, match=r"postgresql\+asyncpg"):
        PostgresSessionDB("postgresql://localhost/hermes")
    with pytest.raises(ValueError, match="host and database"):
        PostgresSessionDB("postgresql+asyncpg://")


def test_postgres_constructor_is_state_only(monkeypatch):
    implementation_globals = PostgresSessionDB.__init__.__globals__

    def fail_if_called(*args, **kwargs):
        raise AssertionError("engine creation must be awaited")

    monkeypatch.setitem(implementation_globals, "_create_async_engine", fail_if_called)
    database = PostgresSessionDB(
        "postgresql+asyncpg://user:secret@localhost/hermes"
    )
    assert database._engine is None
    assert database._ready is False
    assert "secret" not in repr(database)


def test_postgres_sync_helpers_preserve_sqlite_behavior():
    assert PostgresSessionDB.sanitize_title("  hello\nworld ") == "hello world"
    assert PostgresSessionDB.session_unread(
        {"last_read_at": 1.0, "last_active": 2.0}
    )


def test_session_db_boundary_accepts_native_async_duck_types_only():
    class AsyncStore:
        async def create_session(self, *args, **kwargs):
            return "id"

        async def append_message(self, *args, **kwargs):
            return 1

        async def get_session(self, *args, **kwargs):
            return None

        async def end_session(self, *args, **kwargs):
            return None

        async def close(self):
            return None

    class SyncStore(AsyncStore):
        def close(self):
            return None

    _validate_async_session_db(AsyncStore())
    with pytest.raises(TypeError, match="native async"):
        _validate_async_session_db(SyncStore())


@pytest.mark.asyncio
async def test_missing_postgres_extra_fails_before_network(monkeypatch):
    # Patch the globals actually captured by the retained implementation's
    # method, even when another test has re-imported the source module under a
    # fresh module object.
    implementation_globals = PostgresSessionDB._ensure_ready.__globals__
    monkeypatch.setitem(implementation_globals, "_sa", None)
    monkeypatch.setitem(implementation_globals, "_create_async_engine", None)
    database = PostgresSessionDB("postgresql+asyncpg://user:pass@localhost/hermes")
    with pytest.raises(ImportError, match="postgres.*extra"):
        await database.get_session("missing")


@pytest.mark.asyncio
async def test_postgres_engine_options_are_captured_from_creation_profile(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text(
        """database:\n  postgres:\n    pool_size: 2\n    max_overflow: 1\n    pool_timeout: 4\n    pool_pre_ping: false\n    connect_args:\n      timeout: 11\n      command_timeout: 7\n      server_settings:\n        application_name: profile-a\n""",
        encoding="utf-8",
    )
    (home_b / "config.yaml").write_text(
        """database:\n  postgres:\n    pool_size: 9\n    connect_args:\n      timeout: 99\n      server_settings:\n        application_name: profile-b\n""",
        encoding="utf-8",
    )

    creation_token = set_hermes_home_override(home_a)
    try:
        database = PostgresSessionDB(
            "postgresql+asyncpg://user:pass@localhost/hermes"
        )
    finally:
        reset_hermes_home_override(creation_token)

    active_token = set_hermes_home_override(home_b)
    try:
        options = await database._resolve_engine_options()
    finally:
        reset_hermes_home_override(active_token)

    assert options["pool_size"] == 2
    assert options["max_overflow"] == 1
    assert options["pool_timeout"] == 4
    assert options["pool_pre_ping"] is False
    assert options["connect_args"]["timeout"] == 11
    assert options["connect_args"]["command_timeout"] == 7
    assert options["connect_args"]["server_settings"] == {
        "application_name": "profile-a"
    }
    (home_a / "config.yaml").write_text(
        "database:\n  postgres:\n    pool_size: 8\n",
        encoding="utf-8",
    )
    assert (await database._resolve_engine_options())["pool_size"] == 2
    await database.close()

    creation_token = set_hermes_home_override(home_a)
    try:
        replacement = PostgresSessionDB(
            "postgresql+asyncpg://user:pass@localhost/hermes"
        )
        replacement_options = await replacement._resolve_engine_options()
    finally:
        reset_hermes_home_override(creation_token)
    assert replacement_options["pool_size"] == 8
    await replacement.close()


@pytest.mark.asyncio
async def test_postgres_read_only_options_force_server_setting(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        """database:\n  postgres:\n    connect_args:\n      server_settings:\n        application_name: readonly-test\n""",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        database = PostgresSessionDB(
            "postgresql+asyncpg://user:pass@localhost/hermes",
            read_only=True,
        )
        options = await database._resolve_engine_options()
    finally:
        reset_hermes_home_override(token)
    assert options["connect_args"]["server_settings"] == {
        "application_name": "readonly-test",
        "default_transaction_read_only": "on",
    }
    await database.close()


@pytest.mark.asyncio
async def test_postgres_engine_options_are_isolated_for_concurrent_profiles(tmp_path):
    homes = []
    for name, pool_size in (("a", 2), ("b", 7)):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            f"database:\n  postgres:\n    pool_size: {pool_size}\n",
            encoding="utf-8",
        )
        homes.append(home)

    stores = []
    for home in homes:
        token = set_hermes_home_override(home)
        try:
            stores.append(
                PostgresSessionDB(
                    "postgresql+asyncpg://user:pass@localhost/hermes"
                )
            )
        finally:
            reset_hermes_home_override(token)
    try:
        options = await asyncio.gather(
            *(store._resolve_engine_options() for store in stores)
        )
    finally:
        await asyncio.gather(*(store.close() for store in stores))
    assert [item["pool_size"] for item in options] == [2, 7]


@pytest.mark.asyncio
async def test_postgres_config_rejects_unknown_and_reserved_options(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        """database:\n  postgres:\n    pool_szie: 2\n""",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        database = PostgresSessionDB(
            "postgresql+asyncpg://user:pass@localhost/hermes"
        )
        with pytest.raises(ValueError, match="unsupported"):
            await database._resolve_engine_options()
    finally:
        reset_hermes_home_override(token)
    await database.close()


@pytest.mark.asyncio
async def test_postgres_config_rejects_invalid_types_before_network(tmp_path):
    cases = (
        (
            "database:\n  postgres:\n    pool_size: true\n",
            "pool_size must be an integer",
        ),
        (
            "database:\n  postgres:\n    max_overflow: -2\n",
            "max_overflow must be >= -1",
        ),
        (
            "database:\n  postgres:\n    connect_args:\n      command_timeout: 0\n",
            "connect_args.command_timeout must be > 0",
        ),
        (
            "database:\n  postgres:\n    connect_args:\n      server_settings:\n        statement_timeout: 60000\n",
            "only string keys and values",
        ),
    )
    for index, (contents, message) in enumerate(cases):
        home = tmp_path / f"profile-{index}"
        home.mkdir()
        (home / "config.yaml").write_text(contents, encoding="utf-8")
        token = set_hermes_home_override(home)
        try:
            database = PostgresSessionDB(
                "postgresql+asyncpg://user:pass@localhost/hermes"
            )
            with pytest.raises(ValueError, match=message):
                await database._resolve_engine_options()
        finally:
            reset_hermes_home_override(token)
        await database.close()

    (home / "config.yaml").write_text(
        """database:\n  postgres:\n    connect_args:\n      server_settings:\n        default_transaction_read_only: off\n""",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        database = PostgresSessionDB(
            "postgresql+asyncpg://user:pass@localhost/hermes",
        )
        with pytest.raises(ValueError, match="controlled by read_only"):
            await database._resolve_engine_options()
    finally:
        reset_hermes_home_override(token)
    await database.close()


@pytest.mark.asyncio
async def test_postgres_config_error_can_be_fixed_before_engine_creation(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(
        "database:\n  postgres:\n    pool_size: true\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        database = PostgresSessionDB(
            "postgresql+asyncpg://user:pass@localhost/hermes"
        )
        with pytest.raises(ValueError, match="pool_size"):
            await database._resolve_engine_options()
        config_path.write_text(
            "database:\n  postgres:\n    pool_size: 3\n",
            encoding="utf-8",
        )
        options = await database._resolve_engine_options()
    finally:
        reset_hermes_home_override(token)
    assert options["pool_size"] == 3
    assert database._engine is None
    await database.close()


@pytest.mark.asyncio
async def test_postgres_crud_search_and_lock_contract():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    session_id = f"pytest-{uuid.uuid4()}"
    try:
        assert await database.create_session(session_id, "test") == session_id
        assert (await database.get_session(session_id))["source"] == "test"
        first = await database.append_message(session_id, "user", "hello postgres")
        second = await database.append_message(session_id, "assistant", "hello back")
        assert second > first
        assert await database.message_count(session_id) == 2
        assert (await database.get_messages_as_conversation(session_id))[0]["role"] == "user"
        hits = await database.search_messages("postgres")
        assert [hit["session_id"] for hit in hits] == [session_id]
        exported = await database.export_session(session_id)
        assert exported and len(exported["messages"]) == 2
        assert await database.try_acquire_compression_lock(session_id, "owner")
        assert not await database.try_acquire_compression_lock(session_id, "other")
        await database.release_compression_lock(session_id, "owner")
        assert await database.get_compression_lock_holder(session_id) is None
    finally:
        await database.delete_session(session_id)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_close_repeated_cancellation_is_deterministic():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    await database._ensure_ready()
    task = asyncio.create_task(database.close())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await database.close()


@pytest.mark.asyncio
async def test_postgres_concurrent_agents_share_one_database_pool():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    session_ids = [f"pytest-concurrent-{uuid.uuid4()}-{index}" for index in range(8)]
    shared_id = f"pytest-concurrent-shared-{uuid.uuid4()}"

    async def _write(session_id: str) -> None:
        await database.create_session(session_id, "concurrent")
        await database.append_messages_batch(
            session_id,
            [
                {"role": "user", "content": f"question:{session_id}"},
                {"role": "assistant", "content": "answer"},
            ],
            chunk_rows=1,
        )

    try:
        await asyncio.gather(*(_write(session_id) for session_id in session_ids))
        assert await database.session_count(source="concurrent") == len(session_ids)
        counts = await asyncio.gather(
            *(database.message_count(session_id) for session_id in session_ids)
        )
        assert counts == [2] * len(session_ids)
        await database.create_session(shared_id, "concurrent")
        await asyncio.gather(
            *(
                database.append_message(shared_id, "user", f"message-{index}")
                for index in range(8)
            )
        )
        assert await database.message_count(shared_id) == 8
    finally:
        await asyncio.gather(
            *(
                database.delete_session(session_id)
                for session_id in [*session_ids, shared_id]
            )
        )
        await database.close()


@pytest.mark.asyncio
async def test_postgres_four_process_workers_preserve_all_sessions():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    batch = f"pytest-process-{uuid.uuid4()}"
    source = f"process-{uuid.uuid4()}"
    session_ids = [
        f"{batch}-{worker}-{index}"
        for worker in range(4)
        for index in range(25)
    ]
    child_code = """
import asyncio
import os
from hermes_state_postgres import SessionDB

async def main():
    database = SessionDB(os.environ["HERMES_POSTGRES_TEST_DSN"])
    batch = os.environ["HERMES_PROCESS_BATCH"]
    source = os.environ["HERMES_PROCESS_SOURCE"]
    worker = int(os.environ["HERMES_PROCESS_WORKER"])
    try:
        for index in range(25):
            session_id = f"{batch}-{worker}-{index}"
            await database.create_session(session_id, source)
            await database.append_message(session_id, "user", session_id)
    finally:
        await database.close()

asyncio.run(main())
"""
    processes = []
    try:
        for worker in range(4):
            env = {
                **os.environ,
                "HERMES_PROCESS_BATCH": batch,
                "HERMES_PROCESS_SOURCE": source,
                "HERMES_PROCESS_WORKER": str(worker),
            }
            processes.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    child_code,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
        results = await asyncio.gather(
            *(process.communicate() for process in processes)
        )
        for process, (stdout, stderr) in zip(processes, results):
            assert process.returncode == 0, stderr.decode() or stdout.decode()
        assert await database.session_count(source=source) == 100
        assert await asyncio.gather(
            *(database.message_count(session_id) for session_id in session_ids)
        ) == [1] * 100
    finally:
        await database.delete_sessions(session_ids)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_export_import_and_native_maintenance_roundtrip():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    source_id = f"pytest-export-{uuid.uuid4()}"
    imported_id = f"pytest-import-{uuid.uuid4()}"
    try:
        await database.create_session(
            source_id,
            "export",
            model="test-model",
            model_config={"temperature": 0},
            title="roundtrip",
        )
        await database.append_message(
            source_id,
            "user",
            content={"nested": ["value", 1]},
            tool_calls=[{"id": "call-1", "type": "function"}],
        )
        exported = await database.export_session(source_id)
        assert exported is not None
        exported["id"] = imported_id
        imported = await database.import_sessions([exported])
        assert imported == {"imported": 1, "errors": []}
        restored = await database.get_messages(imported_id)
        assert restored[0]["content"] == {"nested": ["value", 1]}
        assert restored[0]["tool_calls"] == [
            {"id": "call-1", "type": "function"}
        ]
        assert await database.rebuild_fts() == 1
        assert await database.optimize_fts() == 1
        assert await database.vacuum() == 1
        optimized = await database.optimize_fts_storage(vacuum=False)
        assert optimized == {"ok": True, "vacuumed": False}
    finally:
        await database.delete_session(source_id)
        await database.delete_session(imported_id)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_read_only_backend_rejects_writes_before_connecting():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn, read_only=True)
    try:
        with pytest.raises(PermissionError, match="read-only"):
            await database.create_session(f"pytest-read-only-{uuid.uuid4()}", "test")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_postgres_read_only_enforces_server_transaction_and_reads_existing_schema():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    session_id = f"pytest-read-only-existing-{uuid.uuid4()}"
    writer = PostgresSessionDB(dsn)
    reader = PostgresSessionDB(dsn, read_only=True)
    try:
        await writer.create_session(session_id, "read-only-test")
        await writer.append_message(session_id, "user", "visible")
        assert (await reader.get_session(session_id))["id"] == session_id

        await reader._ensure_ready()
        connections = await asyncio.gather(
            reader._engine.connect(),
            reader._engine.connect(),
        )
        try:
            states = await asyncio.gather(
                *(
                    connection.execute(
                        text("SHOW transaction_read_only")
                    )
                    for connection in connections
                )
            )
            assert [result.scalar_one() for result in states] == ["on", "on"]
            with pytest.raises(Exception) as caught:
                await connections[0].execute(
                    text(
                        "INSERT INTO state_meta(key, value) "
                        "VALUES ('pytest-read-only', 'blocked')"
                    )
                )
            original = getattr(caught.value, "orig", None)
            assert getattr(original, "sqlstate", None) == "25006"
        finally:
            await asyncio.gather(*(connection.close() for connection in connections))
    finally:
        await writer.delete_session(session_id)
        await reader.close()
        await writer.close()


@pytest.mark.asyncio
async def test_postgres_read_only_never_creates_an_uninitialized_schema(tmp_path):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    import asyncpg

    schema = f"readonly_empty_{uuid.uuid4().hex}"
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    connect_args:\n"
        f"      server_settings:\n        search_path: {schema}\n",
        encoding="utf-8",
    )
    admin_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    admin = await asyncpg.connect(admin_dsn)
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await admin.close()

    token = set_hermes_home_override(home)
    database = PostgresSessionDB(dsn, read_only=True)
    try:
        with pytest.raises(RuntimeError, match="initialized schema"):
            await database._ensure_ready()
    finally:
        reset_hermes_home_override(token)
        await database.close()

    admin = await asyncpg.connect(admin_dsn)
    try:
        assert await admin.fetchval(
            "SELECT to_regclass($1)", f'"{schema}".schema_version'
        ) is None
        await admin.execute(f'DROP SCHEMA "{schema}"')
    finally:
        await admin.close()


@pytest.mark.asyncio
async def test_postgres_read_only_rejects_a_newer_schema():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text

    writer = PostgresSessionDB(dsn)
    reader = PostgresSessionDB(dsn, read_only=True)
    newer = SCHEMA_VERSION + 1
    try:
        await writer._ensure_ready()
        async with writer._engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO schema_version(version) VALUES (:version)"),
                {"version": newer},
            )
        with pytest.raises(RuntimeError, match="newer"):
            await reader._ensure_ready()
    finally:
        async with writer._engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM schema_version WHERE version = :version"),
                {"version": newer},
            )
        await reader.close()
        await writer.close()


@pytest.mark.asyncio
async def test_postgres_pool_configuration_reaches_real_sqlalchemy_engine(tmp_path):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        """database:\n  postgres:\n    pool_size: 2\n    max_overflow: 1\n    pool_timeout: 4\n    pool_recycle: 60\n    pool_pre_ping: false\n    pool_use_lifo: true\n    connect_args:\n      timeout: 11\n      command_timeout: 7\n      statement_cache_size: 100\n      max_cached_statement_lifetime: 300.5\n      max_cacheable_statement_size: 15360\n      server_settings:\n        application_name: postgres-config-test\n""",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    database = PostgresSessionDB(dsn)
    try:
        await database._ensure_ready()
    finally:
        reset_hermes_home_override(token)
    try:
        pool = database._engine.pool
        assert pool.size() == 2
        assert pool._max_overflow == 1
        assert pool._timeout == 4
        assert pool._pre_ping is False
        assert pool._pool.use_lifo is True
        assert database._engine_options["connect_args"] == {
            "timeout": 11,
            "command_timeout": 7,
            "statement_cache_size": 100,
            "max_cached_statement_lifetime": 300.5,
            "max_cacheable_statement_size": 15360,
            "server_settings": {"application_name": "postgres-config-test"},
        }
        async with database._engine.connect() as connection:
            application_name = (
                await connection.execute(
                    text("SHOW application_name")
                )
            ).scalar_one()
            assert application_name == "postgres-config-test"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_postgres_command_and_statement_timeouts_reach_real_connection(tmp_path):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    connect_args:\n"
        # Keep the client timeout above the server timeout so PostgreSQL can
        # finish cancelling the statement before SQLAlchemy resets the
        # connection on context exit.  The assertion below exercises the
        # server-side statement timeout; command-timeout propagation is
        # covered by the option-capture test above.
        "      command_timeout: 1.0\n"
        "      server_settings:\n        statement_timeout: '100'\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    database = PostgresSessionDB(dsn)
    try:
        await database._ensure_ready()
        async with database._engine.connect() as connection:
            assert (
                await connection.execute(text("SHOW statement_timeout"))
            ).scalar_one() == "100ms"
            with pytest.raises((TimeoutError, DBAPIError)) as timeout_error:
                await connection.execute(text("SELECT pg_sleep(0.5)"))
            if isinstance(timeout_error.value, DBAPIError):
                assert getattr(timeout_error.value.orig, "sqlstate", None) == "57014"
            await connection.rollback()
    finally:
        reset_hermes_home_override(token)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_pool_timeout_and_connection_return_after_checkout(tmp_path):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text
    from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    pool_size: 1\n"
        "    max_overflow: 0\n    pool_timeout: 0.1\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    database = PostgresSessionDB(dsn)
    try:
        await database._ensure_ready()
        holder = await database._engine.connect()
        await holder.execute(text("SELECT 1"))

        async def wait_for_checkout():
            async with database._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))

        with pytest.raises(SQLAlchemyTimeoutError):
            await wait_for_checkout()
        await holder.close()
        async with database._engine.connect() as connection:
            assert (await connection.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        reset_hermes_home_override(token)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_pre_ping_reconnects_after_invalidated_connection(tmp_path):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    pool_pre_ping: true\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    database = PostgresSessionDB(dsn)
    try:
        await database._ensure_ready()
        connection = await database._engine.connect()
        try:
            assert (
                await connection.execute(text("SELECT 1"))
            ).scalar_one() == 1
            await connection.invalidate()
        finally:
            await connection.close()
        async with database._engine.connect() as replacement:
            assert (
                await replacement.execute(text("SELECT 2"))
            ).scalar_one() == 2
    finally:
        reset_hermes_home_override(token)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_cancelled_transaction_rolls_back_and_returns_connection(
    tmp_path,
):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    from sqlalchemy import text

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    pool_size: 1\n    max_overflow: 0\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    database = PostgresSessionDB(dsn)
    key = f"cancelled-{uuid.uuid4()}"
    started = asyncio.Event()

    async def write_then_wait():
        async with database._engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO state_meta(key, value) VALUES (:key, 'stale')"),
                {"key": key},
            )
            started.set()
            await connection.execute(text("SELECT pg_sleep(30)"))

    task = None
    try:
        await database._ensure_ready()
        task = asyncio.create_task(write_then_wait())
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with database._engine.connect() as connection:
            value = (
                await connection.execute(
                    text("SELECT value FROM state_meta WHERE key = :key"),
                    {"key": key},
                )
            ).scalar_one_or_none()
            assert value is None
    finally:
        if task is not None and not task.done():
            task.cancel()
            await task
        reset_hermes_home_override(token)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_title_provenance_and_uniqueness_match_sqlite():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    first = f"pytest-title-{uuid.uuid4()}"
    second = f"pytest-title-{uuid.uuid4()}"
    try:
        await database.create_session(first, "test")
        await database.create_session(second, "test")
        assert await database.set_auto_title(
            first, "Derived", source=database.TITLE_SOURCE_DERIVED
        )
        assert await database.set_auto_title(
            first, "Model", source=database.TITLE_SOURCE_LLM
        )
        assert not await database.set_auto_title(
            first, "Older", source=database.TITLE_SOURCE_DERIVED
        )
        assert await database.set_session_title(first, "Manual")
        assert not await database.set_auto_title(
            first, "Ignored", source=database.TITLE_SOURCE_LLM
        )
        with pytest.raises(ValueError, match="invalid automatic title source"):
            await database.set_auto_title(first, "Invalid", source="unknown")
        with pytest.raises(ValueError, match="already in use"):
            await database.set_session_title(second, "Manual")
        assert await database.set_session_title_source(
            first, database.TITLE_SOURCE_DERIVED
        )
        with pytest.raises(ValueError, match="invalid title source"):
            await database.set_session_title_source(first, "unknown")
    finally:
        await database.delete_session(first)
        await database.delete_session(second)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_retains_sqlite_edge_behavior_and_agent_ownership():
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    database = PostgresSessionDB(dsn)
    parent = f"pytest-edge-parent-{uuid.uuid4()}"
    child = f"pytest-edge-child-{uuid.uuid4()}"
    archived = f"pytest-edge-archived-{uuid.uuid4()}"
    source = f"pytest-edge-source-{uuid.uuid4()}"
    try:
        await database.create_session(
            parent,
            source,
            cwd="/workspace/project",
            session_key="edge",
        )
        await database.end_session(parent, "compression")
        await database.create_session(child, source, parent_session_id=parent)
        assert (await database.get_session(child))["cwd"] == "/workspace/project"
        assert (await database.get_session(child))["session_key"] == "edge"

        await database.create_session(archived, source)
        await database.set_session_archived(archived, True)
        assert await database.session_count(source=source, archived_only=True) == 1
        by_source = await database.session_count_by_source()
        assert by_source[source] == 2
        by_source_all = await database.session_count_by_source(include_archived=True)
        assert by_source_all[source] == 3

        await database.append_message(
            child,
            "user",
            "广西旅行计划",
            display_metadata={"phase": "retry"},
        )
        await database.append_message(child, "assistant", "桂林旅行计划")
        assert {
            row["session_id"]
            for row in await database.search_messages("广西 OR 桂林")
        } == {child}
        target_id = await database.append_message(
            child,
            "user",
            "retry this",
            display_metadata={"phase": "retry"},
        )
        rewind = await database.rewind_to_message(child, target_id)
        assert rewind["target_message"]["display_metadata"] == json.dumps(
            {"phase": "retry"}
        )

        agent = AIAgent(
            model="test-model",
            provider="openai",
            api_key="test-key",
            base_url="http://127.0.0.1:9/v1",
            session_db=database,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
            load_soul_identity=False,
        )
        assert agent._session_db is database
        assert agent._owns_session_db is False
        await agent.close()
        assert database._closed is False
        assert await database.get_session(child) is not None
    finally:
        for session_id in (parent, child, archived):
            await database.delete_session(session_id)
        await database.close()
