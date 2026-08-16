"""Real PostgreSQL multi-worker and failure-recovery readiness checks."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import uuid
from pathlib import Path

import aiofiles.os
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state_postgres import SessionDB


pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).with_name("_postgres_readiness_worker.py")
_RESULT_PREFIX = "POSTGRES_READINESS_RESULT="


def _dsn() -> str:
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    return dsn


def _worker_env(root: Path, dsn: str) -> dict[str, str]:
    """Pass only the credentials and runtime paths needed by the child."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": str(_REPO),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HOME": str(root / "os-home"),
        "HERMES_HOME": str(root / "hermes-home"),
        "TMPDIR": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "NO_PROXY": "127.0.0.1,localhost",
        "NO_COLOR": "1",
        "TERM": "dumb",
        "HERMES_POSTGRES_TEST_DSN": dsn,
    }


async def _start_worker(
    mode: str,
    root: Path,
    dsn: str,
    session_id: str,
    *,
    worker: str = "worker",
    count: int = 1,
    delay: float = 0.0,
    ready_path: Path | None = None,
) -> asyncio.subprocess.Process:
    await aiofiles.os.makedirs(root / "os-home", exist_ok=True)
    await aiofiles.os.makedirs(root / "hermes-home", exist_ok=True)
    await aiofiles.os.makedirs(root / "tmp", exist_ok=True)
    command = [
        sys.executable,
        str(_WORKER),
        mode,
        session_id,
        "--worker",
        worker,
        "--count",
        str(count),
        "--delay",
        str(delay),
    ]
    if ready_path is not None:
        command.extend(("--ready-path", str(ready_path)))
    return await asyncio.create_subprocess_exec(
        *command,
        cwd=root,
        env=_worker_env(root, dsn),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _finish_worker(
    process: asyncio.subprocess.Process,
    *,
    expected_code: int | None = 0,
    timeout: float = 30,
) -> dict[str, object] | None:
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        if process.returncode is None:
            process.kill()
        stdout, stderr = await process.communicate()
        pytest.fail(
            f"PostgreSQL readiness worker timed out after {timeout}s\n"
            f"stdout={stdout.decode(errors='replace')[-4000:]}\n"
            f"stderr={stderr.decode(errors='replace')[-4000:]}"
        )
    if expected_code is not None:
        assert process.returncode == expected_code, (
            f"worker exited {process.returncode}\n"
            f"stdout={stdout.decode(errors='replace')[-4000:]}\n"
            f"stderr={stderr.decode(errors='replace')[-4000:]}"
        )
    else:
        assert process.returncode != 0
        return None
    lines = [
        line.removeprefix(_RESULT_PREFIX)
        for line in stdout.decode(errors="replace").splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(lines) == 1, stdout.decode(errors="replace")[-4000:]
    return json.loads(lines[0])


async def _wait_for_path(
    path: Path,
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 10,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await aiofiles.os.path.exists(path):
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            pytest.fail(
                f"worker exited {process.returncode} before {path.name}\n"
                f"stdout={stdout.decode(errors='replace')}\n"
                f"stderr={stderr.decode(errors='replace')}"
            )
        if asyncio.get_running_loop().time() >= deadline:
            process.kill()
            stdout, stderr = await process.communicate()
            pytest.fail(
                f"worker did not create {path.name}\n"
                f"stdout={stdout.decode(errors='replace')}\n"
                f"stderr={stderr.decode(errors='replace')}"
            )
        await asyncio.sleep(0.02)


def _pool_metrics(database: SessionDB) -> dict[str, int]:
    pool = database._engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
    }


@pytest.mark.asyncio
async def test_postgres_multi_process_same_session_order_and_pool_cleanup(tmp_path):
    dsn = _dsn()
    database = SessionDB(dsn)
    session_id = f"readiness-shared-{uuid.uuid4()}"
    root = tmp_path / "workers"
    worker_count = 4
    messages_per_worker = 20
    processes: list[asyncio.subprocess.Process] = []
    try:
        await database.create_session(session_id, "postgres-readiness")
        processes = [
            await _start_worker(
                "append",
                root / f"worker-{index}",
                dsn,
                session_id,
                worker=f"{index}",
                count=messages_per_worker,
            )
            for index in range(worker_count)
        ]
        results = await asyncio.gather(
            *(_finish_worker(process) for process in processes)
        )
        assert [result["written"] for result in results if result] == [
            messages_per_worker
        ] * worker_count
        assert len({result["pid"] for result in results if result}) == worker_count
        assert all(result["duration_seconds"] > 0 for result in results if result)

        rows = await database.get_messages(session_id)
        assert len(rows) == worker_count * messages_per_worker
        contents = [row["content"] for row in rows]
        assert len(set(contents)) == len(contents)
        for worker in map(str, range(worker_count)):
            sequences = [
                int(content.rsplit("=", 1)[1])
                for content in contents
                if content.startswith(f"worker={worker};")
            ]
            assert sequences == list(range(messages_per_worker))
        assert _pool_metrics(database)["checked_out"] == 0
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        await database.delete_session(session_id)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_worker_kill_then_cold_resume_has_no_duplicate_rows(tmp_path):
    dsn = _dsn()
    database = SessionDB(dsn)
    session_id = f"readiness-restart-{uuid.uuid4()}"
    root = tmp_path / "restart-worker"
    ready_path = root / "ready"
    interrupted: asyncio.subprocess.Process | None = None
    try:
        await database.create_session(session_id, "postgres-readiness")
        interrupted = await _start_worker(
            "append",
            root,
            dsn,
            session_id,
            worker="interrupted",
            count=200,
            delay=0.02,
            ready_path=ready_path,
        )
        await _wait_for_path(ready_path, interrupted)
        await asyncio.sleep(0.08)
        interrupted.kill()
        await _finish_worker(interrupted, expected_code=None)

        after_kill = await database.get_messages(session_id)
        after_kill_contents = [row["content"] for row in after_kill]
        assert len(after_kill_contents) == len(set(after_kill_contents))
        assert await database.get_session(session_id) is not None

        resumed = await _start_worker(
            "resume",
            root / "resumed",
            dsn,
            session_id,
            worker="worker-b",
        )
        result = await _finish_worker(resumed)
        assert result is not None and result["written"] == 1
        rows = await database.get_messages(session_id)
        contents = [row["content"] for row in rows]
        assert contents.count("restart:worker-b") == 1
        assert len(contents) == len(set(contents))
        assert _pool_metrics(database)["checked_out"] == 0
    finally:
        if interrupted is not None and interrupted.returncode is None:
            interrupted.kill()
            await interrupted.communicate()
        await database.delete_session(session_id)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_terminated_backend_is_reconnected_by_pre_ping(tmp_path):
    dsn = _dsn()
    import asyncpg
    from sqlalchemy import text

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "database:\n  postgres:\n    pool_size: 2\n"
        "    max_overflow: 0\n    pool_pre_ping: true\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    database = SessionDB(dsn)
    connection = None
    admin = None
    try:
        await database._ensure_ready()
        connection = await database._engine.connect()
        backend_pid = (
            await connection.execute(text("SELECT pg_backend_pid()"))
        ).scalar_one()
        await connection.close()
        connection = None
        admin = await asyncpg.connect(
            dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        )
        assert await admin.fetchval(
            "SELECT pg_terminate_backend($1::integer)", backend_pid
        ) is True
        await admin.close()
        admin = None
        async with database._engine.connect() as replacement:
            assert (
                await replacement.execute(text("SELECT 42"))
            ).scalar_one() == 42
        assert _pool_metrics(database)["checked_out"] == 0
    finally:
        if connection is not None:
            await connection.close()
        if admin is not None:
            await admin.close()
        reset_hermes_home_override(token)
        await database.close()


@pytest.mark.asyncio
async def test_postgres_readiness_processes_have_no_signal_leaks(tmp_path):
    """A failed child is always reaped before the test returns."""
    dsn = _dsn()
    database = SessionDB(dsn)
    session_id = f"readiness-signal-{uuid.uuid4()}"
    root = tmp_path / "signal-worker"
    process = None
    try:
        await database.create_session(session_id, "postgres-readiness")
        process = await _start_worker(
            "append",
            root,
            dsn,
            session_id,
            worker="signal",
            count=100,
            delay=0.02,
            ready_path=root / "ready",
        )
        await _wait_for_path(root / "ready", process)
        process.send_signal(signal.SIGTERM)
        await _finish_worker(process, expected_code=None)
        assert process.returncode is not None
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.communicate()
        await database.delete_session(session_id)
        await database.close()
