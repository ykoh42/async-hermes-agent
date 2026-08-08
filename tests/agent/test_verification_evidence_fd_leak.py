"""Regression: the async verification ledger closes every connection."""

from __future__ import annotations

from typing import Any

import aiosqlite
import pytest

from agent import verification_evidence as ve

pytestmark = pytest.mark.asyncio


class _TrackingConnection:
    """Delegate to a real aiosqlite connection and record close calls."""

    def __init__(self, real: aiosqlite.Connection, closed: list[int]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed", closed)

    async def close(self) -> None:
        self._closed.append(id(self._real))
        await self._real.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._real, name, value)


def _point_ledger(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ve, "_db_path", lambda: tmp_path / "verification_evidence.db"
    )


def _track_connections(monkeypatch) -> tuple[list[int], list[int]]:
    opened: list[int] = []
    closed: list[int] = []
    real_connect = ve.aiosqlite.connect

    async def tracking_connect(*args, **kwargs):
        conn = await real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _TrackingConnection(conn, closed)

    monkeypatch.setattr(ve.aiosqlite, "connect", tracking_connect)
    return opened, closed


def _python_project(root) -> None:
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")


async def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _point_ledger(monkeypatch, tmp_path)
    _python_project(tmp_path)
    opened, closed = _track_connections(monkeypatch)

    await ve.record_terminal_result(
        command="python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 passed",
    )
    await ve.verification_status(session_id="s1", cwd=tmp_path)
    await ve.mark_workspace_edited(
        session_id="s1", cwd=tmp_path, paths=["mod.py"]
    )

    assert opened
    assert opened == closed


async def test_exception_during_operation_still_closes_connection(
    monkeypatch, tmp_path
):
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    with pytest.raises(aiosqlite.IntegrityError):
        async with ve._transaction() as conn:
            await conn.execute("INSERT INTO verification_events (id) VALUES (1)")

    assert len(opened) == 1
    assert opened == closed


async def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    _point_ledger(monkeypatch, tmp_path)
    opened: list[int] = []
    closed: list[int] = []
    real_connect = ve.aiosqlite.connect

    class _FailingSchemaConnection(_TrackingConnection):
        async def executescript(self, _sql: str):
            raise aiosqlite.OperationalError("simulated schema init failure")

    async def tracking_connect(*args, **kwargs):
        conn = await real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _FailingSchemaConnection(conn, closed)

    monkeypatch.setattr(ve.aiosqlite, "connect", tracking_connect)

    with pytest.raises(aiosqlite.OperationalError):
        async with ve._transaction():
            pass

    assert len(opened) == 1
    assert opened == closed
