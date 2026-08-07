"""Native-async coverage for state database writability preflight."""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest
from blockbuster import BlockBuster

import hermes_state
from hermes_state import SessionDB, preflight_db_writability


pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX chmod semantics"),
    pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses file permission checks",
    ),
]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _make_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE t (x)")
        connection.execute("INSERT INTO t VALUES (1)")
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_repairs_readonly_db_inside_home(hermes_home):
    path = hermes_home / "state.db"
    _make_db(path)
    os.chmod(path, 0o444)

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await preflight_db_writability(path, db_label="state.db")
    finally:
        blockbuster.deactivate()

    assert os.access(path, os.W_OK)


@pytest.mark.asyncio
async def test_refuses_readonly_db_outside_home(hermes_home, tmp_path):
    directory = tmp_path / "elsewhere"
    directory.mkdir()
    path = directory / "custom.db"
    _make_db(path)
    os.chmod(path, 0o444)
    try:
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                await preflight_db_writability(path, db_label="custom.db")
        finally:
            blockbuster.deactivate()
        message = str(exc_info.value)
        assert str(path) in message
        assert "chmod u+rw" in message
        assert not os.access(path, os.W_OK)
    finally:
        os.chmod(path, 0o644)


@pytest.mark.asyncio
async def test_healthy_db_is_untouched(hermes_home):
    path = hermes_home / "state.db"
    _make_db(path)
    before = stat.S_IMODE(path.stat().st_mode)

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await preflight_db_writability(path)
    finally:
        blockbuster.deactivate()

    assert stat.S_IMODE(path.stat().st_mode) == before


@pytest.mark.asyncio
async def test_sessiondb_self_heals_readonly_db_in_home(hermes_home):
    path = hermes_home / "state.db"
    first = SessionDB(path)
    await first.ensure_session("first")
    await first.close()
    os.chmod(path, 0o444)

    database = SessionDB(path)
    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await database.ensure_session("second")
    finally:
        blockbuster.deactivate()
        await database.close()

    assert os.access(path, os.W_OK)


@pytest.mark.asyncio
async def test_sessiondb_records_actionable_preflight_error(
    hermes_home,
    tmp_path,
):
    directory = tmp_path / "custom-loc"
    directory.mkdir()
    path = directory / "state.db"
    _make_db(path)
    os.chmod(path, 0o444)
    hermes_state._set_last_init_error(None)
    database = SessionDB(path)
    try:
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            with pytest.raises(sqlite3.OperationalError, match="chmod"):
                await database.get_session("missing")
        finally:
            blockbuster.deactivate()
        assert str(path) in (hermes_state.get_last_init_error() or "")
    finally:
        os.chmod(path, 0o644)
        hermes_state._set_last_init_error(None)
        await database.close()
