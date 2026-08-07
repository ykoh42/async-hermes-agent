"""Parity tests for the v2026.8.3 CJK-bigram FTS index."""

import asyncio
import shutil
import sqlite3
import time
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SCHEMA_SQL, SessionDB


REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "native" / "fts5_cjk" / "fts5_cjk.c"
VENDOR = REPO / "native" / "fts5_cjk" / "vendor"


@pytest_asyncio.fixture(scope="session")
async def cjk_so(tmp_path_factory):
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None or not SRC.exists():
        pytest.skip("no C toolchain / tokenizer source")
    output = tmp_path_factory.mktemp("fts5cjk") / "libfts5_cjk.so"
    process = await asyncio.create_subprocess_exec(
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{VENDOR}",
        str(SRC),
        "-o",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode:
        pytest.skip(
            f"tokenizer build failed: {stderr.decode(errors='replace')[:200]}"
        )

    probe = await aiosqlite.connect(":memory:")
    try:
        await probe.enable_load_extension(True)
        await probe.load_extension(str(output))
    except Exception as exc:
        pytest.skip(f"extension loading unavailable: {exc}")
    finally:
        await probe.close()
    return output


@pytest_asyncio.fixture
async def db(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    database = SessionDB(db_path=tmp_path / "state.db")
    await database.create_session(session_id="s1", source="cli", model="m")
    assert database._fts_cjk_loaded
    assert database._fts_cjk_available
    await database.append_message(
        "s1", role="user", content="웅기가 shared default 프로필을 요청했다"
    )
    await database.append_message(
        "s1", role="assistant", content="일본 MCP 후보 우선순위 정리했습니다"
    )
    await database.append_message(
        "s1", role="user", content="graphiti daemon looks healthy"
    )
    await database.append_message(
        "s1", role="tool", content="일본 tool output blob", tool_name="terminal"
    )
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_two_char_korean_hits_cjk_index(db):
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            rows = await db.search_messages("웅기", limit=10)
            assert rows and "웅기" in rows[0]["snippet"]
            assert await db.search_messages("일본", limit=10)
        finally:
            blocker.deactivate()


@pytest.mark.asyncio
async def test_extension_load_does_not_block_or_leak(
    cjk_so, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    database = SessionDB(db_path=tmp_path / "state.db")
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            await database.create_session("s1", source="test")
            assert database._fts_cjk_loaded
            assert database._fts_cjk_available
            await database.close()
        finally:
            blocker.deactivate()


@pytest.mark.asyncio
async def test_mixed_and_ascii_queries(db):
    assert await db.search_messages("graphiti", limit=10)
    assert await db.search_messages('"shared default" AND 웅기', limit=10)
    assert await db.search_messages("우선순위", limit=10)


@pytest.mark.asyncio
async def test_lone_single_cjk_char_routes_like(db):
    assert db._describe_search_path("가") == "like_scan"
    assert await db.search_messages("가", limit=10)


@pytest.mark.asyncio
async def test_config_toggle_disables_cjk(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    monkeypatch.setenv("HERMES_CJK_FTS", "0")
    database = SessionDB(db_path=tmp_path / "state.db")
    try:
        connection = await database._get_connection()
        assert not database._fts_cjk_loaded
        assert not database._fts_cjk_available
        row = await (
            await connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'messages_fts_cjk'"
            )
        ).fetchone()
        assert row is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_existing_v23_db_gains_cjk_via_optimize(
    cjk_so, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "absent.so"))
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    await old.create_session(session_id="s1", source="cli", model="m")
    for index in range(10):
        await old.append_message(
            "s1", role="user", content=f"기존 메시지 {index}"
        )
    await old.close()

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    database = SessionDB(db_path=db_path)
    try:
        await database.message_count("s1")
        assert database._fts_cjk_loaded
        assert not database._fts_cjk_available
        status = await database.fts_cjk_rebuild_status()
        assert status is not None and status["pending"]
        assert await database.fts_optimize_available()
        await database.append_message("s1", role="user", content="새로운 메시지")
        assert await database.search_messages("기존", limit=10)

        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
        ):
            blocker = BlockBuster()
            blocker.activate()
            try:
                result = await database.optimize_fts_storage(vacuum=False)
            finally:
                blocker.deactivate()
        assert result["ok"]
        assert database._fts_cjk_available
        assert await database.fts_cjk_rebuild_status() is None
        assert database._describe_search_path("기존") == "fts_cjk"
        assert len(await database.search_messages("기존", limit=20)) == 10
        assert await database.search_messages("새로운", limit=10)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tokenizer_gap_marks_cjk_stale_and_rebuilds(
    cjk_so, tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    original = SessionDB(db_path=db_path)
    await original.create_session(session_id="s1", source="cli", model="m")
    await original.append_message("s1", role="user", content="서울 원본")
    await original.close()

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "absent.so"))
    without_tokenizer = SessionDB(db_path=db_path)
    await without_tokenizer.append_message("s1", role="user", content="부산 누락")
    assert await without_tokenizer.get_meta("fts_cjk_stale") == "1"
    await without_tokenizer.close()

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    recovered = SessionDB(db_path=db_path)
    try:
        await recovered.message_count("s1")
        assert not recovered._fts_cjk_available
        assert await recovered.fts_optimize_available()
        result = await recovered.optimize_fts_storage(vacuum=False)
        assert result["ok"]
        assert recovered._fts_cjk_available
        assert await recovered.get_meta("fts_cjk_stale") is None
        assert await recovered.search_messages("서울", limit=10)
        assert await recovered.search_messages("부산", limit=10)
    finally:
        await recovered.close()


@pytest.mark.asyncio
async def test_legacy_v22_optimize_lands_on_cjk(
    cjk_so, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "state.db"
    connection = await aiosqlite.connect(db_path)
    try:
        await connection.executescript(SCHEMA_SQL)
        await connection.executescript(
            """
            DROP TABLE IF EXISTS messages_fts;
            DROP TABLE IF EXISTS messages_fts_trigram;
            DROP VIEW IF EXISTS messages_fts_trigram_src;
            CREATE VIRTUAL TABLE messages_fts USING fts5(content);
            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content)
                VALUES (new.id, COALESCE(new.content,''));
            END;
            """
        )
        await connection.execute("DELETE FROM schema_version")
        await connection.execute(
            "INSERT INTO schema_version (version) VALUES (10)"
        )
        await connection.execute(
            "INSERT INTO sessions (id, source, started_at) "
            "VALUES ('s1', 'cli', ?)",
            (time.time(),),
        )
        for role, content in (
            ("user", "레거시 일본 메시지"),
            ("assistant", "legacy english reply"),
            ("tool", "레거시 tool output"),
        ):
            await connection.execute(
                "INSERT INTO messages (session_id, timestamp, role, content) "
                "VALUES ('s1', ?, ?, ?)",
                (time.time(), role, content),
            )
        await connection.commit()
    finally:
        await connection.close()

    database = SessionDB(db_path=db_path)
    try:
        await database.message_count("s1")
        assert await database.fts_optimize_available()
        assert not database._fts_cjk_available
        result = await database.optimize_fts_storage(vacuum=False)
        assert result["ok"]
        assert database._fts_cjk_available
        assert await database.fts_cjk_rebuild_status() is None
        assert database._describe_search_path("일본") == "fts_cjk"
        assert await database.search_messages("일본", limit=10)
        assert await database.search_messages("legacy english", limit=10)
        connection = await database._get_connection()
        indexed = await (
            await connection.execute("SELECT COUNT(*) FROM messages_fts_cjk")
        ).fetchone()
        non_tool = await (
            await connection.execute(
                "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
            )
        ).fetchone()
        assert indexed[0] == non_tool[0]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_pure_latin_embedded_in_cjk_recovered_via_cjk_index(db):
    await db.append_message("s1", role="user", content="修改youer服务端的계획")
    rows = await db.search_messages("youer", limit=10)
    assert rows and "youer" in rows[0]["snippet"]
    await db.append_message("s1", role="user", content="에러코드ab확인")
    assert await db.search_messages("ab", limit=10)


@pytest.mark.asyncio
async def test_fresh_db_index_counts_exclude_tool_rows(db):
    connection = await db._get_connection()
    indexed = await (
        await connection.execute("SELECT COUNT(*) FROM messages_fts_cjk")
    ).fetchone()
    non_tool = await (
        await connection.execute(
            "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
        )
    ).fetchone()
    assert indexed[0] == non_tool[0]


@pytest.mark.asyncio
async def test_integrity_after_lifecycle(db):
    await db.append_message("s1", role="user", content="무결성 검사")
    connection = await db._get_connection()
    await connection.execute(
        "INSERT INTO messages_fts_cjk(messages_fts_cjk) "
        "VALUES('integrity-check')"
    )
