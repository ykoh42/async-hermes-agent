#!/usr/bin/env python3
"""Native-async SQLite session state for Hermes Agent."""

import asyncio
import datetime
import errno
import hashlib
import json
import logging
import os
import platform
import random
import re
import sqlite3
import stat
import sys
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
import aiofiles.os

from agent.memory_manager import sanitize_context
from agent.message_sanitization import _sanitize_surrogates
from agent.skill_commands import describe_skill_invocation
from hermes_constants import get_hermes_home
from hermes_state_common import (
    DEFERRED_INDEX_SQL,
    FTS_CJK_STALE_KEY,
    FTS_CJK_TABLE_SQL,
    FTS_CJK_TRIGGER_SQL,
    FTS_SQL,
    FTS_STORAGE_VERSION,
    FTS_TRIGRAM_SQL,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    MAX_FTS5_QUERY_CHARS,
    SCHEMA_VERSION,
    SCHEMA_SQL,
    _COMPRESSION_CHILD_SQL,
    _FTS_CJK_TRIGGERS,
    _FTS_TRIGGERS,
    _LISTABLE_CHILD_SQL,
    _PREVIEW_RAW_SELECT,
    _shape_preview,
    _sql_session_last_active,
    _sql_session_last_active_by_id,
)

try:
    import psutil
except ImportError:  # pragma: no cover - minimal installs
    psutil = None

logger = logging.getLogger(__name__)
_COMPRESSION_LOCK_HOLDER_PID_RE = re.compile(r"(?:^|:)pid=(\d+)(?::|$)")
_chmod = aiofiles.os.wrap(os.chmod)
_realpath = aiofiles.os.wrap(os.path.realpath)
_os_open = aiofiles.os.wrap(os.open)
_os_close = aiofiles.os.wrap(os.close)
_os_lseek = aiofiles.os.wrap(os.lseek)
_utime = aiofiles.os.wrap(os.utime)
_live_connection_counts: Dict[str, int] = {}

_DISK_FULL_MARKERS = (
    "no space left on device",
    "not enough space",
    "database or disk is full",
    "disk full",
    "full disk",
    "enospc",
)

_MALFORMED_SCHEMA_MARKERS = (
    "malformed database schema",
    "database disk image is malformed",
)

_WAL_INCOMPAT_MARKERS = (
    "locking protocol",
    "not authorized",
    "disk i/o error",
)

_last_init_error: Optional[str] = None
_wal_fallback_warned_paths: set[str] = set()
_wal_reset_bug_warned_paths: set[str] = set()
_repair_attempted_paths: set[str] = set()


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record or clear the most recent state database initialization error."""
    global _last_init_error
    _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Return the most recent state database initialization error."""
    return _last_init_error


def format_session_db_unavailable(
    prefix: str = "Session database not available",
) -> str:
    """Format an unavailable-state error with its captured cause."""
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    hint = ""
    if any(marker in cause.lower() for marker in _WAL_INCOMPAT_MARKERS):
        hint = (
            " (state.db may be on NFS/SMB/FUSE/ZFS — see "
            "https://www.sqlite.org/wal.html)"
        )
    return f"{prefix}: {cause}{hint}."


async def _on_disk_journal_mode(conn) -> Optional[str]:
    """Read the effective journal mode without changing it."""
    try:
        row = await (await conn.execute("PRAGMA journal_mode")).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None or row[0] is None:
        return None
    mode = row[0]
    if isinstance(mode, bytes):
        try:
            mode = mode.decode("ascii")
        except UnicodeDecodeError:
            return None
    return str(mode).strip().lower()


async def _apply_macos_checkpoint_barrier(conn) -> None:
    """Enable SQLite's macOS checkpoint durability barrier, best effort."""
    if sys.platform != "darwin":
        return
    try:
        await conn.execute("PRAGMA checkpoint_fullfsync=1")
    except sqlite3.OperationalError:
        pass


async def _enforce_macos_synchronous_full(conn) -> None:
    """Use FULL synchronous mode for WAL connections on macOS."""
    if sys.platform != "darwin":
        return
    try:
        await conn.execute("PRAGMA synchronous=FULL")
    except sqlite3.OperationalError:
        pass


def is_sqlite_wal_reset_vulnerable(
    version_info: Optional[tuple] = None,
) -> bool:
    """Return whether the linked SQLite contains the WAL-reset bug."""
    values = list(
        version_info if version_info is not None else sqlite3.sqlite_version_info
    )
    values.extend([0] * (3 - len(values)))
    info = tuple(int(value) for value in values[:3])
    if info < (3, 7, 0) or info >= (3, 51, 3):
        return False
    if (3, 50, 7) <= info < (3, 51, 0):
        return False
    if (3, 44, 6) <= info < (3, 45, 0):
        return False
    return True


async def sqlite_source_id() -> str:
    """Return ``sqlite_source_id()``, or an empty string when unavailable."""
    import aiosqlite

    try:
        connection = await aiosqlite.connect(":memory:")
        try:
            row = await (
                await connection.execute("SELECT sqlite_source_id()")
            ).fetchone()
        finally:
            await connection.close()
    except sqlite3.Error:
        return ""
    return str(row[0]) if row and row[0] is not None else ""


async def resolve_journal_mode() -> str:
    """Return the configured journal mode (``wal`` or ``delete``)."""
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly() or {}
        database = config.get("database", {})
        if not isinstance(database, dict):
            return "wal"
        raw = database.get("journal_mode", "wal")
    except Exception:
        return "wal"
    if not isinstance(raw, str):
        return "wal"
    mode = raw.strip().lower()
    return mode if mode in {"wal", "delete"} else "wal"


class WalUnsupportedError(sqlite3.OperationalError):
    """Raised when a caller requires WAL but the filesystem refuses it."""


def _log_wal_reset_bug_once(db_label: str, *, kept_wal: bool) -> None:
    if db_label in _wal_reset_bug_warned_paths:
        return
    _wal_reset_bug_warned_paths.add(db_label)
    action = (
        "is already in WAL mode — leaving WAL in place (no live downgrade "
        "under concurrent openers)"
        if kept_wal
        else "using journal_mode=DELETE instead of enabling WAL"
    )
    logger.warning(
        "%s: linked SQLite %s is vulnerable to the WAL-reset corruption "
        "bug (https://sqlite.org/wal.html#walresetbug) — %s. Upgrade to "
        "SQLite 3.51.3+ (or backports 3.50.7 / 3.44.6).",
        db_label,
        sqlite3.sqlite_version,
        action,
    )


def _log_wal_fallback_once(db_label: str, exc: Exception) -> None:
    if db_label in _wal_fallback_warned_paths:
        return
    _wal_fallback_warned_paths.add(db_label)
    logger.error(
        "%s: WAL journal_mode unsupported on this filesystem (%s) — "
        "falling back to journal_mode=DELETE. See "
        "https://www.sqlite.org/wal.html for details.",
        db_label,
        exc,
    )


async def _apply_delete_for_wal_reset_bug(
    conn,
    *,
    db_label: str,
    require_delete: bool = False,
) -> str:
    current = await _on_disk_journal_mode(conn) or ""
    if current == "wal":
        _log_wal_reset_bug_once(db_label, kept_wal=True)
        await _apply_macos_checkpoint_barrier(conn)
        await _enforce_macos_synchronous_full(conn)
        return "wal"

    actual = ""
    try:
        row = await (
            await conn.execute("PRAGMA journal_mode=DELETE")
        ).fetchone()
        if row and row[0] is not None:
            actual = str(row[0]).strip().lower()
    except sqlite3.OperationalError:
        if require_delete:
            raise
    if require_delete and actual != "delete":
        raise sqlite3.OperationalError(
            "could not set configured journal_mode=delete "
            f"(got {actual or 'no result'})"
        )
    _log_wal_reset_bug_once(db_label, kept_wal=False)
    return "delete"


async def apply_wal_with_fallback(
    conn,
    *,
    db_label: str = "state.db",
    require_wal: bool = False,
) -> str:
    """Enable WAL safely, preserving upstream fallback and durability rules."""
    configured = await resolve_journal_mode()
    if is_sqlite_wal_reset_vulnerable():
        return await _apply_delete_for_wal_reset_bug(
            conn,
            db_label=db_label,
            require_delete=configured == "delete",
        )

    current = await _on_disk_journal_mode(conn)
    if current == "wal":
        await _apply_macos_checkpoint_barrier(conn)
        await _enforce_macos_synchronous_full(conn)
        return "wal"

    if configured == "delete":
        row = await (
            await conn.execute("PRAGMA journal_mode=DELETE")
        ).fetchone()
        actual = str(row[0]).lower() if row else ""
        if actual != "delete":
            raise sqlite3.OperationalError(
                "could not set configured journal_mode=delete "
                f"(got {actual or 'no result'})"
            )
        return actual

    try:
        row = await (
            await conn.execute("PRAGMA journal_mode=WAL")
        ).fetchone()
        mode = str(row[0]).strip().lower() if row and row[0] is not None else ""
        if mode == "wal":
            await _apply_macos_checkpoint_barrier(conn)
            await _enforce_macos_synchronous_full(conn)
            return "wal"
        refusal = WalUnsupportedError(
            f"journal_mode=WAL refused without raising (still {mode!r})"
        )
        if require_wal:
            raise refusal
        _log_wal_fallback_once(db_label, refusal)
        return mode or "delete"
    except sqlite3.OperationalError as caught:
        exc = caught

    if isinstance(exc, WalUnsupportedError):
        raise exc
    message = str(exc).lower()
    if not any(marker in message for marker in _WAL_INCOMPAT_MARKERS):
        raise exc
    if "disk i/o error" in message:
        for _ in range(2):
            await asyncio.sleep(0.05)
            try:
                row = await (
                    await conn.execute("PRAGMA journal_mode=WAL")
                ).fetchone()
            except sqlite3.OperationalError as retry_exc:
                if "disk i/o error" not in str(retry_exc).lower():
                    raise
                exc = retry_exc
                continue
            mode = (
                str(row[0]).strip().lower()
                if row and row[0] is not None
                else ""
            )
            if mode == "wal":
                await _apply_macos_checkpoint_barrier(conn)
                await _enforce_macos_synchronous_full(conn)
                return "wal"
            break
    if await _on_disk_journal_mode(conn) == "wal":
        raise exc
    if require_wal:
        raise WalUnsupportedError(str(exc)) from exc
    _log_wal_fallback_once(db_label, exc)
    await conn.execute("PRAGMA journal_mode=DELETE")
    return "delete"


async def apply_database_pragmas(
    conn,
    *,
    db_label: str = "state.db",
) -> None:
    """Apply optional integer database PRAGMAs from ``config.yaml``."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        config = await load_config_readonly()
    except Exception:
        return
    for pragma_name in (
        "cache_size",
        "mmap_size",
        "temp_store",
        "wal_autocheckpoint",
        "journal_size_limit",
    ):
        raw_value = cfg_get(config, "database", pragma_name, default=None)
        if raw_value is None:
            continue
        try:
            value = int(str(raw_value).strip())
        except (TypeError, ValueError):
            logger.warning(
                "%s: ignoring non-integer database.%s=%r",
                db_label,
                pragma_name,
                raw_value,
            )
            continue
        try:
            await conn.execute(f"PRAGMA {pragma_name}={value}")
        except sqlite3.OperationalError:
            pass


def is_malformed_db_error(exc: BaseException) -> bool:
    """Return whether SQLite reported malformed schema or database bytes."""
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in str(exc).lower() for marker in _MALFORMED_SCHEMA_MARKERS
    )


def _claim_repair_attempt(db_path: Path) -> bool:
    """Claim the process-local one-shot repair attempt for ``db_path``."""
    key = str(db_path)
    if key in _repair_attempted_paths:
        return False
    _repair_attempted_paths.add(key)
    return True


async def _backup_db_file(db_path: Path) -> Optional[Path]:
    """Copy a malformed database and its sidecars beside the original."""
    tracking_key = str(await _realpath(str(db_path)))
    if _live_connection_counts.get(tracking_key, 0) > 0:
        logger.error(
            "Refusing to raw-copy %s for backup: a connection to it is still "
            "open in this process. Close all SessionDB handles first.",
            db_path,
        )
        return None

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(
        f"{db_path.name}.malformed-backup-{stamp}"
    )
    try:
        for source, destination in (
            (db_path, backup_path),
            (
                db_path.with_name(db_path.name + "-wal"),
                backup_path.with_name(backup_path.name + "-wal"),
            ),
            (
                db_path.with_name(db_path.name + "-shm"),
                backup_path.with_name(backup_path.name + "-shm"),
            ),
        ):
            if not await aiofiles.os.path.exists(source):
                continue
            metadata = await aiofiles.os.stat(source)
            async with (
                aiofiles.open(source, "rb") as reader,
                aiofiles.open(destination, "wb") as writer,
            ):
                while chunk := await reader.read(1024 * 1024):
                    await writer.write(chunk)
            await _chmod(destination, stat.S_IMODE(metadata.st_mode))
            await _utime(
                destination,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
            )
        return backup_path
    except Exception as exc:
        logger.warning("Could not back up malformed DB %s: %s", db_path, exc)
        return None


async def _db_opens_cleanly(db_path: Path) -> Optional[str]:
    """Probe database integrity plus the FTS read and write paths."""
    import aiosqlite

    conn = await aiosqlite.connect(db_path, isolation_level=None)
    try:
        await load_fts5_cjk_extension(conn)
        await (await conn.execute("PRAGMA journal_mode")).fetchone()
        rows = await (await conn.execute("PRAGMA integrity_check")).fetchall()
        problems = [
            str(row[0])
            for row in rows
            if row and str(row[0]).lower() != "ok"
        ]
        if problems:
            return "; ".join(problems[:3])
        await (await conn.execute("SELECT COUNT(*) FROM sessions")).fetchone()

        for fts_table in (
            "messages_fts",
            "messages_fts_trigram",
            "messages_fts_cjk",
        ):
            try:
                await (
                    await conn.execute(
                        f"SELECT 1 FROM {fts_table} "
                        f"WHERE {fts_table} MATCH '\"\"' LIMIT 1"
                    )
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if SessionDB._is_fts5_unavailable_error(exc):
                    continue
                message = str(exc).lower()
                if "no such table" in message or "no such column" in message:
                    continue
                return f"fts5 read probe failed on {fts_table}: {exc}"
            except sqlite3.DatabaseError as exc:
                return f"fts5 read probe failed on {fts_table}: {exc}"

        probe_session_id = f"_hermes_fts_health_probe_{time.time_ns()}"
        write_error: Optional[sqlite3.OperationalError] = None
        try:
            await conn.execute("BEGIN IMMEDIATE")
            await conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (probe_session_id, "_health_probe", time.time()),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (probe_session_id, "user", "_fts_health_probe", time.time()),
            )
            await conn.execute("ROLLBACK")
        except sqlite3.OperationalError as caught:
            write_error = caught
        if write_error is not None:
            try:
                await conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            message = str(write_error).lower()
            if "no such table" in message or "no such column" in message:
                return None
            if "no such tokenizer: cjk_unicode61" in message:
                return None
            return str(write_error)
        return None
    except sqlite3.DatabaseError as exc:
        return str(exc)
    finally:
        await conn.close()


async def repair_state_db_schema(
    db_path: Path,
    *,
    backup: bool = True,
) -> Dict[str, Any]:
    """Repair malformed schema, FTS, and stale B-tree index damage."""
    import aiosqlite

    report: Dict[str, Any] = {
        "repaired": False,
        "strategy": None,
        "backup_path": None,
        "error": None,
    }
    db_path = Path(db_path)
    if not await aiofiles.os.path.exists(db_path):
        report["error"] = f"{db_path} does not exist"
        return report
    if await _db_opens_cleanly(db_path) is None:
        report["repaired"] = True
        report["strategy"] = "already_healthy"
        return report
    if backup:
        backup_path = await _backup_db_file(db_path)
        report["backup_path"] = str(backup_path) if backup_path else None

    try:
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        try:
            await load_fts5_cjk_extension(conn)
            for table_name in (
                "messages_fts",
                "messages_fts_trigram",
                "messages_fts_cjk",
            ):
                try:
                    await conn.execute(
                        f"INSERT INTO {table_name}({table_name}) "
                        "VALUES('rebuild')"
                    )
                except sqlite3.OperationalError:
                    continue
        finally:
            await conn.close()
        if await _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "rebuild_fts"
            logger.warning(
                "state.db FTS indexes rebuilt in place (schema preserved): %s",
                db_path,
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db FTS in-place rebuild pass failed: %s", exc)

    try:
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        try:
            await conn.execute("REINDEX")
            await conn.commit()
        finally:
            await conn.close()
        if await _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "reindex_btree"
            logger.warning(
                "state.db B-tree indexes rebuilt via REINDEX: %s",
                db_path,
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db REINDEX pass failed: %s", exc)

    try:
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        try:
            await conn.execute("PRAGMA writable_schema=ON")
            duplicates = await (
                await conn.execute(
                    "SELECT type, name, COUNT(*) AS c, MIN(rowid) AS keep "
                    "FROM sqlite_master GROUP BY type, name HAVING c > 1"
                )
            ).fetchall()
            for type_, name, _count, keep in duplicates:
                await conn.execute(
                    "DELETE FROM sqlite_master "
                    "WHERE type IS ? AND name IS ? AND rowid <> ?",
                    (type_, name, keep),
                )
            await conn.execute("PRAGMA writable_schema=OFF")
            await conn.commit()
        finally:
            await conn.close()
        if await _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "dedup_schema"
            logger.warning(
                "state.db schema repaired by de-duplicating sqlite_master "
                "(FTS index preserved): %s",
                db_path,
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db dedup repair pass failed: %s", exc)

    try:
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        try:
            await conn.execute("PRAGMA writable_schema=ON")
            await conn.execute(
                "DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'"
            )
            await conn.execute("PRAGMA writable_schema=OFF")
            await conn.commit()
            await conn.execute("VACUUM")
        finally:
            await conn.close()
        reason = await _db_opens_cleanly(db_path)
        if reason is None:
            report["repaired"] = True
            report["strategy"] = "drop_fts_rebuild"
            logger.warning(
                "state.db schema repaired by dropping FTS schema; indexes "
                "will rebuild from messages on next open: %s",
                db_path,
            )
            return report
        report["error"] = reason
    except sqlite3.DatabaseError as exc:
        report["error"] = str(exc)

    if not report["repaired"]:
        logger.error(
            "state.db schema repair could not recover %s automatically "
            "(backup: %s); manual restore from backup may be required.",
            db_path,
            report["backup_path"],
        )
    return report


async def preflight_db_writability(
    db_path: Path,
    *,
    db_label: str = "state.db",
) -> None:
    """Repair or reject read-only database files before opening SQLite."""
    raw = str(db_path)
    if raw == ":memory:" or raw.startswith("file:"):
        return

    try:
        home = Path(await _realpath(str(get_hermes_home())))
    except Exception:
        home = None

    async def _in_repair_scope(path: Path) -> bool:
        if home is None:
            return False
        try:
            resolved = Path(await _realpath(str(path)))
            return resolved.is_relative_to(home)
        except (OSError, ValueError):
            return False

    async def _ensure_writable(path: Path, *, is_dir: bool = False) -> None:
        if await aiofiles.os.access(path, os.R_OK | os.W_OK):
            return
        if await _in_repair_scope(path):
            try:
                mode = (await aiofiles.os.stat(path)).st_mode
                additions = stat.S_IRUSR | stat.S_IWUSR
                if is_dir:
                    additions |= stat.S_IXUSR
                await _chmod(path, mode | additions)
            except OSError:
                pass
            if await aiofiles.os.access(path, os.R_OK | os.W_OK):
                logger.info(
                    "%s preflight: repaired read-only %s (chmod u+rw%s)",
                    db_label,
                    path,
                    "x" if is_dir else "",
                )
                return
        kind = "directory" if is_dir else "file"
        wal_note = (
            " Do NOT delete the -wal file — it contains committed data that "
            "will be merged into the database once it is writable."
            if path.name.endswith("-wal")
            else ""
        )
        raise sqlite3.OperationalError(
            f"{db_label} is not writable: {kind} {path} is read-only for this "
            "user. Hermes needs read-write access to open the database. "
            f"Fix with: chmod u+rw{'x' if is_dir else ''} '{path}'"
            f" (files owned by another user may need sudo/chown).{wal_note}"
        )

    parent = db_path.parent
    if await aiofiles.os.path.isdir(parent):
        await _ensure_writable(parent, is_dir=True)
    for suffix in ("", "-wal", "-shm"):
        path = db_path.with_name(db_path.name + suffix) if suffix else db_path
        if await aiofiles.os.path.isfile(path):
            await _ensure_writable(path)


async def is_zeroed_state_db(
    path: Path,
    *,
    probe_bytes: int = 100,
    force: bool = False,
) -> bool:
    """Detect a non-empty state database whose header contains only NULs."""
    try:
        tracking_key = str(await _realpath(str(path)))
        if not force and _live_connection_counts.get(tracking_key, 0) > 0:
            return False
        size = (await aiofiles.os.stat(path)).st_size
    except OSError:
        return False
    if size <= 0:
        return False
    try:
        async with aiofiles.open(path, "rb") as handle:
            head = await handle.read(max(16, probe_bytes))
    except OSError:
        return False
    if not head or head.startswith(b"SQLite format 3"):
        return False
    return all(byte == 0 for byte in head)


async def quarantine_zeroed_state_db(path: Path) -> Optional[Path]:
    """Move a zeroed database aside under a cross-process startup lock."""
    lock_path = path.with_name(path.name + ".quarantine.lock")
    await aiofiles.os.makedirs(lock_path.parent, exist_ok=True)
    descriptor = await _os_open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        deadline = time.monotonic() + 5.0
        if platform.system() == "Windows":
            import msvcrt

            while True:
                try:
                    await _os_lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.020)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.020)
        if not acquired:
            logger.error(
                "quarantine lock for %s not acquired within 5s — refusing "
                "to quarantine without the cross-process lock",
                path,
            )
            return None
        if not await aiofiles.os.path.exists(path):
            return None
        if not await is_zeroed_state_db(path, force=True):
            return None

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        destination = path.with_name(
            f"{path.name}.zeroed-{timestamp}-{os.getpid()}.bak"
        )
        counter = 0
        while await aiofiles.os.path.exists(destination):
            counter += 1
            destination = path.with_name(
                f"{path.name}.zeroed-{timestamp}-{os.getpid()}-{counter}.bak"
            )
        try:
            await aiofiles.os.rename(path, destination)
        except OSError as exc:
            logger.error("Failed to quarantine zeroed %s: %s", path, exc)
            return None
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if await aiofiles.os.path.exists(sidecar):
                try:
                    await aiofiles.os.rename(
                        sidecar,
                        Path(str(destination) + suffix),
                    )
                except OSError:
                    pass
        return destination
    finally:
        try:
            if acquired:
                if platform.system() == "Windows":
                    import msvcrt

                    await _os_lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            await _os_close(descriptor)


def fts5_cjk_so_path() -> Path:
    """Location of the cjk_unicode61 loadable extension."""
    env = os.getenv("HERMES_FTS5_CJK_SO")
    if env:
        return Path(env).expanduser()
    return get_hermes_home() / "lib" / "libfts5_cjk.so"


def _cjk_fts_config_enabled() -> bool:
    """config.yaml ``sessions.cjk_fts`` (default on), via its env bridge."""
    return os.getenv("HERMES_CJK_FTS", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


async def load_fts5_cjk_extension(conn) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer into ``connection``."""
    if not _cjk_fts_config_enabled():
        return False
    path = fts5_cjk_so_path()
    if not await aiofiles.os.path.exists(path):
        return False
    try:
        await conn.enable_load_extension(True)
        try:
            await conn.load_extension(str(path))
        finally:
            await conn.enable_load_extension(False)
        return True
    except Exception:
        logger.warning("fts5_cjk extension load failed (%s)", path, exc_info=True)
        return False


def _system_prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def is_disk_full_error(exc: BaseException | str | None) -> bool:
    """Return whether an error reports ENOSPC or SQLite's disk-full state."""
    if exc is None:
        return False
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    text = exc if isinstance(exc, str) else str(exc)
    return any(marker in text.lower() for marker in _DISK_FULL_MARKERS)


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """Return true only when a structured local compression owner is gone."""
    match = _COMPRESSION_LOCK_HOLDER_PID_RE.search(holder or "")
    if match is None:
        return False
    try:
        pid = int(match.group(1))
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    if psutil is not None:
        try:
            return not psutil.pid_exists(pid)
        except Exception:
            return False
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError, OverflowError):
        return False
    return False


def _scrub_surrogates(value: Any) -> Any:
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def workspace_key(row: Dict[str, Any]) -> Optional[str]:
    """A session's workspace grouping key: its git repo root when known, else
    its cwd.

    Branch is deliberately excluded so checking out a new branch doesn't
    fragment a workspace's session history. Returns None for cwd-less (unbound)
    sessions. Both fields are already recorded on ``sessions`` — this just picks
    the coarser identity for grouping/filtering.
    """
    root = (row.get("git_repo_root") or "").strip()
    if root:
        return root

    cwd = (row.get("cwd") or "").strip()
    return cwd or None


def _delegate_from_json(col: str = "model_config") -> str:
    return f"json_extract(COALESCE({col}, '{{}}'), '$._delegate_from')"


async def _collect_delegate_child_ids(connection, parent_ids: List[str]) -> List[str]:
    """Return recursively discovered delegate children, excluding parents."""
    delegate_from = _delegate_from_json()
    seeds = {session_id for session_id in parent_ids if session_id}
    found = set(seeds)
    frontier = list(seeds)
    while frontier:
        placeholders = ",".join("?" for _ in frontier)
        rows = await (
            await connection.execute(
                f"SELECT id FROM sessions WHERE {delegate_from} IN ({placeholders}) "
                f"OR (parent_session_id IN ({placeholders}) "
                f"AND {delegate_from} IS NOT NULL)",
                [*frontier, *frontier],
            )
        ).fetchall()
        frontier = [row["id"] for row in rows if row["id"] not in found]
        found.update(frontier)
    return [session_id for session_id in found if session_id not in seeds]


async def _delete_delegate_children(connection, parent_ids: List[str]) -> List[str]:
    session_ids = await _collect_delegate_child_ids(connection, parent_ids)
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        await connection.execute(
            f"DELETE FROM messages WHERE session_id IN ({placeholders})",
            session_ids,
        )
        await connection.execute(
            f"UPDATE sessions SET parent_session_id = NULL "
            f"WHERE parent_session_id IN ({placeholders})",
            session_ids,
        )
        await connection.execute(
            f"DELETE FROM sessions WHERE id IN ({placeholders})",
            session_ids,
        )
    return session_ids


def _cwd_prefix_clause(cwd_prefix: str) -> Tuple[str, List[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    return "(s.cwd = ? OR s.cwd LIKE ? OR s.cwd LIKE ?)", [
        prefix,
        f"{prefix}/%",
        f"{prefix}\\%",
    ]


def _workspace_key_clause(key: str) -> Tuple[str, List[str]]:
    """Match sessions whose ``workspace_key(row)`` equals ``key``.

    Mirrors :func:`workspace_key`: a session belongs to workspace ``key``
    when its recorded ``git_repo_root`` equals ``key``, or — for rows that
    predate per-session git metadata — when its ``cwd`` is at or under
    ``key`` (so a session started in ``repo/src`` still groups with ``repo``).
    Used by ``hermes -c``/``--resume`` to continue the most recent session in
    the *current* workspace rather than the global MRU.
    """
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR "
        f"(COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


DEFAULT_DB_PATH = get_hermes_home() / "state.db"
_IMPORT_DEFAULT_DB_PATH = DEFAULT_DB_PATH


def _default_db_path() -> Path:
    if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH:
        return DEFAULT_DB_PATH
    return get_hermes_home() / "state.db"


_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(message: Dict[str, Any]) -> bool:
    if not isinstance(message, dict) or message.get("role") not in {"user", "system"}:
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    return any(content.lstrip().startswith(prefix) for prefix in _REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not messages:
        return messages
    result: List[Dict[str, Any]] = []
    skip_next_assistant = False
    for message in messages:
        if _is_background_review_harness_message(message):
            skip_next_assistant = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if isinstance(message, dict) and message.get("role") == "assistant":
                continue
        result.append(message)
    return result


class CompressionSessionClosedError(RuntimeError):
    """A durable write targeted a parent already closed by compression."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} is closed by compression; "
            "adopt its live continuation before appending messages"
        )


class CompressionSessionBusyError(RuntimeError):
    """A non-owner tried to write while compression owns the session."""


class SessionCompressionInProgressError(CompressionSessionBusyError):
    """A normal writer collided with another writer's live compression lease."""


class SessionDB:
    """Native-async session store used by the agent turn path.

    The constructor accepts a database path. The SQLite connection and schema
    are created lazily on the first awaited operation, keeping
    ``AIAgent.__init__`` state-only.

    Deliberately do not provide a generic ``__getattr__`` bridge: an unknown
    synchronous database method must fail loudly instead of silently moving
    work into a thread pool.
    """

    _WRITE_PATIENCE_S = 60.0
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S = 0.02
    _WRITE_RETRY_MAX_S = 0.15

    _FTS_REBUILD_CHUNK_ROWS = 500
    _FTS_REBUILD_DUTY_FACTOR = 4.0
    _FTS_REBUILD_MIN_PAUSE = 0.2
    _FTS_TRASH_PREFIX = "fts_v22_trash_"

    _CONTENT_JSON_PREFIX = "\x00json:"
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, timestamp, "
        "api_content, display_kind, display_metadata"
    )

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        err = str(exc).lower()
        if "no such module" in err and "fts5" in err:
            return True
        # SQLite builds that have FTS5 but lack the optional trigram tokenizer
        # raise "no such tokenizer: trigram" instead of "no such module".
        # Scope to trigram specifically to avoid masking unrelated tokenizer errors.
        if "no such tokenizer: trigram" in err:
            return True
        # The cjk_unicode61 tokenizer is a loadable extension — a process
        # that couldn't load it sees the same capability-error shape.
        if "no such tokenizer: cjk_unicode61" in err:
            return True
        return False

    @staticmethod
    def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        err = str(exc).lower()
        return (
            "no such tokenizer: trigram" in err
            or "no such tokenizer: cjk_unicode61" in err
        )


    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize structured (list/dict) message content for sqlite.

        sqlite3 can only bind ``str``, ``bytes``, ``int``, ``float``, and ``None``
        to query parameters. Multimodal messages have ``content`` as a list of
        parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``), which
        raises ``ProgrammingError: Error binding parameter N: type 'list' is
        not supported`` when bound directly.

        Returns the value unchanged when it's already a safe scalar, or a
        sentinel-prefixed JSON string for lists/dicts. Paired with
        :meth:`_decode_content` on read.
        """
        if isinstance(content, str):
            # Lone UTF-16 surrogates reach here inside tool results scraped
            # from the web/social platforms (the same input that crashed the
            # guardrail hasher). The proactive sanitizer upstream only cleans
            # the *api_messages* copy, and the recovery sanitizer only runs
            # after the API call itself raises — which it no longer does — so
            # the canonical history keeps them and this write is where they
            # land. Left raw, sqlite3 raises UnicodeEncodeError, the flush is
            # abandoned, and the session silently stops persisting for the
            # rest of its life. Scrub so persistence never fails.
            return _sanitize_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            # json.dumps defaults to ensure_ascii=True, which escapes any
            # surrogate as \udXXX — already safe to bind.
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            # Last-resort fallback: stringify so persistence never fails.
            return _sanitize_surrogates(str(content))


    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to decode JSON-encoded message content; "
                    "returning raw string"
                )
                return content
        return content


    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """Serialize ``display_metadata`` for its TEXT column without double-encoding.

        Import/replace paths can hand us an already-serialized JSON string (the
        same hazard ``tool_calls`` guards against above). ``json.dumps`` on that
        string would store a quoted JSON string, and the single ``json.loads``
        on read then yields a ``str`` instead of a dict.
        """
        if not display_metadata:
            return None
        if isinstance(display_metadata, str):
            try:
                parsed = json.loads(display_metadata)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Ignoring non-JSON display metadata on write")
                return None
            if not isinstance(parsed, dict):
                logger.warning("Ignoring non-object display metadata on write")
                return None
            return json.dumps(parsed)
        if isinstance(display_metadata, dict):
            return json.dumps(display_metadata)
        logger.warning(
            "Ignoring unexpected display metadata type on write: %s",
            type(display_metadata).__name__,
        )
        return None


    @staticmethod
    def _decode_display_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Decode a ``display_metadata`` column into the dict every reader expects.

        Every message read path must go through this. Returning the raw TEXT
        instead reaches the desktop as a string, where ``'task_count' in meta``
        throws and fails the whole resume. Rows written before the encode guard
        landed are double-encoded, so unwrap a second layer when we find one.
        """
        if raw is None:
            return None
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(meta, str):
                meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid display metadata on message row")
            return None
        if not isinstance(meta, dict):
            logger.warning("Ignoring non-object display metadata on message row")
            return None
        return meta


    @staticmethod
    def _is_duplicate_replayed_user_message(messages: List[Dict[str, Any]], msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        for prev in reversed(messages):
            if prev.get("role") == "user" and prev.get("content") == content:
                return True
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return False
        return False


    def _rows_to_conversation(
        self,
        rows,
        *,
        session_id: str,
        include_ancestors: bool,
        repair_alternation: bool,
        include_row_ids: bool = False,
    ) -> List[Dict[str, Any]]:
        """Decode fetched message rows into the OpenAI conversation format.

        Extracted from get_messages_as_conversation so get_resume_conversations
        can build the model-fed and display views from one SELECT. ``rows`` must
        already be ordered by ``id`` (insertion order) and filtered to the
        desired session set / active state by the caller.
        """
        messages = []
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = row["id"]
            # api_content is the byte-fidelity sidecar: the exact string sent
            # to the API when it differed from the clean content. Returned
            # VERBATIM — no sanitize_context, no strip — because the replay
            # path substitutes it for content to keep the provider prompt
            # cache prefix byte-stable across turns. Cleaning it here would
            # re-introduce the divergence it exists to remove.
            if row["api_content"]:
                msg["api_content"] = row["api_content"]
            if row["display_kind"]:
                msg["display_kind"] = row["display_kind"]
            if row["display_metadata"]:
                decoded = self._decode_display_metadata(row["display_metadata"])
                if decoded is not None:
                    msg["display_metadata"] = decoded
            if row["timestamp"]:
                msg["timestamp"] = row["timestamp"]
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["effect_disposition"]:
                msg["effect_disposition"] = row["effect_disposition"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []
            # Surface the platform-side message id (e.g. yuanbao msg_id,
            # telegram update_id) so platform-specific flows like recall
            # can match by external identifier instead of having to fall
            # back to content-match heuristics.  Exposed as ``message_id``
            # for backward compatibility with the JSONL transcript shape.
            if row["platform_message_id"]:
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["finish_reason"]:
                    msg["finish_reason"] = row["finish_reason"]
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize reasoning_details, falling back to None")
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_reasoning_items, falling back to None")
                        msg["codex_reasoning_items"] = None
                if row["codex_message_items"]:
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_message_items, falling back to None")
                        msg["codex_message_items"] = None
            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)
        # DEFENSE-IN-DEPTH against background-review session pollution: a forked
        # skill/memory review that (in older builds, before the _persist_disabled
        # fix) shared the parent's session_id wrote its harness turn into this
        # real session. The harness is a user/system message instructing the
        # agent to "Review the conversation above and update the skill library /
        # save to memory" under a hard tool restriction; re-loading it as live
        # history makes the agent adopt the curator role and refuse the user's
        # actual task. Strip any such harness message AND the curator-mode
        # assistant reply immediately following it, so a polluted session
        # resumes clean even if stray rows exist.
        messages = _strip_background_review_harness(messages)
        if repair_alternation and messages:
            # Lazy import: hermes_state already depends on agent.* (see
            # sanitize_context above), but keep this optional path from
            # widening the import surface at module load.
            from agent.agent_runtime_helpers import repair_message_sequence

            repaired = repair_message_sequence(None, messages)
            if repaired:
                logger.info(
                    "Repaired %d message-alternation violation(s) while "
                    "restoring session %s — durable transcript kept them, "
                    "see repair_message_sequence",
                    repaired,
                    session_id,
                )
        return messages

    async def list_recent_user_messages(
        self,
        session_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the most-recent real user turns, newest first."""
        active_clause = "" if include_inactive else " AND active = 1"
        connection = await self._get_connection()
        cursor = await connection.execute(
            "SELECT id, timestamp, content FROM messages "
            "WHERE session_id = ? AND role = 'user'"
            f"{active_clause} AND (display_kind IS NULL OR display_kind = '') "
            "ORDER BY id DESC LIMIT ?",
            (session_id, int(limit)),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()

        result: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_content(row["content"])
            if isinstance(decoded, list):
                text_parts = [
                    part.get("text", "")
                    for part in decoded
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                preview = " ".join(text for text in text_parts if text).strip()
                if not preview:
                    preview = "[multimodal content]"
            elif isinstance(decoded, str):
                preview = describe_skill_invocation(decoded) or decoded
            else:
                preview = ""
            preview = " ".join(preview.split())
            if len(preview) > 80:
                preview = preview[:77] + "..."
            result.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "preview": preview,
                }
            )
        return result

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}``, the column-filter operator ``:`` and bare
        boolean operators (``AND``, ``OR``, ``NOT``) have special meaning.
        Passing raw user input directly to MATCH can cause
        ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Cap user-controlled FTS input before any regex processing. Search
        # queries do not need to be arbitrarily large, and bounding them keeps
        # sanitizer/runtime behavior predictable under adversarial input.
        query = query[:MAX_FTS5_QUERY_CHARS]

        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders. Do this with a
        # single linear scan rather than a regex so pathological quote runs
        # cannot induce backtracking.
        _quoted_parts: list = []
        pieces: list[str] = []
        i = 0
        while i < len(query):
            ch = query[i]
            if ch != '"':
                pieces.append(ch)
                i += 1
                continue
            end = query.find('"', i + 1)
            if end == -1:
                # Unmatched quote: replace with whitespace like the old
                # sanitizer's special-char stripping step.
                pieces.append(" ")
                i += 1
                continue
            _quoted_parts.append(query[i:end + 1])
            pieces.append(f"\x00Q{len(_quoted_parts) - 1}\x00")
            i = end + 1

        sanitized = "".join(pieces)

        # Step 2: Strip remaining (unmatched) FTS5-special characters.  ``:`` is
        # FTS5's column-filter operator (``col:term``); since the FTS table has a
        # single ``content`` column, an unquoted colon query like ``TODO: fix``
        # parses as ``column:term`` and raises "no such column" — swallowed at
        # the execute site into zero results.  Strip it like the others.
        sanitized = re.sub(r'[+{}():\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted, hyphenated and underscored
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()

    @staticmethod
    def _is_cjk_codepoint(codepoint: int) -> bool:
        return (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
            or 0x3000 <= codepoint <= 0x303F
            or 0x3040 <= codepoint <= 0x309F
            or 0x30A0 <= codepoint <= 0x30FF
            or 0xAC00 <= codepoint <= 0xD7AF
        )

    @classmethod
    def _contains_cjk(cls, text: str) -> bool:
        """Return whether *text* contains Chinese, Japanese, or Korean text."""
        return any(cls._is_cjk_codepoint(ord(character)) for character in text)

    @classmethod
    def _count_cjk(cls, text: str) -> int:
        return sum(
            cls._is_cjk_codepoint(ord(character)) for character in text
        )

    @classmethod
    def _has_lone_cjk_run(cls, query: str) -> bool:
        run = 0
        for character in query:
            if cls._is_cjk_codepoint(ord(character)):
                run += 1
            else:
                if run == 1:
                    return True
                run = 0
        return run == 1

    @staticmethod
    def _trigram_eligible_tokens(query: str) -> bool:
        tokens = [
            token
            for token in query.strip('"').strip().split()
            if token.upper() not in {"AND", "OR", "NOT"}
        ]
        return bool(tokens) and all(len(token) >= 3 for token in tokens)


    @staticmethod
    async def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        import aiosqlite

        ref = await aiosqlite.connect(":memory:")
        try:
            await ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            cursor = await ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            for (tbl,) in await cursor.fetchall():
                cols: Dict[str, str] = {}
                column_cursor = await ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                )
                for row in await column_cursor.fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            await ref.close()

    def __init__(
        self,
        db_path: "os.PathLike[str] | str" = None,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._db_path = self.db_path
        self.read_only = bool(read_only)
        self._connection = None
        self._connection_tracking_key = None
        self._connect_lock = None
        self._write_lock = None
        self._schema_ready = False
        self._wal_active = False
        self._closed = False
        self._fts_enabled = False
        self._trigram_available = False
        self._fts_cjk_loaded = False
        self._fts_cjk_available = False
        self._fts_runtime_rebuild_attempted = False
        self._fts_unavailable_warned = False
        self._trigram_unavailable_warned = False

    def _get_connect_lock(self) -> asyncio.Lock:
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        return self._connect_lock

    def _get_write_lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def _get_connection(self):
        if self._closed:
            raise RuntimeError("SessionDB is closed")
        if self._connection is not None:
            return self._connection

        async with self._get_connect_lock():
            if self._connection is not None:
                return self._connection
            import aiosqlite

            import aiofiles.os

            async def _connect_and_initialize():
                connection = None
                initialized = False
                try:
                    if not self.read_only and not await aiofiles.os.path.exists(
                        self._db_path.parent
                    ):
                        await aiofiles.os.makedirs(
                            self._db_path.parent,
                            exist_ok=True,
                        )
                    if not self.read_only:
                        await preflight_db_writability(
                            self._db_path,
                            db_label="state.db",
                        )
                        if await is_zeroed_state_db(self._db_path):
                            try:
                                zeroed_size = (
                                    await aiofiles.os.stat(self._db_path)
                                ).st_size
                            except OSError:
                                zeroed_size = -1
                            quarantine_path = await quarantine_zeroed_state_db(
                                self._db_path
                            )
                            snapshots = self._db_path.parent / "state-snapshots"
                            message = (
                                "state.db looks ZEROED "
                                f"({zeroed_size} bytes, no SQLite header). "
                                "Preserved at "
                                f"{quarantine_path or '(quarantine failed — file left in place)'}. "
                                f"Restore from {snapshots} via `hermes snapshot list` / "
                                "`hermes snapshot restore <id>` if available. "
                                "Opening a fresh empty database so the agent can start."
                            )
                            logger.error(message)
                            _set_last_init_error(message)
                            if (
                                quarantine_path is None
                                and await aiofiles.os.path.exists(self._db_path)
                                and await is_zeroed_state_db(self._db_path)
                            ):
                                raise sqlite3.DatabaseError(message)
                    database = (
                        f"file:{os.path.abspath(self._db_path)}?mode=ro"
                        if self.read_only
                        else self._db_path
                    )
                    connection = await aiosqlite.connect(
                        database,
                        timeout=1.0,
                        isolation_level=None,
                        uri=self.read_only,
                    )
                    connection.row_factory = sqlite3.Row
                    if not self.read_only:
                        self._wal_active = (
                            await apply_wal_with_fallback(
                                connection,
                                db_label="state.db",
                            )
                            == "wal"
                        )
                    await apply_database_pragmas(
                        connection,
                        db_label="state.db",
                    )
                    await connection.execute("PRAGMA foreign_keys=ON")
                    await connection.execute("PRAGMA busy_timeout=1000")
                    if not self.read_only:
                        self._fts_cjk_loaded = await load_fts5_cjk_extension(
                            connection
                        )
                        await self._ensure_schema(connection)
                    else:
                        self._fts_enabled = (
                            await self._fts_table_probe(
                                connection,
                                "messages_fts",
                            )
                            is True
                        )
                        if self._fts_enabled:
                            self._trigram_available = (
                                await self._fts_table_probe(
                                    connection,
                                    "messages_fts_trigram",
                                )
                                is True
                            )
                    initialized = True
                    return connection
                finally:
                    if not initialized and connection is not None:
                        close_task = asyncio.create_task(connection.close())
                        try:
                            await asyncio.shield(close_task)
                        except asyncio.CancelledError:
                            await asyncio.shield(close_task)
                            raise

            deadline = time.monotonic() + self._WRITE_PATIENCE_S
            while True:
                database_error: Optional[sqlite3.DatabaseError] = None
                try:
                    connection = await _connect_and_initialize()
                except sqlite3.DatabaseError as caught:
                    database_error = caught
                except Exception as exc:
                    _set_last_init_error(f"{type(exc).__name__}: {exc}")
                    raise

                if database_error is None:
                    break
                if (
                    not self.read_only
                    and is_malformed_db_error(database_error)
                    and _claim_repair_attempt(self._db_path)
                ):
                    logger.error(
                        "state.db schema is malformed (%s) — attempting "
                        "automatic repair (a backup copy is made first).",
                        database_error,
                    )
                    try:
                        report = await repair_state_db_schema(self._db_path)
                    except Exception as exc:
                        _set_last_init_error(f"{type(exc).__name__}: {exc}")
                        raise
                    if report.get("repaired"):
                        self._schema_ready = False
                        continue
                message = str(database_error).lower()
                if (
                    isinstance(database_error, sqlite3.OperationalError)
                    and ("locked" in message or "busy" in message)
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(
                        min(
                            random.uniform(
                                self._WRITE_RETRY_MIN_S,
                                self._WRITE_RETRY_MAX_S,
                            ),
                            max(deadline - time.monotonic(), 0.001),
                        )
                    )
                    continue
                _set_last_init_error(
                    f"{type(database_error).__name__}: {database_error}"
                )
                raise database_error
            tracking_key = str(await _realpath(str(self._db_path)))
            _live_connection_counts[tracking_key] = (
                _live_connection_counts.get(tracking_key, 0) + 1
            )
            self._connection_tracking_key = tracking_key
            self._connection = connection
            return connection

    async def _ensure_schema(self, connection) -> None:
        """Create/reconcile the transcript tables without a sync DB hop.

        Column reconciliation and FTS initialization are deliberately
        performed through ``aiosqlite`` so a first turn against a fresh or
        older database never executes synchronous SQLite I/O.
        """
        if self._schema_ready:
            return
        await connection.executescript(SCHEMA_SQL)
        expected = await self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_columns in expected.items():
            cursor = await connection.execute(
                f'PRAGMA table_info("{table_name}")'
            )
            live_columns = {row[1] for row in await cursor.fetchall()}
            for column_name, column_type in declared_columns.items():
                if column_name in live_columns:
                    continue
                safe_name = column_name.replace('"', '""')
                await connection.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {column_type}'
                )
        await self._heal_session_model_usage_pk(connection)
        await self._migrate_inline_system_prompts(connection)
        await connection.executescript(DEFERRED_INDEX_SQL)
        await connection.execute(
            "UPDATE messages SET active = 1 WHERE active IS NULL"
        )
        if await self._sqlite_supports_fts5(connection):
            legacy_layout = await self._db_has_legacy_inline_fts(connection)
            triggers_need_repair = (
                await self._fts_trigger_count(connection) < len(_FTS_TRIGGERS)
            )
            if legacy_layout:
                self._fts_enabled = await self._ensure_fts_schema(
                    connection, "messages_fts", LEGACY_FTS_SQL
                )
                if self._fts_enabled:
                    self._trigram_available = await self._ensure_fts_schema(
                        connection,
                        "messages_fts_trigram",
                        LEGACY_FTS_TRIGRAM_SQL,
                    )
                    if triggers_need_repair:
                        await self._rebuild_legacy_fts_indexes(
                            connection,
                            include_trigram=self._trigram_available,
                        )
            else:
                self._fts_enabled = await self._ensure_fts_schema(
                    connection, "messages_fts", FTS_SQL
                )
                if self._fts_enabled:
                    self._trigram_available = await self._ensure_fts_schema(
                        connection, "messages_fts_trigram", FTS_TRIGRAM_SQL
                    )
                    await self._ensure_fts_cjk_schema(connection)
                    if triggers_need_repair:
                        await self._rebuild_fts_indexes(
                            connection,
                            include_trigram=self._trigram_available,
                        )
        else:
            await self._drop_fts_triggers(connection)
        version_row = await (
            await connection.execute("SELECT version FROM schema_version LIMIT 1")
        ).fetchone()
        if version_row is None:
            await connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        else:
            await connection.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )
        await connection.commit()
        self._schema_ready = True

    async def _sqlite_supports_fts5(self, connection) -> bool:
        try:
            await connection.execute(
                "CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)"
            )
            await connection.execute("DROP TABLE temp._hermes_fts5_probe")
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            self._warn_fts5_unavailable(exc)
            return False

    @staticmethod
    async def _fts_trigger_count(connection) -> int:
        placeholders = ",".join("?" for _ in _FTS_TRIGGERS)
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            _FTS_TRIGGERS,
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        return int(row[0])

    @staticmethod
    async def _db_has_legacy_inline_fts(connection) -> bool:
        cursor = await connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            return False
        return "tool_name" not in (row[0] or "")

    async def _fts_table_probe(self, connection, table_name: str) -> Optional[bool]:
        try:
            cursor = await connection.execute(
                f"SELECT * FROM {table_name} LIMIT 0"
            )
            await cursor.close()
            return True
        except sqlite3.OperationalError as exc:
            if self._is_fts5_unavailable_error(exc):
                if self._is_trigram_unavailable_error(exc):
                    self._warn_trigram_unavailable(exc)
                else:
                    self._warn_fts5_unavailable(exc)
                return None
            if "no such table" in str(exc).lower():
                return False
            raise

    async def _ensure_fts_schema(
        self,
        connection,
        table_name: str,
        ddl: str,
    ) -> bool:
        if await self._fts_table_probe(connection, table_name) is None:
            return False
        try:
            await connection.executescript(ddl)
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            if self._is_trigram_unavailable_error(exc):
                self._warn_trigram_unavailable(exc)
            else:
                self._warn_fts5_unavailable(exc)
            return False

    async def _ensure_fts_cjk_schema(self, connection) -> None:
        """Create, repair, or safely disable the CJK-bigram index."""
        row = await (
            await connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'messages_fts_cjk'"
            )
        ).fetchone()
        cjk_present = row is not None

        if not self._fts_cjk_loaded:
            if cjk_present:
                placeholders = ",".join("?" for _ in _FTS_CJK_TRIGGERS)
                cursor = await connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    f"AND name IN ({placeholders})",
                    _FTS_CJK_TRIGGERS,
                )
                try:
                    live = [item[0] for item in await cursor.fetchall()]
                finally:
                    await cursor.close()
                if live:
                    logger.warning(
                        "messages_fts_cjk triggers present but the "
                        "cjk_unicode61 tokenizer is unavailable (%s) — "
                        "dropping the cjk triggers so message writes keep "
                        "working. CJK search falls back to trigram/LIKE; "
                        "run `hermes sessions optimize-storage` on a host "
                        "with the extension to rebuild.",
                        fts5_cjk_so_path(),
                    )
                    await connection.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = '1'",
                        (FTS_CJK_STALE_KEY,),
                    )
                    for trigger in live:
                        await connection.execute(
                            f"DROP TRIGGER IF EXISTS {trigger}"
                        )
            self._fts_cjk_available = False
            return

        try:
            await connection.executescript(FTS_CJK_TABLE_SQL)
            if not cjk_present:
                await connection.execute(
                    "DELETE FROM state_meta WHERE key = ?",
                    (FTS_CJK_STALE_KEY,),
                )
                count_row = await (
                    await connection.execute(
                        "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
                    )
                ).fetchone()
                if count_row[0] > 0:
                    high_water_row = await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(id), 0) FROM messages"
                        )
                    ).fetchone()
                    for key, value in (
                        ("fts_cjk_rebuild_high_water", str(high_water_row[0])),
                        ("fts_cjk_rebuild_progress", "0"),
                    ):
                        await connection.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (key, value),
                        )
            stale = await (
                await connection.execute(
                    "SELECT 1 FROM state_meta WHERE key = ?",
                    (FTS_CJK_STALE_KEY,),
                )
            ).fetchone()
            if stale:
                self._fts_cjk_available = False
                return
            await connection.executescript(FTS_CJK_TRIGGER_SQL)
            pending = await (
                await connection.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
                )
            ).fetchone()
            self._fts_cjk_available = pending is None
        except sqlite3.OperationalError:
            logger.warning(
                "messages_fts_cjk ensure failed; CJK search stays on "
                "trigram/LIKE",
                exc_info=True,
            )
            self._fts_cjk_available = False

    @staticmethod
    async def _rebuild_fts_indexes(
        connection,
        *,
        include_trigram: bool = True,
    ) -> None:
        await connection.execute(
            "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
        )
        if include_trigram:
            await connection.execute(
                "INSERT INTO messages_fts_trigram(messages_fts_trigram) "
                "VALUES('rebuild')"
            )
        await connection.execute(
            "DELETE FROM state_meta WHERE key IN "
            "('fts_rebuild_high_water', 'fts_rebuild_progress')"
        )

    @staticmethod
    async def _rebuild_legacy_fts_indexes(
        connection,
        *,
        include_trigram: bool = True,
    ) -> None:
        await connection.execute("DELETE FROM messages_fts")
        await connection.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') "
            "FROM messages"
        )
        if include_trigram:
            await connection.execute("DELETE FROM messages_fts_trigram")
            await connection.execute(
                "INSERT INTO messages_fts_trigram(rowid, content) "
                "SELECT id, COALESCE(content, '') || ' ' || "
                "COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') "
                "FROM messages"
            )

    async def _drop_fts_triggers(self, connection) -> None:
        for trigger in _FTS_TRIGGERS:
            try:
                await connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.OperationalError:
                pass

    def _warn_trigram_unavailable(self, exc: sqlite3.OperationalError) -> None:
        if self._trigram_unavailable_warned:
            return
        self._trigram_unavailable_warned = True
        logger.info(
            "SQLite trigram tokenizer unavailable for %s; "
            "CJK/substring search will fall back to LIKE: %s",
            self.db_path,
            exc,
        )

    def _warn_fts5_unavailable(self, exc: sqlite3.OperationalError) -> None:
        self._fts_enabled = False
        if self._fts_unavailable_warned:
            return
        self._fts_unavailable_warned = True
        logger.warning(
            "SQLite FTS5 unavailable for %s; full-text session search "
            "disabled: %s",
            self.db_path,
            exc,
        )

    @staticmethod
    async def _store_system_prompt(connection, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = _system_prompt_hash(system_prompt)
        await connection.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    async def _delete_unreferenced_system_prompts(connection) -> None:
        await connection.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions "
            "WHERE sessions.system_prompt_hash = system_prompts.hash)"
        )

    async def _migrate_inline_system_prompts(self, connection) -> None:
        rows = await (
            await connection.execute(
                "SELECT id, system_prompt FROM sessions "
                "WHERE system_prompt IS NOT NULL AND system_prompt_hash IS NULL"
            )
        ).fetchall()
        for row in rows:
            prompt_hash = await self._store_system_prompt(
                connection, row["system_prompt"]
            )
            await connection.execute(
                "UPDATE sessions SET system_prompt = NULL, system_prompt_hash = ? "
                "WHERE id = ?",
                (prompt_hash, row["id"]),
            )
        if rows:
            await self._delete_unreferenced_system_prompts(connection)

    @staticmethod
    def _session_row_dict(row) -> Dict[str, Any]:
        data = dict(row)
        resolved = data.pop("_system_prompt_resolved", None)
        if "system_prompt" in data:
            data["system_prompt"] = resolved
        return data

    async def _heal_session_model_usage_pk(self, connection) -> None:
        """Rebuild legacy usage tables whose composite key omits ``task``."""
        rows = await (
            await connection.execute('PRAGMA table_info("session_model_usage")')
        ).fetchall()
        if not rows:
            return
        pk_columns = {row["name"] for row in rows if row["pk"]}
        if "task" in pk_columns:
            return

        logger.info(
            "session_model_usage has legacy primary key %r; rebuilding",
            sorted(pk_columns),
        )
        await connection.execute("PRAGMA foreign_keys=OFF")
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "ALTER TABLE session_model_usage "
                "RENAME TO session_model_usage_legacy_pk"
            )
            await connection.execute(
                """CREATE TABLE session_model_usage (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    billing_provider TEXT NOT NULL DEFAULT '',
                    billing_base_url TEXT NOT NULL DEFAULT '',
                    billing_mode TEXT NOT NULL DEFAULT '',
                    task TEXT NOT NULL DEFAULT '',
                    api_call_count INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    actual_cost_usd REAL NOT NULL DEFAULT 0,
                    cost_status TEXT,
                    cost_source TEXT,
                    first_seen REAL,
                    last_seen REAL,
                    PRIMARY KEY (
                        session_id, model, billing_provider,
                        billing_base_url, billing_mode, task
                    )
                )"""
            )
            await connection.execute(
                """INSERT OR IGNORE INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   )
                   SELECT session_id, model,
                          COALESCE(billing_provider, ''),
                          COALESCE(billing_base_url, ''),
                          COALESCE(billing_mode, ''), COALESCE(task, ''),
                          api_call_count, input_tokens, output_tokens,
                          cache_read_tokens, cache_write_tokens,
                          reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                          cost_status, cost_source, first_seen, last_seen
                   FROM session_model_usage_legacy_pk"""
            )
            await connection.execute("DROP TABLE session_model_usage_legacy_pk")
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
                "ON session_model_usage(session_id)"
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
                "ON session_model_usage(model)"
            )
            await connection.commit()
        except sqlite3.OperationalError as exc:
            await connection.rollback()
            logger.debug("session_model_usage PK heal skipped: %s", exc)
        finally:
            await connection.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _is_retryable_lock_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message

    async def _write(self, operation):
        """Run one short transaction, yielding between SQLite lock retries."""
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        compression_deadline: Optional[float] = None
        delay = self._WRITE_RETRY_MIN_S
        while True:
            try:
                async with self._get_write_lock():
                    connection = await self._get_connection()
                    await connection.execute("BEGIN IMMEDIATE")
                    try:
                        result = await operation(connection)
                        await connection.commit()
                        return result
                    except BaseException:
                        await connection.rollback()
                        raise
            except asyncio.CancelledError:
                raise
            except SessionCompressionInProgressError:
                if compression_deadline is None:
                    compression_deadline = min(
                        time.monotonic() + self._COMPRESSION_BUSY_WAIT_S,
                        deadline,
                    )
                remaining = compression_deadline - time.monotonic()
                if remaining <= 0:
                    raise
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, self._WRITE_RETRY_MAX_S)
            except Exception as exc:
                if (
                    isinstance(exc, sqlite3.DatabaseError)
                    # A new caller cancellation must supersede this stale DB error.
                    and await self._try_runtime_fts_rebuild(exc)  # noqa: ASYNC120
                ):
                    continue
                no_more_rows = (
                    isinstance(exc, sqlite3.Error)
                    and "no more rows available" in str(exc).lower()
                )
                if not no_more_rows and not self._is_retryable_lock_error(exc):
                    raise
                if time.monotonic() >= deadline:
                    if no_more_rows:
                        raise
                    raise sqlite3.OperationalError(
                        "database is locked by another Hermes process; "
                        "the database appears healthy but remained busy past "
                        f"the {self._WRITE_PATIENCE_S:g}s write deadline"
                    ) from exc
                await asyncio.sleep(random.uniform(delay / 2, delay))
                delay = min(delay * 2, self._WRITE_RETRY_MAX_S)

    async def _check_transcript_write_guards(
        self,
        connection,
        session_id: str,
        compression_lock_holder: Optional[str],
    ) -> None:
        """Apply transcript admission guards inside the active transaction."""
        active_lock = await (
            await connection.execute(
                "SELECT holder FROM compression_locks "
                "WHERE session_id = ? AND expires_at > ?",
                (session_id, time.time()),
            )
        ).fetchone()
        if (
            active_lock is not None
            and active_lock["holder"] != compression_lock_holder
        ):
            raise SessionCompressionInProgressError(
                f"Session {session_id!r} is being compressed by another writer"
            )
        session = await (
            await connection.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        if (
            session is not None
            and session["ended_at"] is not None
            and session["end_reason"] == "compression"
        ):
            raise CompressionSessionClosedError(session_id)

    async def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create or enrich the turn's session row without blocking the loop."""
        model_config = kwargs.get("model_config")
        parent_session_id = kwargs.get("parent_session_id")

        async def _create(connection):
            system_prompt_hash = await self._store_system_prompt(
                connection, kwargs.get("system_prompt")
            )
            await connection.execute(
                """INSERT INTO sessions (
                   id, source, user_id, session_key, chat_id, chat_type, thread_id,
                   model, model_config, system_prompt, system_prompt_hash,
                   parent_session_id, cwd,
                   profile_name, git_repo_root, started_at
                )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = COALESCE(sessions.model_config, excluded.model_config),
                       system_prompt_hash = COALESCE(
                           sessions.system_prompt_hash,
                           excluded.system_prompt_hash
                       ),
                       system_prompt = CASE
                           WHEN sessions.system_prompt_hash IS NULL
                                AND excluded.system_prompt_hash IS NOT NULL
                           THEN NULL
                           ELSE sessions.system_prompt
                       END,
                       session_key = COALESCE(sessions.session_key, excluded.session_key),
                       chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                       chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                       thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                       parent_session_id = COALESCE(sessions.parent_session_id, excluded.parent_session_id),
                       cwd = COALESCE(sessions.cwd, excluded.cwd),
                       profile_name = COALESCE(sessions.profile_name, excluded.profile_name),
                       git_repo_root = COALESCE(sessions.git_repo_root, excluded.git_repo_root)""",
                (
                    session_id,
                    source,
                    kwargs.get("user_id"),
                    kwargs.get("session_key"),
                    kwargs.get("chat_id"),
                    kwargs.get("chat_type"),
                    kwargs.get("thread_id"),
                    kwargs.get("model"),
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    kwargs.get("cwd"),
                    kwargs.get("profile_name"),
                    kwargs.get("git_repo_root"),
                    time.time(),
                ),
            )
            if system_prompt_hash is not None:
                await self._delete_unreferenced_system_prompts(connection)
            if parent_session_id:
                await connection.execute(
                    """UPDATE sessions
                       SET cwd = COALESCE(sessions.cwd,
                                 (SELECT p.cwd FROM sessions p
                                   WHERE p.id = sessions.parent_session_id)),
                           git_repo_root = COALESCE(sessions.git_repo_root,
                                           (SELECT p.git_repo_root FROM sessions p
                                             WHERE p.id = sessions.parent_session_id)),
                           git_branch = COALESCE(sessions.git_branch,
                                        (SELECT p.git_branch FROM sessions p
                                          WHERE p.id = sessions.parent_session_id)),
                           profile_name = COALESCE(sessions.profile_name,
                                          (SELECT p.profile_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL""",
                    (session_id,),
                )
                await connection.execute(
                    """UPDATE sessions
                       SET user_id = COALESCE(sessions.user_id,
                                     (SELECT p.user_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           session_key = COALESCE(sessions.session_key,
                                         (SELECT p.session_key FROM sessions p
                                           WHERE p.id = sessions.parent_session_id)),
                           chat_id = COALESCE(sessions.chat_id,
                                     (SELECT p.chat_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           chat_type = COALESCE(sessions.chat_type,
                                       (SELECT p.chat_type FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           thread_id = COALESCE(sessions.thread_id,
                                       (SELECT p.thread_id FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           display_name = COALESCE(sessions.display_name,
                                          (SELECT p.display_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id)),
                           origin_json = COALESCE(sessions.origin_json,
                                         (SELECT p.origin_json FROM sessions p
                                           WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL
                       AND EXISTS (
                           SELECT 1 FROM sessions p
                           WHERE p.id = sessions.parent_session_id
                             AND p.end_reason = 'compression'
                       )""",
                    (session_id,),
                )

        await self._write(_create)
        return session_id

    async def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
        **kwargs,
    ) -> str:
        """Ensure a session row exists and preserve the upstream return value."""
        return await self.create_session(
            session_id,
            source,
            model=model,
            **kwargs,
        )

    async def update_session_cwd(
        self,
        session_id: str,
        cwd: str,
        git_branch: str = None,
        git_repo_root: str = None,
    ) -> None:
        """Persist workspace identity without blocking the event loop."""
        if not session_id or not cwd:
            return

        assignments = ["cwd = ?"]
        parameters: list[Any] = [cwd]
        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()
        if branch:
            assignments.append("git_branch = ?")
            parameters.append(branch)
        if repo_root:
            assignments.append("git_repo_root = ?")
            parameters.append(repo_root)
        parameters.append(session_id)

        async def _update(connection):
            await connection.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )

        await self._write(_update)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
        effect_disposition: Optional[str] = None,
        timestamp: Any = None,
        api_content: Optional[str] = None,
        display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None,
        compression_lock_holder: Optional[str] = None,
    ) -> int:
        """Append one transcript row using the same wire serialization as SessionDB."""
        display_metadata_json = self._encode_display_metadata(display_metadata)
        reasoning_details_json = json.dumps(reasoning_details) if reasoning_details else None
        codex_items_json = json.dumps(codex_reasoning_items) if codex_reasoning_items else None
        codex_message_items_json = (
            json.dumps(codex_message_items) if codex_message_items else None
        )
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                tool_calls = []
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        stored_content = self._encode_content(content)
        message_timestamp = time.time()
        if timestamp is not None:
            try:
                message_timestamp = float(
                    timestamp.timestamp() if hasattr(timestamp, "timestamp") else timestamp
                )
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid explicit message timestamp: %r", timestamp)
        num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else int(tool_calls is not None)

        async def _append(connection):
            await self._check_transcript_write_guards(
                connection, session_id, compression_lock_holder
            )
            cursor = await connection.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, role, stored_content, tool_call_id, tool_calls_json,
                    _scrub_surrogates(tool_name), effect_disposition, message_timestamp,
                    token_count, finish_reason, _scrub_surrogates(reasoning),
                    _scrub_surrogates(reasoning_content), reasoning_details_json,
                    codex_items_json, codex_message_items_json, platform_message_id,
                    1 if observed else 0, 1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(display_kind) if isinstance(display_kind, str) else None,
                    display_metadata_json,
                ),
            )
            if num_tool_calls:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + 1, "
                    "tool_call_count = tool_call_count + ? WHERE id = ?",
                    (num_tool_calls, session_id),
                )
            else:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return cursor.lastrowid

        return await self._write(_append)

    async def _insert_message_rows(
        self,
        connection,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, int]:
        """Insert transcript rows in the caller's active transaction."""
        now_ts = time.time()
        inserted = 0
        tool_calls_total = 0
        for message in messages:
            role = message.get("role", "unknown")
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []
            timestamp = message.get("timestamp", now_ts)
            try:
                timestamp = float(
                    timestamp.timestamp()
                    if hasattr(timestamp, "timestamp")
                    else timestamp
                )
            except (TypeError, ValueError):
                logger.debug(
                    "Ignoring invalid explicit message timestamp: %r",
                    message.get("timestamp"),
                )
                timestamp = now_ts
            reasoning_details = (
                message.get("reasoning_details") if role == "assistant" else None
            )
            codex_reasoning_items = (
                message.get("codex_reasoning_items") if role == "assistant" else None
            )
            codex_message_items = (
                message.get("codex_message_items") if role == "assistant" else None
            )
            api_content = message.get("api_content")
            display_kind = message.get("display_kind")
            await connection.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    self._encode_content(message.get("content")),
                    message.get("tool_call_id"),
                    json.dumps(tool_calls) if tool_calls else None,
                    _scrub_surrogates(message.get("tool_name")),
                    message.get("effect_disposition"),
                    timestamp,
                    message.get("token_count"),
                    message.get("finish_reason"),
                    _scrub_surrogates(message.get("reasoning"))
                    if role == "assistant"
                    else None,
                    _scrub_surrogates(message.get("reasoning_content"))
                    if role == "assistant"
                    else None,
                    json.dumps(reasoning_details) if reasoning_details else None,
                    json.dumps(codex_reasoning_items) if codex_reasoning_items else None,
                    json.dumps(codex_message_items) if codex_message_items else None,
                    message.get("platform_message_id") or message.get("message_id"),
                    1 if message.get("observed") else 0,
                    1,
                    _scrub_surrogates(api_content)
                    if isinstance(api_content, str)
                    else None,
                    _scrub_surrogates(display_kind)
                    if isinstance(display_kind, str)
                    else None,
                    self._encode_display_metadata(message.get("display_metadata")),
                ),
            )
            inserted += 1
            if tool_calls is not None:
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else 1
                )
            now_ts = max(now_ts + 1e-6, timestamp + 1e-6)
        return inserted, tool_calls_total

    async def append_messages_batch(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        compression_lock_holder: Optional[str] = None,
        chunk_rows: Optional[int] = None,
    ) -> int:
        """Append one turn atomically, optionally chunking large copy jobs."""
        if not messages:
            return 0
        if chunk_rows is not None:
            if chunk_rows <= 0:
                raise ValueError("chunk_rows must be positive")
            if len(messages) > chunk_rows:
                inserted_total = 0
                for start in range(0, len(messages), chunk_rows):
                    inserted_total += await self.append_messages_batch(
                        session_id,
                        messages[start : start + chunk_rows],
                        compression_lock_holder=compression_lock_holder,
                    )
                return inserted_total

        async def _append_batch(connection):
            await self._check_transcript_write_guards(
                connection, session_id, compression_lock_holder
            )
            inserted, tool_calls_total = await self._insert_message_rows(
                connection, session_id, messages
            )
            if tool_calls_total:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + ?, "
                    "tool_call_count = tool_call_count + ? WHERE id = ?",
                    (inserted, tool_calls_total, session_id),
                )
            else:
                await connection.execute(
                    "UPDATE sessions SET message_count = message_count + ? WHERE id = ?",
                    (inserted, session_id),
                )
            return inserted

        return await self._write(_append_batch)

    async def replace_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        active_only: bool = False,
    ) -> None:
        """Atomically replace all or only active transcript rows."""
        active_clause = " AND active = 1" if active_only else ""

        async def _replace(connection):
            session = await (
                await connection.execute(
                    "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                    (session_id,),
                )
            ).fetchone()
            if (
                session is not None
                and session["ended_at"] is not None
                and session["end_reason"] == "compression"
            ):
                raise CompressionSessionClosedError(session_id)
            await connection.execute(
                f"DELETE FROM messages WHERE session_id = ?{active_clause}",
                (session_id,),
            )
            await connection.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 "
                "WHERE id = ?",
                (session_id,),
            )
            total_messages, total_tool_calls = await self._insert_message_rows(
                connection,
                session_id,
                messages,
            )
            await connection.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? "
                "WHERE id = ?",
                (total_messages, total_tool_calls, session_id),
            )

        await self._write(_replace)

    async def has_archived_messages(self, session_id: str) -> bool:
        """Return True if the session has any soft-archived (``active = 0``) rows.

        Used by callers (e.g. the ACP adapter's ``_persist``) that must decide
        whether a full-history :meth:`replace_messages` would destroy durable
        compaction-archived turns. Cheap existence probe — does not load rows.
        """
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT 1 FROM messages "
                "WHERE session_id = ? AND active = 0 LIMIT 1",
                (session_id,),
            )
        ).fetchone()
        return row is not None

    async def end_session(self, session_id: str, end_reason: str) -> None:
        async def _end(connection):
            await connection.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )

        await self._write(_end)

    async def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        async def _reopen(connection):
            await connection.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )

        await self._write(_reopen)

    async def try_acquire_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Atomically acquire an expiring compression lease."""
        if not session_id or not holder:
            return False
        now = time.time()

        async def _acquire(connection):
            row = await (
                await connection.execute(
                    "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                    (session_id,),
                )
            ).fetchone()
            if row is not None and (
                float(row["expires_at"]) < now
                or _compression_lock_holder_process_is_dead(row["holder"])
            ):
                await connection.execute(
                    "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?",
                    (session_id, row["holder"]),
                )
            await connection.execute(
                "INSERT OR IGNORE INTO compression_locks "
                "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                (session_id, holder, now, now + ttl_seconds),
            )
            owner = await (
                await connection.execute(
                    "SELECT holder FROM compression_locks WHERE session_id = ?",
                    (session_id,),
                )
            ).fetchone()
            return owner is not None and owner["holder"] == holder

        try:
            return bool(await self._write(_acquire))
        except sqlite3.Error as exc:
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False

    async def refresh_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Extend a lease only while its owner still matches."""
        if not session_id or not holder:
            return False

        async def _refresh(connection):
            cursor = await connection.execute(
                "UPDATE compression_locks SET expires_at = ? "
                "WHERE session_id = ? AND holder = ?",
                (time.time() + ttl_seconds, session_id, holder),
            )
            return cursor.rowcount > 0

        try:
            return bool(await self._write(_refresh))
        except sqlite3.Error as exc:
            logger.warning("refresh_compression_lock(%s) failed: %s", session_id, exc)
            return False

    async def release_compression_lock(self, session_id: str, holder: str) -> None:
        """Release a compression lease iff it is still owned by *holder*."""
        if not session_id or not holder:
            return

        async def _release(connection):
            await connection.execute(
                "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?",
                (session_id, holder),
            )

        try:
            await self._write(_release)
        except sqlite3.Error as exc:
            logger.warning("release_compression_lock(%s) failed: %s", session_id, exc)

    async def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Return the live compression-lease owner, if any."""
        if not session_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT holder FROM compression_locks "
                "WHERE session_id = ? AND expires_at >= ?",
                (session_id, time.time()),
            )
        ).fetchone()
        return row["holder"] if row is not None else None

    async def archive_and_compact(
        self,
        session_id: str,
        compacted_messages: List[Dict[str, Any]],
    ) -> int:
        """Atomically archive live rows and insert the compacted transcript."""
        async def _archive(connection):
            await connection.execute(
                "UPDATE messages SET active = 0, compacted = 1 "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            )
            now_ts = time.time()
            tool_calls_total = 0
            for message in compacted_messages:
                role = message.get("role", "unknown")
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, str):
                    try:
                        tool_calls = json.loads(tool_calls)
                    except (json.JSONDecodeError, TypeError):
                        tool_calls = []
                timestamp = message.get("timestamp", now_ts)
                try:
                    timestamp = float(
                        timestamp.timestamp()
                        if hasattr(timestamp, "timestamp")
                        else timestamp
                    )
                except (TypeError, ValueError):
                    timestamp = now_ts
                reasoning_details = (
                    message.get("reasoning_details") if role == "assistant" else None
                )
                codex_reasoning_items = (
                    message.get("codex_reasoning_items") if role == "assistant" else None
                )
                codex_message_items = (
                    message.get("codex_message_items") if role == "assistant" else None
                )
                await connection.execute(
                    """INSERT INTO messages (session_id, role, content, tool_call_id,
                       tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        role,
                        self._encode_content(message.get("content")),
                        message.get("tool_call_id"),
                        json.dumps(tool_calls) if tool_calls else None,
                        _scrub_surrogates(message.get("tool_name")),
                        message.get("effect_disposition"),
                        timestamp,
                        message.get("token_count"),
                        message.get("finish_reason"),
                        _scrub_surrogates(message.get("reasoning"))
                        if role == "assistant"
                        else None,
                        _scrub_surrogates(message.get("reasoning_content"))
                        if role == "assistant"
                        else None,
                        json.dumps(reasoning_details) if reasoning_details else None,
                        json.dumps(codex_reasoning_items) if codex_reasoning_items else None,
                        json.dumps(codex_message_items) if codex_message_items else None,
                        message.get("platform_message_id") or message.get("message_id"),
                        1 if message.get("observed") else 0,
                        1,
                        _scrub_surrogates(message.get("api_content"))
                        if isinstance(message.get("api_content"), str)
                        else None,
                        _scrub_surrogates(message.get("display_kind"))
                        if isinstance(message.get("display_kind"), str)
                        else None,
                        self._encode_display_metadata(message.get("display_metadata")),
                    ),
                )
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else int(tool_calls is not None)
                )
                now_ts = max(now_ts + 1e-6, timestamp + 1e-6)
            await connection.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (len(compacted_messages), tool_calls_total, session_id),
            )
            return len(compacted_messages)

        return await self._write(_archive)

    async def update_system_prompt(
        self, session_id: str, system_prompt: Optional[str]
    ) -> None:
        """Persist the assembled prompt without a synchronous SQLite call."""
        async def _update(connection):
            system_prompt_hash = await self._store_system_prompt(
                connection, system_prompt
            )
            await connection.execute(
                "UPDATE sessions SET system_prompt = NULL, "
                "system_prompt_hash = ? WHERE id = ?",
                (system_prompt_hash, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update)

    async def update_session_runtime_lock(
        self,
        session_id: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        route_source: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """Merge a runtime lock while preserving unrelated model metadata."""
        lock = {
            "provider": provider or "",
            "model": model or "",
            "model_options": model_options or {},
            "route_source": route_source or "",
            "confirmed": bool(confirmed),
            "updated_at": time.time(),
        }

        async def _update(connection):
            row = await (
                await connection.execute(
                    "SELECT model_config FROM sessions WHERE id = ?", (session_id,)
                )
            ).fetchone()
            if row is None:
                return
            config: Dict[str, Any] = {}
            raw = row["model_config"]
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        config = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            config["browser_model_lock"] = lock
            await connection.execute(
                "UPDATE sessions SET model_config = ?, model = COALESCE(?, model), "
                "system_prompt = NULL, system_prompt_hash = NULL WHERE id = ?",
                (json.dumps(config), model, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update)

    async def update_session_meta(
        self,
        session_id: str,
        model_config_json: str,
        model: Optional[str] = None,
    ) -> None:
        """Update model metadata while preserving a stored model when omitted."""
        await self.flush_token_counts()

        async def _update(connection):
            await connection.execute(
                "UPDATE sessions SET model_config = ?, "
                "model = COALESCE(?, model) WHERE id = ?",
                (model_config_json, model, session_id),
            )

        await self._write(_update)

    async def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        """Persist resolved git repo roots for cwds that don't have one yet.

        Backfills history so projects light up for sessions created before the
        column existed, without clobbering an already-recorded root. Only
        non-empty roots are written (a non-git cwd stays NULL).
        """
        pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
        if not pairs:
            return

        async def _backfill(connection):
            for root, cwd in pairs:
                await connection.execute(
                    "UPDATE sessions SET git_repo_root = ? "
                    "WHERE cwd = ? AND COALESCE(git_repo_root, '') = ''",
                    (root, cwd),
                )

        await self._write(_backfill)

    async def update_session_model(self, session_id: str, model: str) -> None:
        """Persist a mid-session model switch and clear its stale runtime lock."""
        await self.flush_token_counts()

        async def _update(connection):
            await connection.execute(
                """UPDATE sessions SET
                   model = ?,
                   model_config = CASE
                       WHEN model_config IS NULL THEN NULL
                       WHEN json_valid(model_config)
                           THEN json_remove(model_config, '$.browser_model_lock')
                       ELSE model_config
                   END,
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (model, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update)

    async def set_latest_user_api_content(
        self, session_id: str, content: Any, api_content: str
    ) -> int:
        """Backfill the API sidecar on the newest matching active user row."""
        encoded = self._encode_content(content)

        async def _update(connection):
            cursor = await connection.execute(
                "UPDATE messages SET api_content = ? WHERE id = ("
                "SELECT id FROM messages "
                "WHERE session_id = ? AND role = 'user' AND active = 1 "
                "ORDER BY id DESC LIMIT 1"
                ") AND content IS ?",
                (_scrub_surrogates(api_content), session_id, encoded),
            )
            return cursor.rowcount

        return await self._write(_update)

    async def record_auxiliary_usage(
        self,
        session_id: str,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        """Persist one auxiliary-model usage delta without sync SQLite I/O."""
        if not session_id or not task:
            return
        await self.create_session(session_id, "unknown")
        now = time.time()

        async def _record(connection):
            await connection.execute(
                """INSERT INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   ) VALUES (?, ?, ?, ?, '', ?, 1, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)
                   ON CONFLICT(session_id, model, billing_provider, billing_base_url,
                               billing_mode, task)
                   DO UPDATE SET
                       api_call_count = api_call_count + 1,
                       input_tokens = input_tokens + excluded.input_tokens,
                       output_tokens = output_tokens + excluded.output_tokens,
                       cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                       cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                       reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                       estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                       last_seen = excluded.last_seen""",
                (
                    session_id,
                    model or "unknown",
                    billing_provider or "",
                    billing_base_url or "",
                    task,
                    input_tokens or 0,
                    output_tokens or 0,
                    cache_read_tokens or 0,
                    cache_write_tokens or 0,
                    reasoning_tokens or 0,
                    float(estimated_cost_usd or 0.0),
                    now,
                    now,
                ),
            )

        await self._write(_record)

    async def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Persist one token-accounting update through the async connection."""
        await self.create_session(session_id, "unknown", model=model)
        has_accounted_usage = bool(
            input_tokens or output_tokens or cache_read_tokens
            or cache_write_tokens or reasoning_tokens or api_call_count
            or estimated_cost_usd or actual_cost_usd
        )
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?, output_tokens = ?,
                   cache_read_tokens = ?, cache_write_tokens = ?,
                   reasoning_tokens = ?, estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE WHEN ? IS NULL THEN actual_cost_usd ELSE ? END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?), api_call_count = ?
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?, output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = COALESCE(api_call_count, 0) + ?
                   WHERE id = ?"""
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None,
            model if has_accounted_usage else None,
            api_call_count,
            session_id,
        )

        async def _update(connection):
            await connection.execute(sql, params)
            if absolute or not has_accounted_usage:
                return
            await connection.execute(
                """INSERT INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, model, billing_provider, billing_base_url,
                               billing_mode, task)
                   DO UPDATE SET
                       api_call_count = api_call_count + excluded.api_call_count,
                       input_tokens = input_tokens + excluded.input_tokens,
                       output_tokens = output_tokens + excluded.output_tokens,
                       cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                       cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                       reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                       estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                       actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                       cost_status = COALESCE(excluded.cost_status, cost_status),
                       cost_source = COALESCE(excluded.cost_source, cost_source),
                       last_seen = excluded.last_seen""",
                (
                    session_id,
                    model or "unknown",
                    billing_provider or "",
                    billing_base_url or "",
                    billing_mode or "",
                    api_call_count or 0,
                    input_tokens or 0,
                    output_tokens or 0,
                    cache_read_tokens or 0,
                    cache_write_tokens or 0,
                    reasoning_tokens or 0,
                    float(estimated_cost_usd or 0.0),
                    float(actual_cost_usd or 0.0),
                    cost_status,
                    cost_source,
                    time.time(),
                    time.time(),
                ),
            )

        await self._write(_update)

    async def queue_token_counts(self, session_id: str, **kwargs) -> None:
        """Persist a token/cost update through the native async writer.

        Accepts the same keyword arguments as :meth:`update_token_counts`.
        The synchronous implementation needed a background thread to move
        SQLite work off the turn thread; the async implementation awaits the
        non-blocking database operation directly and preserves enqueue order.
        """
        await self.update_token_counts(session_id, **kwargs)

    async def update_session_billing_route(
        self,
        session_id: str,
        *,
        provider: str,
        base_url: str,
        billing_mode: Optional[str] = None,
    ) -> None:
        """Persist a model-route change through the native async connection."""
        async def _update_route(connection):
            await connection.execute(
                """UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (provider, base_url, billing_mode, session_id),
            )
            await self._delete_unreferenced_system_prompts(connection)

        await self._write(_update_route)

    async def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the durable ``state_meta`` store."""
        connection = await self._get_connection()
        cursor = await connection.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row is not None else None

    async def set_meta(self, key: str, value: str, *, cursor: Any = None) -> None:
        """Atomically upsert a value in the durable ``state_meta`` store."""

        if cursor is not None:
            await cursor.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            return

        async def _set(connection):
            await connection.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

        await self._write(_set)

    async def delete_meta(self, key: str) -> bool:
        """Delete one metadata key and report whether a row was removed."""

        async def _delete(connection):
            cursor = await connection.execute(
                "DELETE FROM state_meta WHERE key = ?", (key,)
            )
            return cursor.rowcount > 0

        return await self._write(_delete)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return one session row through the adapter's async connection."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) "
                "AS _system_prompt_resolved "
                "FROM sessions s LEFT JOIN system_prompts sp "
                "ON sp.hash = s.system_prompt_hash WHERE s.id = ?",
                (session_id,),
            )
        ).fetchone()
        return self._session_row_dict(row) if row is not None else None

    async def resolve_session_id(
        self, session_id_or_prefix: str
    ) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session id."""
        exact = await self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]
        escaped = (
            session_id_or_prefix.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' "
                "ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
        ).fetchall()
        return rows[0]["id"] if len(rows) == 1 else None

    async def session_count_ge(self, n: int = 1) -> bool:
        """Return whether at least *n* sessions exist, including archived rows."""
        connection = await self._get_connection()
        rows = await (
            await connection.execute("SELECT 1 FROM sessions LIMIT ?", (n,))
        ).fetchall()
        return len(rows) >= n

    async def count_empty_sessions(self) -> int:
        """Count ended, non-archived sessions with no transcript rows."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT COUNT(*) AS count FROM sessions "
                "WHERE message_count = 0 "
                "AND ended_at IS NOT NULL "
                "AND archived = 0"
            )
        ).fetchone()
        return int(row["count"] if row is not None else 0)

    @staticmethod
    async def _remove_session_files(
        sessions_dir: Optional[Path], session_id: str
    ) -> None:
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            try:
                await aiofiles.os.remove(sessions_dir / f"{session_id}{suffix}")
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            names = await aiofiles.os.listdir(sessions_dir)
        except OSError:
            return
        prefix = f"request_dump_{session_id}_"
        for name in names:
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            try:
                await aiofiles.os.remove(sessions_dir / name)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    async def get_session_delete_targets(self, session_id: str) -> List[str]:
        connection = await self._get_connection()
        exists = await (
            await connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            )
        ).fetchone()
        if exists is None:
            return []
        delegate_ids = await _collect_delegate_child_ids(connection, [session_id])
        return [session_id, *sorted(delegate_ids)]

    async def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        removed_delegate_ids: List[str] = []
        expected_ids = (
            set(expected_delete_ids) if expected_delete_ids is not None else None
        )

        async def _delete(connection):
            exists = await (
                await connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
                )
            ).fetchone()
            if exists is None:
                return False
            if expected_ids is not None:
                actual_ids = {
                    session_id,
                    *(await _collect_delegate_child_ids(connection, [session_id])),
                }
                if actual_ids != expected_ids:
                    return False
            removed_delegate_ids.extend(
                await _delete_delegate_children(connection, [session_id])
            )
            await connection.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            await connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            await connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await self._delete_unreferenced_system_prompts(connection)
            return True

        deleted = bool(await self._write(_delete))
        if deleted:
            for delegate_id in removed_delegate_ids:
                await self._remove_session_files(sessions_dir, delegate_id)
            await self._remove_session_files(sessions_dir, session_id)
        return deleted

    async def delete_session_if_empty(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        async def _delete(connection):
            cursor = await connection.execute(
                """DELETE FROM sessions
                   WHERE id = ?
                     AND title IS NULL
                     AND NOT EXISTS (
                         SELECT 1 FROM messages
                         WHERE messages.session_id = sessions.id
                     )
                     AND NOT EXISTS (
                         SELECT 1 FROM sessions child
                         WHERE child.parent_session_id = sessions.id
                     )""",
                (session_id,),
            )
            if cursor.rowcount > 0:
                await self._delete_unreferenced_system_prompts(connection)
            return cursor.rowcount > 0

        deleted = bool(await self._write(_delete))
        if deleted:
            await self._remove_session_files(sessions_dir, session_id)
        return deleted

    async def delete_sessions(
        self,
        session_ids: List[str],
        sessions_dir: Optional[Path] = None,
    ) -> int:
        ids = list(
            {
                session_id
                for session_id in session_ids
                if isinstance(session_id, str) and session_id
            }
        )
        if not ids:
            return 0
        removed_ids: List[str] = []
        removed_delegate_ids: List[str] = []

        async def _delete(connection):
            placeholders = ",".join("?" for _ in ids)
            rows = await (
                await connection.execute(
                    f"SELECT id FROM sessions WHERE id IN ({placeholders})", ids
                )
            ).fetchall()
            existing = [row["id"] for row in rows]
            if not existing:
                return 0
            existing_placeholders = ",".join("?" for _ in existing)
            removed_delegate_ids.extend(
                await _delete_delegate_children(connection, existing)
            )
            await connection.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({existing_placeholders})",
                existing,
            )
            await connection.execute(
                f"DELETE FROM messages "
                f"WHERE session_id IN ({existing_placeholders})",
                existing,
            )
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({existing_placeholders})",
                existing,
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(existing)
            return len(existing)

        count = int(await self._write(_delete))
        for delegate_id in removed_delegate_ids:
            await self._remove_session_files(sessions_dir, delegate_id)
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    async def delete_empty_sessions(
        self, sessions_dir: Optional[Path] = None
    ) -> int:
        removed_ids: List[str] = []

        async def _delete(connection):
            rows = await (
                await connection.execute(
                    "SELECT id FROM sessions WHERE message_count = 0 "
                    "AND ended_at IS NOT NULL AND archived = 0"
                )
            ).fetchall()
            session_ids = [row["id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            await connection.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})",
                session_ids,
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(session_ids)
            return len(session_ids)

        count = int(await self._write(_delete))
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    @staticmethod
    def _prune_filter_where(
        *,
        last_active_before: Optional[float] = None,
        last_active_after: Optional[float] = None,
        started_before: Optional[float] = None,
        started_after: Optional[float] = None,
        source: Optional[str] = None,
        title_like: Optional[str] = None,
        end_reason: Optional[str] = None,
        cwd_prefix: Optional[str] = None,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None,
        archived: Optional[bool] = None,
        model_like: Optional[str] = None,
        provider: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        branch_like: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_tool_calls: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
    ) -> Tuple[str, list]:
        clauses = ["s.ended_at IS NOT NULL"]
        params: list[Any] = []
        if last_active_before is not None:
            clauses.append(
                "COALESCE((SELECT MAX(m.timestamp) FROM messages m "
                "WHERE m.session_id = s.id), s.started_at) < ?"
            )
            params.append(last_active_before)
        if last_active_after is not None:
            clauses.append(
                "COALESCE((SELECT MAX(m.timestamp) FROM messages m "
                "WHERE m.session_id = s.id), s.started_at) >= ?"
            )
            params.append(last_active_after)
        if started_before is not None:
            clauses.append("s.started_at < ?")
            params.append(started_before)
        if started_after is not None:
            clauses.append("s.started_at >= ?")
            params.append(started_after)
        if source:
            clauses.append("s.source = ?")
            params.append(source)
        if title_like:
            clauses.append("LOWER(COALESCE(s.title, '')) LIKE ?")
            params.append(f"%{title_like.lower()}%")
        if end_reason:
            clauses.append("s.end_reason = ?")
            params.append(end_reason)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            clauses.append(clause)
            params.extend(clause_params)
        if min_messages is not None:
            clauses.append("s.message_count >= ?")
            params.append(min_messages)
        if max_messages is not None:
            clauses.append("s.message_count <= ?")
            params.append(max_messages)
        if model_like:
            clauses.append("LOWER(COALESCE(s.model, '')) LIKE ?")
            params.append(f"%{model_like.lower()}%")
        if provider:
            clauses.append("LOWER(COALESCE(s.billing_provider, '')) = ?")
            params.append(provider.lower())
        if user_id:
            clauses.append("s.user_id = ?")
            params.append(user_id)
        if chat_id:
            clauses.append("s.chat_id = ?")
            params.append(chat_id)
        if chat_type:
            clauses.append("s.chat_type = ?")
            params.append(chat_type)
        if branch_like:
            clauses.append("LOWER(COALESCE(s.git_branch, '')) LIKE ?")
            params.append(f"%{branch_like.lower()}%")
        if min_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + "
                "COALESCE(s.output_tokens, 0)) >= ?"
            )
            params.append(min_tokens)
        if max_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + "
                "COALESCE(s.output_tokens, 0)) <= ?"
            )
            params.append(max_tokens)
        if min_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) >= ?"
            )
            params.append(min_cost)
        if max_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) <= ?"
            )
            params.append(max_cost)
        if min_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) >= ?")
            params.append(min_tool_calls)
        if max_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) <= ?")
            params.append(max_tool_calls)
        if archived is True:
            clauses.append("s.archived = 1")
        elif archived is False:
            clauses.append("s.archived = 0")
        return " AND ".join(clauses), params

    async def list_prune_candidates(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> List[Dict[str, Any]]:
        """Return sessions a matching prune/archive call would touch."""
        if (
            filters.get("last_active_before") is None
            and filters.get("started_before") is None
            and older_than_days is not None
        ):
            filters["last_active_before"] = time.time() - (
                older_than_days * 86_400
            )
        where, params = self._prune_filter_where(source=source, **filters)
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                f"""SELECT s.id, s.source, s.title, s.model, s.started_at,
                           COALESCE(
                               (SELECT MAX(m.timestamp) FROM messages m
                                WHERE m.session_id = s.id),
                               s.started_at
                           ) AS last_active,
                           s.ended_at, s.message_count, s.archived
                    FROM sessions s WHERE {where}
                    ORDER BY last_active ASC, s.started_at ASC""",
                params,
            )
        ).fetchall()
        return [dict(row) for row in rows]

    async def archive_sessions(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> int:
        """Bulk-archive every session matching the prune filter surface."""
        filters.setdefault("archived", False)
        rows = await self.list_prune_candidates(
            older_than_days=older_than_days, source=source, **filters
        )
        for row in rows:
            await self.set_session_archived(row["id"], True)
        return len(rows)

    async def archive_stale_sessions(
        self, idle_days: float, *, exclude_pinned: bool = True
    ) -> int:
        """Archive sessions untouched for at least ``idle_days`` days."""
        if idle_days is None or idle_days < 0:
            return 0
        cutoff = time.time() - float(idle_days) * 86_400.0
        pin_clause = "AND s.pinned = 0" if exclude_pinned else ""
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                f"""
                SELECT s.id FROM sessions s
                WHERE s.archived = 0
                  AND COALESCE(s.end_reason, '') <> 'compression'
                  {pin_clause}
                  AND {_sql_session_last_active("s")} < ?
                ORDER BY s.started_at ASC
                """,
                (cutoff,),
            )
        ).fetchall()
        ids = [row["id"] for row in rows]
        for session_id in ids:
            await self.set_session_archived(session_id, True)
        return len(ids)

    async def prune_sessions(
        self,
        older_than_days: Optional[float] = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
        **filters,
    ) -> int:
        if (
            filters.get("last_active_before") is None
            and filters.get("started_before") is None
            and older_than_days is not None
        ):
            filters["last_active_before"] = time.time() - (
                float(older_than_days) * 86_400
            )
        where, params = self._prune_filter_where(source=source, **filters)
        removed_ids: List[str] = []

        async def _delete(connection):
            rows = await (
                await connection.execute(
                    f"SELECT s.id FROM sessions s WHERE {where}", params
                )
            ).fetchall()
            session_ids = [row["id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            await connection.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                session_ids,
            )
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})",
                session_ids,
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(session_ids)
            return len(session_ids)

        count = int(await self._write(_delete))
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    async def prune_empty_ghost_sessions(
        self, sessions_dir: Optional[Path] = None
    ) -> int:
        cutoff = time.time() - 86_400
        removed_ids: List[str] = []

        async def _delete(connection):
            rows = await (
                await connection.execute(
                    "SELECT id FROM sessions WHERE source = 'tui' "
                    "AND title IS NULL AND ended_at IS NOT NULL "
                    "AND started_at < ? AND NOT EXISTS ("
                    "SELECT 1 FROM messages "
                    "WHERE messages.session_id = sessions.id)",
                    (cutoff,),
                )
            ).fetchall()
            session_ids = [row["id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            await connection.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids
            )
            await self._delete_unreferenced_system_prompts(connection)
            removed_ids.extend(session_ids)
            return len(session_ids)

        count = int(await self._write(_delete))
        for removed_id in removed_ids:
            await self._remove_session_files(sessions_dir, removed_id)
        return count

    async def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuation sessions as ended.

        Targets child sessions that were never finalized: parent is ended
        with reason='compression', child has messages but no end_reason/ended_at
        and api_call_count=0.  Non-destructive: preserves all messages and sets
        end_reason='orphaned_compression'.  Fix for #20001.
        """
        cutoff = time.time() - 604800  # 7 days

        async def _finalize(connection):
            now = time.time()
            result = await connection.execute(
                """
                UPDATE sessions
                SET ended_at = ?,
                    end_reason = 'orphaned_compression'
                WHERE api_call_count = 0
                  AND end_reason IS NULL
                  AND ended_at IS NULL
                  AND started_at < ?
                  AND parent_session_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM sessions p
                      WHERE p.id = sessions.parent_session_id
                        AND p.end_reason = 'compression'
                        AND p.ended_at IS NOT NULL
                  )
                  AND EXISTS (
                      SELECT 1 FROM messages m
                      WHERE m.session_id = sessions.id
                  )
                """,
                (now, cutoff),
            )
            return result.rowcount

        return int(await self._write(_finalize) or 0)

    @staticmethod
    def _decode_message_row(row) -> Dict[str, Any]:
        """Decode one SQLite message row without touching ``SessionDB``."""
        message = dict(row)
        if "content" in message:
            message["content"] = SessionDB._decode_content(message["content"])
        if message.get("tool_calls"):
            try:
                message["tool_calls"] = json.loads(message["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to deserialize tool_calls in async session read; "
                    "falling back to []"
                )
                message["tool_calls"] = []
        if message.get("display_metadata") is not None:
            message["display_metadata"] = SessionDB._decode_display_metadata(
                message["display_metadata"]
            )
        return message

    async def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Load a session's messages through the native async connection."""
        connection = await self._get_connection()
        active_clause = "" if include_inactive else " AND active = 1"
        query = (
            "SELECT * FROM messages WHERE session_id = ?"
            f"{active_clause} ORDER BY id"
        )
        params: list[Any] = [session_id]
        if limit is not None or offset:
            query += " LIMIT ? OFFSET ?"
            params.extend([-1 if limit is None else limit, offset])
        cursor = await connection.execute(query, params)
        return [self._decode_message_row(row) for row in await cursor.fetchall()]

    async def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for one session."""
        connection = await self._get_connection()
        if session_id:
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE session_id = ?",
                (session_id,),
            )
        else:
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM messages"
            )
        row = await cursor.fetchone()
        return int(row["count"] if row is not None else 0)

    async def latest_message_row_id(
        self,
        session_id: str,
        *,
        role: str = "user",
        offset: int = 0,
        require_text: bool = True,
    ) -> Optional[int]:
        """Return the latest visible active message id for a user-facing role."""
        if not session_id or role not in {"user", "assistant"} or offset < 0:
            return None
        text_filter = (
            "AND content IS NOT NULL AND TRIM(content) != '' "
            if require_text
            else ""
        )
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = ? "
                f"AND active = 1 {text_filter}"
                "ORDER BY id DESC LIMIT 1 OFFSET ?",
                (session_id, role, int(offset)),
            )
        ).fetchone()
        return int(row["id"]) if row is not None else None

    async def latest_user_message_row_id(
        self, session_id: str
    ) -> Optional[int]:
        """Return the latest active user message id."""
        return await self.latest_message_row_id(session_id, role="user")

    async def get_message_role(
        self, session_id: str, row_id: int
    ) -> Optional[str]:
        """Return the role of an active message owned by *session_id*."""
        if not session_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT role FROM messages "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (int(row_id), session_id),
            )
        ).fetchone()
        return row["role"] if row is not None else None

    async def clear_messages(self, session_id: str) -> None:
        """Delete every transcript row and reset persisted counters."""
        async def _clear(connection):
            await connection.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            await connection.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 "
                "WHERE id = ?",
                (session_id,),
            )

        await self._write(_clear)

    async def rewind_to_message(
        self, session_id: str, target_message_id: int
    ) -> Dict[str, Any]:
        """Soft-delete the target user message and every later active row."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (target_message_id, session_id),
            )
        ).fetchone()
        if row is None:
            raise ValueError(
                f"message {target_message_id} not found in session {session_id}"
            )
        target = self._decode_message_row(row)
        if target.get("role") != "user":
            raise ValueError(
                "rewind target must be a 'user' message "
                f"(got role={target.get('role')!r}, id={target_message_id})"
            )

        async def _rewind(conn):
            cursor = await conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 1",
                (session_id, target_message_id),
            )
            ids = [item[0] for item in await cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await conn.execute(
                    f"UPDATE messages SET active = 0 WHERE id IN ({placeholders})",
                    ids,
                )
            await conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            head = await (
                await conn.execute(
                    "SELECT MAX(id) FROM messages "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )
            ).fetchone()
            return ids, head[0] if head and head[0] is not None else None

        rewound, new_head_id = await self._write(_rewind)
        return {
            "rewound_count": len(rewound),
            "target_message": target,
            "new_head_id": new_head_id,
        }

    async def restore_rewound(
        self, session_id: str, since_message_id: int
    ) -> int:
        """Mark inactive messages with id >= *since_message_id* active again.

        Returns the number of rows flipped back to ``active=1``.
        Intended for undo-of-rewind and test cleanup; not wired to a
        slash command in v1.
        """

        async def _do(connection):
            cursor = await connection.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 0",
                (session_id, since_message_id),
            )
            ids = [r[0] for r in await cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await connection.execute(
                    f"UPDATE messages SET active = 1 "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids)

        return await self._write(_do)

    async def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Return an async anchored message window with boundary counts."""
        connection = await self._get_connection()
        window = max(0, int(window))
        anchor = await (
            await connection.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            )
        ).fetchone()
        if anchor is None:
            return {"window": [], "messages_before": 0, "messages_after": 0}
        before = await (
            await connection.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            )
        ).fetchall()
        after = await (
            await connection.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            )
        ).fetchall()
        rows = list(reversed(before)) + list(after)
        return {
            "window": [self._decode_message_row(row) for row in rows],
            "messages_before": max(0, len(before) - 1),
            "messages_after": len(after),
        }

    async def get_message_storage_state(
        self, message_id: int
    ) -> Optional[Dict[str, Any]]:
        """Return storage flags for one message id without a raw connection."""
        if not message_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT session_id, active, compacted FROM messages WHERE id = ?",
                (message_id,),
            )
        ).fetchone()
        return dict(row) if row is not None else None

    async def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant"),
    ) -> Dict[str, Any]:
        """Return an anchored window and bookends using only aiosqlite."""
        primitive = await self.get_messages_around(
            session_id, around_message_id, window=window
        )
        rows = primitive["window"]
        if not rows:
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }
        if keep_roles is None:
            filtered = rows
        else:
            allowed = set(keep_roles)
            filtered = [
                row for row in rows
                if row.get("id") == around_message_id or row.get("role") in allowed
            ]
        bookend = max(0, int(bookend))
        starts: list[Dict[str, Any]] = []
        ends: list[Dict[str, Any]] = []
        if bookend:
            connection = await self._get_connection()
            role_sql = ""
            role_params: list[Any] = []
            if keep_roles is not None:
                role_sql = " AND role IN (" + ",".join("?" for _ in keep_roles) + ")"
                role_params = list(keep_roles)
            start_rows = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND id < ?"
                    f"{role_sql} AND length(content) > 0 ORDER BY id ASC LIMIT ?",
                    (session_id, rows[0]["id"], *role_params, bookend),
                )
            ).fetchall()
            end_rows = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND id > ?"
                    f"{role_sql} AND length(content) > 0 ORDER BY id DESC LIMIT ?",
                    (session_id, rows[-1]["id"], *role_params, bookend),
                )
            ).fetchall()
            starts = [self._decode_message_row(row) for row in start_rows]
            ends = [self._decode_message_row(row) for row in reversed(end_rows)]
        return {
            "window": filtered,
            "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": starts,
            "bookend_end": ends,
        }

    async def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session title through the async connection."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) "
                "AS _system_prompt_resolved "
                "FROM sessions s LEFT JOIN system_prompts sp "
                "ON sp.hash = s.system_prompt_hash WHERE s.title = ?",
                (title,),
            )
        ).fetchone()
        return self._session_row_dict(row) if row is not None else None

    async def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve an exact or numbered continuation title asynchronously."""
        exact = await self.get_session_by_title(title)
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\' "
                "ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
        ).fetchall()
        if rows:
            return rows[0]["id"]
        return exact["id"] if exact else None

    async def fts_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """Return deferred FTS rebuild progress without sync metadata access."""
        high_water = await self.get_meta("fts_rebuild_high_water")
        if high_water is None:
            return None
        try:
            total = int(high_water)
            indexed = int(await self.get_meta("fts_rebuild_progress") or 0)
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        return {
            "pending": True,
            "total": total,
            "indexed": indexed,
            "percent": min(100, int(100 * indexed / total)),
        }

    async def _fts_rebuild_finish(self) -> None:
        """Finalize the deferred rebuild with a boundary sweep."""
        include_trigram = self._trigram_available

        async def _finish(connection):
            high_water_row = await (
                await connection.execute(
                    "SELECT value FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water'"
                )
            ).fetchone()
            if high_water_row is not None:
                high_water = int(high_water_row[0])
                lower, upper = high_water - 1000, high_water + 1000
                await connection.execute(
                    "INSERT INTO messages_fts"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m WHERE m.id > ? AND m.id <= ? "
                    "AND NOT EXISTS (SELECT 1 FROM messages_fts_docsize d "
                    "WHERE d.id = m.id)",
                    (lower, upper),
                )
                if include_trigram:
                    await connection.execute(
                        "INSERT INTO messages_fts_trigram"
                        "(rowid, content, tool_name, tool_calls) "
                        "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                        "FROM messages m WHERE m.id > ? AND m.id <= ? "
                        "AND m.role <> 'tool' AND NOT EXISTS "
                        "(SELECT 1 FROM messages_fts_trigram_docsize d "
                        "WHERE d.id = m.id)",
                        (lower, upper),
                    )
            await connection.execute(
                "DELETE FROM state_meta WHERE key IN "
                "('fts_rebuild_high_water', 'fts_rebuild_progress')"
            )

        await self._write(_finish)
        logger.info("Deferred FTS rebuild complete — all messages indexed.")

    async def _has_fts_trash(self, connection) -> bool:
        cursor = await connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE ? ESCAPE '\\' LIMIT 1",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        )
        try:
            return await cursor.fetchone() is not None
        finally:
            await cursor.close()

    async def _fts_teardown_trash_step(self) -> bool:
        """Tear down one bounded chunk of a demoted v22 shadow table."""
        connection = await self._get_connection()
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE ? ESCAPE '\\'",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        )
        try:
            trash = [row[0] for row in await cursor.fetchall()]
        finally:
            await cursor.close()
        if not trash:
            return False
        table = trash[0]

        async def _teardown(connection):
            info_cursor = await connection.execute(f"PRAGMA table_info({table})")
            try:
                primary_keys = [
                    row[1] for row in await info_cursor.fetchall() if row[5] > 0
                ]
            finally:
                await info_cursor.close()
            key = ", ".join(primary_keys) if primary_keys else "rowid"
            delete_cursor = await connection.execute(
                f"DELETE FROM {table} WHERE ({key}) IN "
                f"(SELECT {key} FROM {table} LIMIT "
                f"{self._FTS_REBUILD_CHUNK_ROWS})"
            )
            try:
                deleted = delete_cursor.rowcount
            finally:
                await delete_cursor.close()
            if deleted == 0:
                await connection.execute(f"DROP TABLE IF EXISTS {table}")
                logger.info("Old FTS shadow table %s torn down.", table)
            return True

        try:
            return bool(await self._write(_teardown))
        except sqlite3.OperationalError as exc:
            logger.debug("FTS trash teardown chunk failed (will retry): %s", exc)
            return True

    async def fts_rebuild_step(self) -> bool:
        """Backfill one chunk of the deferred FTS rebuild."""
        await self._get_connection()
        if not self._fts_enabled:
            return False
        high_water_raw = await self.get_meta("fts_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)
        include_trigram = self._trigram_available

        async def _backfill(connection):
            progress_row = await (
                await connection.execute(
                    "SELECT value FROM state_meta "
                    "WHERE key = 'fts_rebuild_progress'"
                )
            ).fetchone()
            if progress_row is None:
                return False
            progress = int(progress_row[0])
            if progress >= high_water:
                return False
            upper = min(progress + self._FTS_REBUILD_CHUNK_ROWS, high_water)
            await connection.execute(
                "INSERT INTO messages_fts"
                "(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id > ? AND id <= ?",
                (progress, upper),
            )
            if include_trigram:
                await connection.execute(
                    "INSERT INTO messages_fts_trigram"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT id, content, tool_name, tool_calls FROM messages "
                    "WHERE id > ? AND id <= ? AND role <> 'tool'",
                    (progress, upper),
                )
            await connection.execute(
                "UPDATE state_meta SET value = ? "
                "WHERE key = 'fts_rebuild_progress'",
                (str(upper),),
            )
            return upper < high_water

        try:
            more = await self._write(_backfill)
        except sqlite3.OperationalError as exc:
            logger.debug("FTS rebuild chunk failed (will retry): %s", exc)
            return True
        if more is False:
            status = await self.fts_rebuild_status()
            if status is not None and status["indexed"] >= status["total"]:
                await self._fts_rebuild_finish()
            return False
        return bool(more)

    async def fts_cjk_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """CJK-index backfill progress, or None when none is pending."""
        high_water = await self.get_meta("fts_cjk_rebuild_high_water")
        if high_water is None:
            return None
        total = int(high_water)
        indexed = int(await self.get_meta("fts_cjk_rebuild_progress") or 0)
        if total <= 0:
            return None
        return {
            "pending": True,
            "total": total,
            "indexed": indexed,
            "percent": min(100, int(100 * indexed / total)),
        }

    async def fts_cjk_rebuild_step(self) -> bool:
        """Backfill one chunk of the CJK index. True while work remains."""
        await self._get_connection()
        if not self._fts_enabled or not self._fts_cjk_loaded:
            return False
        high_water_raw = await self.get_meta("fts_cjk_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)

        async def _backfill(connection):
            progress_row = await (
                await connection.execute(
                    "SELECT value FROM state_meta "
                    "WHERE key = 'fts_cjk_rebuild_progress'"
                )
            ).fetchone()
            if progress_row is None:
                return False
            progress = int(progress_row[0])
            if progress >= high_water:
                return False
            upper = min(progress + self._FTS_REBUILD_CHUNK_ROWS, high_water)
            await connection.execute(
                "INSERT INTO messages_fts_cjk"
                "(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id > ? AND id <= ? AND role <> 'tool'",
                (progress, upper),
            )
            await connection.execute(
                "UPDATE state_meta SET value = ? "
                "WHERE key = 'fts_cjk_rebuild_progress'",
                (str(upper),),
            )
            return upper < high_water

        try:
            more = await self._write(_backfill)
        except sqlite3.OperationalError as exc:
            logger.debug("CJK FTS rebuild chunk failed (will retry): %s", exc)
            return True
        if more is False:
            status = await self.fts_cjk_rebuild_status()
            if status is not None and status["indexed"] >= status["total"]:
                await self._fts_cjk_rebuild_finish()
            return False
        return bool(more)

    async def _fts_cjk_rebuild_finish(self) -> None:
        """Boundary sweep, clear CJK markers, and enable CJK reads."""

        async def _finish(connection):
            high_water_row = await (
                await connection.execute(
                    "SELECT value FROM state_meta "
                    "WHERE key = 'fts_cjk_rebuild_high_water'"
                )
            ).fetchone()
            if high_water_row is not None:
                high_water = int(high_water_row[0])
                lower, upper = high_water - 1000, high_water + 1000
                await connection.execute(
                    "INSERT INTO messages_fts_cjk"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m WHERE m.id > ? AND m.id <= ? "
                    "AND m.role <> 'tool' AND NOT EXISTS "
                    "(SELECT 1 FROM messages_fts_cjk_docsize d "
                    "WHERE d.id = m.id)",
                    (lower, upper),
                )
            await connection.execute(
                "DELETE FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', "
                "'fts_cjk_rebuild_progress')"
            )

        await self._write(_finish)
        self._fts_cjk_available = True
        logger.info("CJK FTS index backfill complete — serving CJK search.")

    async def _fts_cjk_reset_if_stale(self) -> None:
        """Reset a stale CJK index so it can be rebuilt from scratch."""
        if not self._fts_cjk_loaded:
            return

        async def _reset(connection):
            stale = await (
                await connection.execute(
                    "SELECT 1 FROM state_meta WHERE key = ?",
                    (FTS_CJK_STALE_KEY,),
                )
            ).fetchone()
            if stale is None:
                return False
            for trigger in _FTS_CJK_TRIGGERS:
                await connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            await connection.execute("DROP TABLE IF EXISTS messages_fts_cjk")
            await connection.execute("DROP VIEW IF EXISTS messages_fts_cjk_src")
            await connection.execute(
                "DELETE FROM state_meta WHERE key IN "
                f"('{FTS_CJK_STALE_KEY}', 'fts_cjk_rebuild_high_water', "
                "'fts_cjk_rebuild_progress')"
            )
            return True

        if await self._write(_reset):
            async with self._get_write_lock():
                connection = await self._get_connection()
                await self._ensure_fts_cjk_schema(connection)
                await connection.commit()

    @staticmethod
    async def _fts_external_index_empty_with_messages(connection) -> bool:
        """Return whether a populated DB has an empty external FTS index."""
        try:
            has_message = await (
                await connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM messages)"
                )
            ).fetchone()
            if not has_message[0]:
                return False
            has_fts = await (
                await connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM messages_fts_docsize)"
                )
            ).fetchone()
            return not has_fts[0]
        except sqlite3.OperationalError:
            return False

    @staticmethod
    async def _fts_index_known_empty(connection) -> bool:
        try:
            row = await (
                await connection.execute(
                    "SELECT COUNT(*) FROM messages_fts_docsize"
                )
            ).fetchone()
            return int(row[0]) == 0
        except sqlite3.OperationalError:
            return True

    @staticmethod
    async def _reset_fts_index_to_empty(connection) -> None:
        for table in ("messages_fts", "messages_fts_trigram"):
            try:
                await connection.execute(
                    f"INSERT INTO {table}({table}) VALUES('delete-all')"
                )
            except sqlite3.OperationalError:
                pass

    async def _seed_fts_rebuild_markers(
        self, connection, *, force: bool = False
    ) -> int:
        existing = await (
            await connection.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water'"
            )
        ).fetchone()
        if existing is not None and not force:
            high_water = int(existing[0])
            progress = await (
                await connection.execute(
                    "SELECT value FROM state_meta "
                    "WHERE key = 'fts_rebuild_progress'"
                )
            ).fetchone()
            if progress is None:
                if not await self._fts_index_known_empty(connection):
                    await self._reset_fts_index_to_empty(connection)
                await connection.execute(
                    "INSERT INTO state_meta (key, value) VALUES "
                    "('fts_rebuild_progress', '0') "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                )
            return high_water

        row = await (
            await connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages"
            )
        ).fetchone()
        high_water = int(row[0])
        for key, value in (
            ("fts_rebuild_high_water", str(high_water)),
            ("fts_rebuild_progress", "0"),
        ):
            await connection.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        return high_water

    async def _repair_optimize_bookkeeping(self) -> None:
        """Heal interrupted demote/backfill bookkeeping before optimize."""

        async def _repair(connection):
            existing = await (
                await connection.execute(
                    "SELECT value FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water'"
                )
            ).fetchone()
            if existing is not None:
                progress = await (
                    await connection.execute(
                        "SELECT 1 FROM state_meta "
                        "WHERE key = 'fts_rebuild_progress'"
                    )
                ).fetchone()
                if progress is None:
                    if not await self._fts_index_known_empty(connection):
                        await self._reset_fts_index_to_empty(connection)
                    await connection.execute(
                        "INSERT INTO state_meta (key, value) VALUES "
                        "('fts_rebuild_progress', '0') "
                        "ON CONFLICT(key) DO UPDATE SET value = '0'"
                    )
                return
            if await self._db_has_legacy_inline_fts(connection):
                return
            if await self._fts_external_index_empty_with_messages(connection):
                await connection.execute(
                    "DELETE FROM state_meta WHERE key = 'fts_storage_version'"
                )
                await self._seed_fts_rebuild_markers(connection, force=True)

        await self._write(_repair)

    async def fts_optimize_available(self) -> bool:
        """Return whether storage migration, backfill, or teardown is pending."""
        connection = await self._get_connection()
        if not self._fts_enabled or self.read_only:
            return False
        async with self._get_write_lock():
            if await self._db_has_legacy_inline_fts(connection):
                return True
            pending = await (
                await connection.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                )
            ).fetchone()
            if pending is not None:
                return True
            if self._fts_cjk_loaded:
                cjk_pending = await (
                    await connection.execute(
                        "SELECT 1 FROM state_meta WHERE key IN "
                        f"('fts_cjk_rebuild_high_water', "
                        f"'{FTS_CJK_STALE_KEY}') LIMIT 1"
                    )
                ).fetchone()
                if cjk_pending is not None:
                    return True
            if await self._has_fts_trash(connection):
                return True
            return await self._fts_external_index_empty_with_messages(connection)

    async def _demote_legacy_fts_to_trash(self) -> int:
        """Demote legacy inline FTS tables and seed a resumable rebuild."""

        async def _stage(connection):
            await self._drop_fts_triggers(connection)
            await connection.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
            had_row = await (
                await connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('messages_fts', 'messages_fts_trigram') "
                    "AND sql LIKE 'CREATE VIRTUAL TABLE%' LIMIT 1"
                )
            ).fetchone()
            if had_row is not None:
                await connection.execute("PRAGMA writable_schema=ON")
                await connection.execute(
                    "DELETE FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('messages_fts', 'messages_fts_trigram') "
                    "AND sql LIKE 'CREATE VIRTUAL TABLE%'"
                )
                await connection.execute("PRAGMA writable_schema=RESET")
                cursor = await connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND (name LIKE 'messages_fts_%' ESCAPE '\\' "
                    "OR name LIKE 'messages_fts_trigram_%' ESCAPE '\\')"
                )
                try:
                    shadows = [row[0] for row in await cursor.fetchall()]
                finally:
                    await cursor.close()
                for shadow in shadows:
                    await connection.execute(
                        f"ALTER TABLE {shadow} "
                        f"RENAME TO fts_v22_trash_{shadow}"
                    )
            high_water = await self._seed_fts_rebuild_markers(
                connection, force=True
            )
            await connection.execute(
                "DELETE FROM state_meta WHERE key = 'fts_optimize_available'"
            )
            return high_water

        high_water = int(await self._write(_stage))
        async with self._get_write_lock():
            connection = await self._get_connection()
            base_ok = await self._ensure_fts_schema(
                connection, "messages_fts", FTS_SQL
            )
            trigram_ok = await self._ensure_fts_schema(
                connection, "messages_fts_trigram", FTS_TRIGRAM_SQL
            )
            self._trigram_available = bool(trigram_ok)
            if not base_ok:
                raise sqlite3.OperationalError(
                    "failed to create v23 messages_fts during "
                    "optimize-storage demote"
                )
            await connection.commit()
        return high_water

    async def optimize_fts_storage(
        self,
        *,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        vacuum: bool = True,
    ) -> Dict[str, Any]:
        """Migrate legacy FTS storage and finish all resumable backfills."""
        connection = await self._get_connection()
        if not self._fts_enabled:
            return {"ok": False, "reason": "fts5_unavailable"}
        if self.read_only:
            return {"ok": False, "reason": "read_only"}

        await self._repair_optimize_bookkeeping()
        async with self._get_write_lock():
            legacy = await self._db_has_legacy_inline_fts(connection)
        pending = await self.get_meta("fts_rebuild_high_water") is not None
        if legacy and not pending:
            await self._demote_legacy_fts_to_trash()
        elif pending and not legacy:
            async with self._get_write_lock():
                base_ok = await self._ensure_fts_schema(
                    connection, "messages_fts", FTS_SQL
                )
                trigram_ok = await self._ensure_fts_schema(
                    connection, "messages_fts_trigram", FTS_TRIGRAM_SQL
                )
                self._trigram_available = bool(trigram_ok)
                if not base_ok:
                    raise sqlite3.OperationalError(
                        "failed to re-create v23 messages_fts "
                        "on optimize-storage resume"
                    )
                await connection.commit()

        await self._fts_cjk_reset_if_stale()
        if self._fts_cjk_loaded:
            async with self._get_write_lock():
                await self._ensure_fts_cjk_schema(connection)
                await connection.commit()

        async def _emit(phase: str) -> None:
            if progress_cb is None:
                return
            status = await self.fts_rebuild_status()
            if status is None:
                status = await self.fts_cjk_rebuild_status()
            progress_cb(
                {
                    "phase": phase,
                    "percent": status["percent"] if status else 100,
                    "indexed": status["indexed"] if status else 0,
                    "total": status["total"] if status else 0,
                }
            )

        async def _pause(chunk_seconds: float) -> None:
            await asyncio.sleep(
                max(
                    self._FTS_REBUILD_MIN_PAUSE,
                    chunk_seconds * self._FTS_REBUILD_DUTY_FACTOR,
                )
            )

        await _emit("backfill")
        while True:
            started = time.monotonic()
            if not await self.fts_rebuild_step():
                break
            await _emit("backfill")
            await _pause(time.monotonic() - started)
        await _emit("backfill")

        while True:
            started = time.monotonic()
            if not await self.fts_cjk_rebuild_step():
                break
            await _emit("backfill")
            await _pause(time.monotonic() - started)

        await _emit("teardown")
        while True:
            started = time.monotonic()
            if not await self._fts_teardown_trash_step():
                break
            await _emit("teardown")
            await _pause(time.monotonic() - started)

        async with self._get_write_lock():
            still_pending = await (
                await connection.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                )
            ).fetchone()
            still_trash = await self._has_fts_trash(connection)
            empty_index = await self._fts_external_index_empty_with_messages(
                connection
            )
        if still_pending is not None or still_trash or empty_index:
            reason = (
                "backfill_incomplete"
                if still_pending is not None or empty_index
                else "teardown_incomplete"
            )
            logger.warning(
                "FTS storage optimization did not settle (%s): "
                "pending=%s trash=%s empty_index=%s",
                reason,
                still_pending is not None,
                still_trash,
                empty_index,
            )
            return {"ok": False, "reason": reason, "vacuumed": None}

        vacuum_ok = None
        if vacuum:
            await _emit("vacuum")
            try:
                async with self._get_write_lock():
                    cursor = await connection.execute("VACUUM")
                    await cursor.close()
                vacuum_ok = True
            except sqlite3.OperationalError as exc:
                logger.warning("VACUUM after FTS optimize failed: %s", exc)
                vacuum_ok = False
            try:
                async with self._get_write_lock():
                    cursor = await connection.execute(
                        "PRAGMA wal_checkpoint(TRUNCATE)"
                    )
                    await cursor.close()
            except Exception as exc:
                logger.debug(
                    "WAL checkpoint (TRUNCATE) after optimize VACUUM "
                    "failed: %s",
                    exc,
                )

        async def _settle(connection):
            pending_row = await (
                await connection.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                )
            ).fetchone()
            if pending_row is not None:
                return "backfill_incomplete"
            if await self._has_fts_trash(connection):
                return "teardown_incomplete"
            if await self._fts_external_index_empty_with_messages(connection):
                return "backfill_incomplete"
            await connection.execute(
                "INSERT INTO state_meta (key, value) VALUES "
                "('fts_storage_version', ?) ON CONFLICT(key) "
                "DO UPDATE SET value = excluded.value",
                (str(FTS_STORAGE_VERSION),),
            )
            await connection.execute(
                "DELETE FROM state_meta WHERE key = 'fts_optimize_available'"
            )
            await connection.execute(
                "UPDATE schema_version SET version = ? WHERE version < ?",
                (SCHEMA_VERSION, SCHEMA_VERSION),
            )
            return None

        refusal = await self._write(_settle)
        if refusal is not None:
            logger.warning("FTS storage optimization settle refused (%s)", refusal)
            return {"ok": False, "reason": refusal, "vacuumed": vacuum_ok}
        await _emit("done")
        logger.info(
            "FTS storage optimization complete (layout v%d).",
            FTS_STORAGE_VERSION,
        )
        return {"ok": True, "vacuumed": vacuum_ok}

    async def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
        fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Instrumented native-async message search.

        This preserves the v2026.8.3 routing and result contract while keeping
        every SQLite operation awaitable.
        """
        started = time.time()
        rows = None
        try:
            rows = await self._search_messages_impl(
                query,
                source_filter=source_filter,
                exclude_sources=exclude_sources,
                role_filter=role_filter,
                limit=limit,
                offset=offset,
                sort=sort,
                include_inactive=include_inactive,
                fields=fields,
            )
            return rows
        finally:
            try:
                threshold = float(os.getenv("HERMES_SEARCH_SLOW_MS", "1000"))
            except (TypeError, ValueError):
                threshold = 1000.0
            elapsed_ms = (time.time() - started) * 1000.0
            if elapsed_ms >= threshold:
                logger.info(
                    "slow session search: path=%s elapsed=%.0fms rows=%s query=%r",
                    self._describe_search_path(query),
                    elapsed_ms,
                    len(rows) if rows is not None else "err",
                    query[:200],
                )

    def _describe_search_path(self, query: str) -> str:
        """Return the upstream route name used for slow-search diagnostics."""
        try:
            sanitized = self._sanitize_fts5_query(query or "")
            if not sanitized:
                return "empty"
            if not self._contains_cjk(sanitized):
                return "fts5"
            raw = sanitized.strip('"').strip()
            if self._fts_cjk_available and not self._has_lone_cjk_run(raw):
                return "fts_cjk"
            tokens = [
                token
                for token in raw.split()
                if token.upper() not in {"AND", "OR", "NOT"}
                and self._contains_cjk(token)
            ]
            short = any(self._count_cjk(token) < 3 for token in tokens)
            if self._count_cjk(raw) >= 3 and not short and self._trigram_available:
                return "trigram"
            return "like_scan"
        except Exception:
            return "unknown"

    async def _run_trigram_search(
        self,
        raw_query: str,
        *,
        table: str = "messages_fts_trigram",
        order_by_sql: str,
        include_inactive: bool,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Search one of the two upstream substring-capable FTS indexes."""
        if table not in {"messages_fts_trigram", "messages_fts_cjk"}:
            raise ValueError(f"unsupported FTS table: {table}")
        parts = [
            token
            if token.upper() in {"AND", "OR", "NOT"}
            else '"' + token.replace('"', '""') + '"'
            for token in raw_query.split()
        ]
        where = [f"{table} MATCH ?"]
        params: list[Any] = [" ".join(parts)]
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   snippet({table}, -1, '>>>', '<<<', '...', 40) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM {table}
            JOIN messages m ON m.id = {table}.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        connection = await self._get_connection()
        try:
            cursor = await connection.execute(sql, params)
            try:
                return [dict(row) for row in await cursor.fetchall()]
            finally:
                await cursor.close()
        except sqlite3.DatabaseError as exc:
            if not await self._try_runtime_fts_rebuild(exc):
                return None
            try:
                cursor = await connection.execute(sql, params)
                try:
                    return [dict(row) for row in await cursor.fetchall()]
                finally:
                    await cursor.close()
            except sqlite3.DatabaseError:
                logger.warning(
                    "%s search still failing after in-place rebuild; "
                    "falling back to LIKE",
                    table,
                )
                return None

    async def _search_messages_impl(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
        fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search with v2026.8.3 routing and result semantics."""
        result_fields = self._search_message_fields(fields)
        connection = await self._get_connection()
        if not self._fts_enabled or not query or not query.strip():
            return []
        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        sort_norm = sort.strip().lower() if isinstance(sort, str) else None
        if sort_norm not in {"newest", "oldest"}:
            sort_norm = None
        if sort_norm == "newest":
            order_by_sql = "ORDER BY m.timestamp DESC, rank"
        elif sort_norm == "oldest":
            order_by_sql = "ORDER BY m.timestamp ASC, rank"
        else:
            order_by_sql = "ORDER BY rank"

        where = ["messages_fts MATCH ?"]
        params: list[Any] = [query]
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   snippet(messages_fts, -1, '>>>', '<<<', '...', 40) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        matches: List[Dict[str, Any]] = []
        is_cjk = self._contains_cjk(query)
        used_like = False
        if is_cjk:
            raw_query = query.strip('"').strip()
            cjk_tokens = [
                token
                for token in raw_query.split()
                if token.upper() not in {"AND", "OR", "NOT"}
                and self._contains_cjk(token)
            ]
            any_short_cjk = any(
                self._count_cjk(token) < 3 for token in cjk_tokens
            )
            wants_tool_rows = bool(role_filter) and "tool" in role_filter
            substring_matches: Optional[List[Dict[str, Any]]] = None
            if (
                self._fts_cjk_available
                and not wants_tool_rows
                and not self._has_lone_cjk_run(raw_query)
            ):
                substring_matches = await self._run_trigram_search(
                    raw_query,
                    table="messages_fts_cjk",
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
            if (
                substring_matches is None
                and self._count_cjk(raw_query) >= 3
                and not any_short_cjk
                and self._trigram_available
                and not wants_tool_rows
            ):
                substring_matches = await self._run_trigram_search(
                    raw_query,
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
            if substring_matches is not None:
                matches = substring_matches
            else:
                used_like = True
                tokens = [
                    token
                    for token in raw_query.split()
                    if token.upper() not in {"AND", "OR", "NOT"}
                ] or [raw_query]
                token_clauses = []
                like_params: list[Any] = []
                for token in tokens:
                    escaped = (
                        token.replace("\\", "\\\\")
                        .replace("%", "\\%")
                        .replace("_", "\\_")
                    )
                    token_clauses.append(
                        "(m.content LIKE ? ESCAPE '\\' "
                        "OR m.tool_name LIKE ? ESCAPE '\\' "
                        "OR m.tool_calls LIKE ? ESCAPE '\\')"
                    )
                    like_params.extend([f"%{escaped}%"] * 3)
                like_where = ["(" + " OR ".join(token_clauses) + ")"]
                if not include_inactive:
                    like_where.append("(m.active = 1 OR m.compacted = 1)")
                if source_filter is not None:
                    like_where.append(
                        "s.source IN ("
                        + ",".join("?" for _ in source_filter)
                        + ")"
                    )
                    like_params.extend(source_filter)
                if exclude_sources is not None:
                    like_where.append(
                        "s.source NOT IN ("
                        + ",".join("?" for _ in exclude_sources)
                        + ")"
                    )
                    like_params.extend(exclude_sources)
                if role_filter:
                    like_where.append(
                        "m.role IN ("
                        + ",".join("?" for _ in role_filter)
                        + ")"
                    )
                    like_params.extend(role_filter)
                like_sql = f"""
                    SELECT m.id, m.session_id, m.role,
                           substr(m.content,
                                  max(1, instr(m.content, ?) - 40),
                                  120) AS snippet,
                           m.content, m.timestamp, m.tool_name,
                           s.source, s.model,
                           s.started_at AS session_started
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(like_where)}
                    ORDER BY m.timestamp DESC
                    LIMIT ? OFFSET ?
                """
                cursor = await connection.execute(
                    like_sql,
                    [tokens[0], *like_params, limit, offset],
                )
                try:
                    matches = [dict(row) for row in await cursor.fetchall()]
                finally:
                    await cursor.close()
        else:
            try:
                cursor = await connection.execute(sql, params)
                try:
                    matches = [dict(row) for row in await cursor.fetchall()]
                finally:
                    await cursor.close()
            except sqlite3.DatabaseError as exc:
                if (
                    isinstance(exc, sqlite3.OperationalError)
                    and not self._is_fts_write_corruption_error(exc)
                ):
                    return []
                # A new caller cancellation must supersede this stale FTS error.
                if not await self._try_runtime_fts_rebuild(exc):  # noqa: ASYNC120
                    raise
                cursor = await connection.execute(sql, params)
                try:
                    matches = [dict(row) for row in await cursor.fetchall()]
                finally:
                    await cursor.close()

        rebuild_status = await self.fts_rebuild_status()
        if not used_like and rebuild_status is not None and len(matches) < limit:
            try:
                gap_matches = await self._search_unindexed_gap(
                    query,
                    limit - len(matches),
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                )
                seen_ids = {match["id"] for match in matches}
                matches.extend(
                    match for match in gap_matches if match["id"] not in seen_ids
                )
            except sqlite3.OperationalError as exc:
                logger.debug("Unindexed-gap supplement skipped: %s", exc)

        if (
            not matches
            and not is_cjk
            and not (bool(role_filter) and "tool" in role_filter)
        ):
            fallback_query = query.strip('"').strip()
            if self._fts_cjk_available:
                cjk_matches = await self._run_trigram_search(
                    fallback_query,
                    table="messages_fts_cjk",
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
                if cjk_matches:
                    matches = cjk_matches
            if (
                not matches
                and self._trigram_available
                and self._trigram_eligible_tokens(query)
            ):
                trigram_matches = await self._run_trigram_search(
                    fallback_query,
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
                if trigram_matches:
                    matches = trigram_matches

        context_matches = (
            matches if result_fields is None or "context" in result_fields else ()
        )
        for match in context_matches:
            try:
                cursor = await connection.execute(
                    """WITH target AS (
                           SELECT session_id, timestamp, id
                           FROM messages WHERE id = ?
                       )
                       SELECT role, content FROM (
                           SELECT m.id, m.timestamp, m.role, m.content
                           FROM messages m JOIN target t
                             ON t.session_id = m.session_id
                           WHERE m.timestamp < t.timestamp
                              OR (m.timestamp = t.timestamp AND m.id < t.id)
                           ORDER BY m.timestamp DESC, m.id DESC LIMIT 1
                       )
                       UNION ALL
                       SELECT role, content FROM messages WHERE id = ?
                       UNION ALL
                       SELECT role, content FROM (
                           SELECT m.id, m.timestamp, m.role, m.content
                           FROM messages m JOIN target t
                             ON t.session_id = m.session_id
                           WHERE m.timestamp > t.timestamp
                              OR (m.timestamp = t.timestamp AND m.id > t.id)
                           ORDER BY m.timestamp ASC, m.id ASC LIMIT 1
                       )""",
                    (match["id"], match["id"]),
                )
                try:
                    context_rows = await cursor.fetchall()
                finally:
                    await cursor.close()
                context = []
                for row in context_rows:
                    decoded = self._decode_content(row["content"])
                    if isinstance(decoded, list):
                        texts = [
                            part.get("text", "")
                            for part in decoded
                            if isinstance(part, dict)
                            and part.get("type") == "text"
                        ]
                        preview = " ".join(
                            text for text in texts if text
                        ).strip()
                        preview = preview or "[multimodal content]"
                    elif isinstance(decoded, str):
                        preview = decoded
                    else:
                        preview = ""
                    context.append(
                        {"role": row["role"], "content": preview[:200]}
                    )
                match["context"] = context
            except Exception:
                match["context"] = []

        for match in matches:
            match.pop("content", None)
        if result_fields is not None:
            matches = [
                {field: match[field] for field in result_fields if field in match}
                for match in matches
            ]
        return matches

    @staticmethod
    def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message for marker in _MALFORMED_SCHEMA_MARKERS
        ) or ("fts5" in message and "corrupt" in message)

    async def _try_runtime_fts_rebuild(
        self, exc: sqlite3.DatabaseError
    ) -> bool:
        """Perform the upstream one-shot FTS repair without blocking SQLite."""
        if (
            self._fts_runtime_rebuild_attempted
            or not self._fts_enabled
            or not self._is_fts_write_corruption_error(exc)
        ):
            return False
        self._fts_runtime_rebuild_attempted = True
        logger.warning(
            "state.db hit an FTS-corruption error (%s); attempting an "
            "in-place FTS rebuild",
            exc,
        )
        try:
            rebuilt = await self.rebuild_fts()
        except Exception as rebuild_exc:
            logger.error("In-place FTS rebuild failed: %s", rebuild_exc)
            return False
        return rebuilt > 0

    async def _search_unindexed_gap(
        self,
        fts_query: str,
        limit: int,
        *,
        include_inactive: bool = False,
        source_filter: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        role_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """LIKE-scan only the deferred FTS rebuild's unindexed id range."""
        status = await self.fts_rebuild_status()
        if status is None or limit <= 0:
            return []
        terms = []
        for raw_token in re.findall(r'"[^"]+"|\S+', fts_query):
            token = raw_token.strip('"').strip("*").strip()
            if token and token.upper() not in {"AND", "OR", "NOT", "NEAR"}:
                terms.append(token)
        if not terms:
            return []
        where = ["m.id > ? AND m.id <= ?"]
        params: list[Any] = [status["indexed"], status["total"]]
        for term in terms:
            escaped = (
                term.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            where.append(
                "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' "
                "OR m.tool_calls LIKE ? ESCAPE '\\')"
            )
            params.extend([f"%{escaped}%"] * 3)
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(
                "s.source IN (" + ",".join("?" for _ in source_filter) + ")"
            )
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(
                "s.source NOT IN ("
                + ",".join("?" for _ in exclude_sources)
                + ")"
            )
            params.extend(exclude_sources)
        if role_filter:
            where.append(
                "m.role IN (" + ",".join("?" for _ in role_filter) + ")"
            )
            params.extend(role_filter)
        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   substr(m.content,
                          max(1, instr(m.content, ?) - 40),
                          120) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY m.timestamp DESC
            LIMIT ?
        """
        connection = await self._get_connection()
        cursor = await connection.execute(sql, [terms[0], *params, limit])
        try:
            return [dict(row) for row in await cursor.fetchall()]
        finally:
            await cursor.close()

    async def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
        workspace_key: str = None,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column
        (freshest of ``last_activity_at`` and latest message timestamp,
        else ``started_at``), ordered by most-recently-used first.

        Pass ``workspace_key`` to scope rows to one workspace - matching
        :func:`workspace_key` semantics (git repo root, else cwd). Used by
        ``hermes -c``/``--resume`` so the "last" session is the last one in
        the *current* workspace, not the global MRU.
        """
        select_with_last_active = (
            "SELECT s.*, "
            "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
            f"{_sql_session_last_active('s')} AS last_active "
            "FROM sessions s "
            "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        )
        where_clauses = []
        params: list = []
        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if workspace_key:
            ws_clause, ws_params = _workspace_key_clause(workspace_key)
            where_clauses.append(ws_clause)
            params.extend(ws_params)
        where_sql = (
            f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        )
        params.extend([limit, offset])
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                f"{select_with_last_active}"
                f"{where_sql} "
                "ORDER BY last_active DESC, s.started_at DESC, "
                "s.id DESC LIMIT ? OFFSET ?",
                params,
            )
        ).fetchall()
        return [self._session_row_dict(row) for row in rows]

    async def search_sessions_by_id(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = True,
        source: str = None,
        sources: List[str] = None,
        exclude_sources: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search surfaced sessions by exact, prefix, or substring id."""
        needle = (query or "").strip().lower()
        if not needle or limit <= 0:
            return []
        candidates = await self.list_sessions_rich(
            source=source,
            sources=sources,
            exclude_sources=exclude_sources,
            limit=max(limit * 4, limit),
            offset=0,
            include_archived=include_archived,
            order_by_last_active=True,
            id_query=needle,
        )

        def score(row: Dict[str, Any]) -> int:
            ids = [
                str(row.get("id") or ""),
                str(row.get("_lineage_root_id") or ""),
            ]
            normalized = [value.lower() for value in ids if value]
            if any(value == needle for value in normalized):
                return 0
            if any(value.startswith(needle) for value in normalized):
                return 1
            return 2

        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (score(item[1]), item[0]),
        )
        return [row for _, row in ranked[:limit]]

    async def session_count(
        self,
        source: str = None,
        sources: List[str] = None,
        cwd_prefix: str = None,
        min_message_count: int = 0,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
        exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions, optionally filtered by source.

        Pass ``exclude_children=True`` to count only the conversations that
        ``list_sessions_rich`` surfaces (root + branch sessions), hiding
        sub-agent runs and compression continuations. Use it whenever the count
        is paired with a ``list_sessions_rich`` page (e.g. sidebar "load more"
        totals) so the total matches the number of listable rows — otherwise the
        raw row count is inflated by children and "load more" never settles.

        Pass ``exclude_sources`` to drop whole source classes from the count
        (e.g. ``["cron"]`` so the recents "load more" total matches a
        cron-excluded ``list_sessions_rich`` page and doesn't keep "load more"
        stuck on for buried scheduler sessions).
        """
        where_clauses = []
        params = []

        if exclude_children:
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(
                f"{_delegate_from_json('s.model_config')} IS NULL"
            )
        include_sources = [source] if source else list(sources or [])
        if include_sources:
            placeholders = ",".join("?" for _ in include_sources)
            where_clauses.append(f"s.source IN ({placeholders})")
            params.extend(include_sources)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where_clauses.append(clause)
            params.extend(clause_params)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = (
            f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        )
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                f"SELECT COUNT(*) FROM sessions s{where_sql}", params
            )
        ).fetchone()
        return row[0]

    async def session_count_by_source(
        self,
        *,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
    ) -> Dict[str, int]:
        """Return a ``{source: count}`` dict via a single ``GROUP BY`` query.

        Replaces the O(N) ``list_sessions_rich`` histogram loop with an
        aggregate query. When ``exclude_children`` is False the query uses
        ``idx_sessions_source``; when True, the child-exclusion predicates
        require a full table scan (same as ``session_count`` and
        ``list_sessions_rich``).

        ``exclude_children=True`` mirrors ``list_sessions_rich`` visibility
        (roots + branch sessions, excluding sub-agent runs, delegates, and
        compression continuations) so the source counts match what the
        Sessions page actually lists.
        """
        where_clauses = []
        params: list = []

        if exclude_children:
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(
                f"{_delegate_from_json('s.model_config')} IS NULL"
            )
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = (
            f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        )
        if self._closed:
            raise RuntimeError("SessionDB connection is closed")
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, "
                "COUNT(*) AS count "
                f"FROM sessions s{where_sql} "
                "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') "
                "ORDER BY count DESC",
                params,
            )
        ).fetchall()
        return {
            str(row["source"]): int(row["count"] or 0) for row in rows
        }

    async def has_platform_message_id(
        self, session_id: str, platform_message_id: str
    ) -> bool:
        """Check if a message with the given platform_message_id exists.

        Uses the idx_messages_platform_msg_id partial index for efficient
        lookup. Used by the gateway's transient-failure dedupe guard (#47237)
        to skip re-persisting a user message that was already saved on a
        prior retry of the same inbound platform message.
        """
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT 1 FROM messages "
                "WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
                (session_id, platform_message_id),
            )
        ).fetchone()
        return row is not None

    _SEARCH_MESSAGE_RESULT_FIELDS = (
        "id",
        "session_id",
        "role",
        "snippet",
        "timestamp",
        "tool_name",
        "source",
        "model",
        "session_started",
        "context",
    )

    @classmethod
    def _search_message_fields(
        cls, fields: Optional[Collection[str]]
    ) -> Optional[Tuple[str, ...]]:
        """Validate and canonically order an optional search projection."""
        if fields is None:
            return None
        if isinstance(fields, str):
            raise TypeError(
                "search fields must be a collection of field names, not a string"
            )
        requested = set(fields)
        unknown = requested.difference(cls._SEARCH_MESSAGE_RESULT_FIELDS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown search result field(s): {names}")
        return tuple(
            field
            for field in cls._SEARCH_MESSAGE_RESULT_FIELDS
            if field in requested
        )

    _SESSION_COMPACT_EXCLUDED = frozenset({"system_prompt", "system_prompt_hash"})
    _session_compact_cols_sql: Optional[str] = None

    @classmethod
    async def _compact_session_cols(cls) -> str:
        """Return the upstream compact projection without blocking SQLite."""
        if cls._session_compact_cols_sql is None:
            declared = (await cls._parse_schema_columns(SCHEMA_SQL))["sessions"]
            cls._session_compact_cols_sql = ", ".join(
                f"s.{name}"
                for name in declared
                if name not in cls._SESSION_COMPACT_EXCLUDED
            )
        return cls._session_compact_cols_sql

    async def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk a compression-continuation chain and return its live tip."""
        connection = await self._get_connection()
        current = session_id
        seen = {current} if current else set()
        for _ in range(100):
            row = await (
                await connection.execute(
                    f"""
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = ?
                      AND parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                    ORDER BY
                      CASE
                        WHEN child.end_reason = 'compression' THEN 0
                        WHEN child.ended_at IS NULL THEN 1
                        ELSE 2
                      END,
                      {_sql_session_last_active("child")} DESC,
                      child.started_at DESC,
                      child.id DESC
                    LIMIT 1
                    """,
                    (current,),
                )
            ).fetchone()
            if row is None:
                return current
            child_id = row["id"]
            if not child_id or child_id in seen:
                return current
            seen.add(child_id)
            current = child_id
        return current

    async def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant session that holds the messages.

        Context compression ends the current session and forks a new child session
        (linked via ``parent_session_id``). The flush cursor is reset, so the
        child is where new messages actually land — the parent ends up with
        ``message_count = 0`` rows unless messages had already been flushed to
        it before compression. See #15000.

        This helper walks ``parent_session_id`` forward from ``session_id`` and
        returns the descendant in the chain that has the **most recent** messages.
        Unlike the original logic, it does NOT short-circuit when the starting
        session already has messages — a descendant that was created by
        compression may hold the continuation content and should be preferred
        by the WebUI and gateway for ``--resume`` and session loading.

        If no descendant (including the starting session) has any messages,
        the original ``session_id`` is returned unchanged.

        The chain is always walked via the child whose ``started_at`` is
        latest; that matches the single-chain shape that compression creates.
        A depth cap (32) guards against accidental loops in malformed data.
        """
        if not session_id:
            return session_id

        # Follow the compression-continuation chain forward to the live tip
        # FIRST. Auto-compression ends the current session and forks a
        # continuation child, but a long-lived parent keeps its own flushed
        # message rows — so the empty-head walk below never redirects it, and
        # resuming the parent id reloads the pre-compression transcript while
        # the turns generated *after* compression (and their responses) sit in
        # the continuation. ``get_compression_tip`` is lineage-aware: it only
        # follows children whose parent ended with ``end_reason='compression'``
        # (created after the parent was ended), so delegation / branch children
        # never hijack the resume. This is the fix for the desktop "I came back
        # and the reply isn't there" report on large sessions.
        try:
            tip = await self.get_compression_tip(session_id)
        except Exception:
            tip = session_id
        if tip and tip != session_id:
            session_id = tip

        connection = await self._get_connection()
        current = session_id
        seen = {current}
        best = None  # tracks the last (deepest) node with messages

        for _ in range(32):
            # Check if the current node has messages.
            try:
                row = await (
                    await connection.execute(
                        "SELECT 1 FROM messages "
                        "WHERE session_id = ? LIMIT 1",
                        (current,),
                    )
                ).fetchone()
            except Exception:
                return session_id
            if row is not None:
                best = current

            # Walk to the most-recently-started child — but skip explicit
            # branch (`_branched_from`), delegate/subagent (`_delegate_from`),
            # and tool children. They also carry a ``parent_session_id`` yet
            # are NOT compression continuations; following them would hijack
            # the resume target to an unrelated session (e.g. a subagent
            # run). This mirrors the child-exclusion in ``get_compression_tip``.
            try:
                child_row = await (
                    await connection.execute(
                        "SELECT id FROM sessions "
                        "WHERE parent_session_id = ? "
                        "  AND json_extract(COALESCE(model_config, '{}'), "
                        "'$._branched_from') IS NULL "
                        "  AND json_extract(COALESCE(model_config, '{}'), "
                        "'$._delegate_from') IS NULL "
                        "  AND COALESCE(source, '') != 'tool' "
                        "ORDER BY started_at DESC, id DESC LIMIT 1",
                        (current,),
                    )
                ).fetchone()
            except Exception:
                return session_id
            if child_row is None:
                break
            child_id = (
                child_row["id"]
                if hasattr(child_row, "keys")
                else child_row[0]
            )
            if not child_id or child_id in seen:
                break
            seen.add(child_id)
            current = child_id

        return best if best is not None else session_id

    def _is_branch_child_row(self, session: Dict[str, Any]) -> bool:
        raw = session.get("model_config")
        if not raw:
            return False
        try:
            cfg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(cfg, dict) and cfg.get("_branched_from") is not None

    async def _is_compression_child_row(
        self, child: Dict[str, Any]
    ) -> bool:
        parent_id = child.get("parent_session_id")
        if not parent_id or self._is_branch_child_row(child):
            return False
        parent = await self.get_session(parent_id)
        return bool(parent and parent.get("end_reason") == "compression")

    async def get_compression_lineage(self, session_id: str) -> List[str]:
        """Return compression ancestors through tip in chronological order."""
        session = await self.get_session(session_id)
        if not session or self._is_branch_child_row(session):
            return [session_id] if session else []

        root = session
        ancestors = {root["id"]}
        while await self._is_compression_child_row(root):
            parent = await self.get_session(root["parent_session_id"])
            if not parent or parent["id"] in ancestors:
                break
            root = parent
            ancestors.add(root["id"])

        lineage = [root["id"]]
        seen = {root["id"]}
        current = root
        connection = await self._get_connection()
        while current.get("end_reason") == "compression":
            rows = await (
                await connection.execute(
                    """
                    SELECT * FROM sessions
                    WHERE parent_session_id = ?
                    ORDER BY started_at ASC
                    """,
                    (current["id"],),
                )
            ).fetchall()
            next_child = None
            for row in rows:
                candidate = dict(row)
                if not self._is_branch_child_row(candidate):
                    next_child = candidate
                    break
            if not next_child or next_child["id"] in seen:
                break
            lineage.append(next_child["id"])
            seen.add(next_child["id"])
            current = next_child
            if current["id"] == session_id:
                # Continue to include later compression tips only when the
                # requested session itself was compacted.
                continue
        return lineage if session_id in lineage else [session_id]

    async def _get_session_rich_rows_batch(
        self, session_ids, compact_rows: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch enriched session rows in bounded native-async batches."""
        ids = [session_id for session_id in session_ids if session_id]
        if not ids:
            return {}
        chunk_size = 900
        if len(ids) > chunk_size:
            result: Dict[str, Dict[str, Any]] = {}
            for start in range(0, len(ids), chunk_size):
                result.update(
                    await self._get_session_rich_rows_batch(
                        ids[start : start + chunk_size],
                        compact_rows=compact_rows,
                    )
                )
            return result

        await self.flush_token_counts()
        select = await self._compact_session_cols() if compact_rows else "s.*"
        placeholders = ",".join("?" for _ in ids)
        prompt_select = (
            ""
            if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            ""
            if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )
        query = f"""
            SELECT {select}{prompt_select},
                COALESCE(
                    (SELECT {_PREVIEW_RAW_SELECT}
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            {prompt_join}
            WHERE s.id IN ({placeholders})
        """
        connection = await self._get_connection()
        rows = await (await connection.execute(query, ids)).fetchall()
        result = {}
        for row in rows:
            session = self._session_row_dict(row)
            session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
            result[session["id"]] = session
        return result

    async def _get_session_rich_row(
        self, session_id: str, compact_rows: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Return one enriched row with the same shape as session listings."""
        return (
            await self._get_session_rich_rows_batch(
                [session_id], compact_rows=compact_rows
            )
        ).get(session_id)

    async def get_session_rich_row(
        self, session_id: str, compact_rows: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Public wrapper for the upstream single-session rich-row API."""
        return await self._get_session_rich_row(
            session_id, compact_rows=compact_rows
        )

    async def list_sessions_rich(
        self,
        source: str = None,
        sources: List[str] = None,
        exclude_sources: List[str] = None,
        cwd_prefix: str = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        min_message_count: int = 0,
        project_compression_tips: bool = True,
        order_by_last_active: bool = False,
        include_archived: bool = False,
        archived_only: bool = False,
        id_query: str = None,
        search_query: str = None,
        compact_rows: bool = False,
        include_pinned: bool = False,
        session_key: str = None,
    ) -> List[Dict[str, Any]]:
        """List enriched sessions with upstream filtering and projection."""
        await self.flush_token_counts()
        where_clauses: list[str] = []
        params: list[Any] = []

        if not include_children:
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(
                f"{_delegate_from_json('s.model_config')} IS NULL"
            )
        include_sources = [source] if source else list(sources or [])
        if include_sources:
            placeholders = ",".join("?" for _ in include_sources)
            where_clauses.append(f"s.source IN ({placeholders})")
            params.extend(include_sources)
        if session_key:
            where_clauses.append("s.session_key = ?")
            params.append(session_key)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where_clauses.append(clause)
            params.extend(clause_params)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = (
            f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        )
        base_where_params = list(params)
        prompt_select = (
            ""
            if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            ""
            if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )
        id_needle = (id_query or "").strip().lower()
        search_needle = (search_query or "").strip().lower()
        select = await self._compact_session_cols() if compact_rows else "s.*"

        if order_by_last_active:
            outer_where = where_sql
            query_params: List[Any] = []
            filter_clauses: List[str] = []

            def _like_pattern(needle: str) -> str:
                escaped = (
                    needle.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                return f"%{escaped}%"

            if id_needle:
                filter_clauses.append(
                    "EXISTS (SELECT 1 FROM chain cq"
                    "        WHERE cq.root_id = s.id"
                    "          AND LOWER(cq.cur_id) LIKE ? ESCAPE '\\')"
                )
                query_params.append(_like_pattern(id_needle))
            if search_needle:
                compact_needle = re.sub(r"[\W_]+", "", search_needle)
                compact_sql = (
                    "REPLACE(REPLACE(REPLACE(REPLACE(LOWER(COALESCE({0}, '')),"
                    " '-', ''), '_', ''), '.', ''), ' ', '')"
                )
                search_clause = (
                    "EXISTS (SELECT 1 FROM chain cq"
                    " JOIN sessions cs ON cs.id = cq.cur_id"
                    " WHERE cq.root_id = s.id"
                    " AND (LOWER(COALESCE(cs.title, '')) LIKE ? ESCAPE '\\'"
                    " OR LOWER(cq.cur_id) LIKE ? ESCAPE '\\'"
                )
                query_params.extend([_like_pattern(search_needle)] * 2)
                if compact_needle:
                    search_clause += (
                        f" OR {compact_sql.format('cs.title')} LIKE ? ESCAPE '\\'"
                    )
                    query_params.append(_like_pattern(compact_needle))
                filter_clauses.append(search_clause + "))")
            if filter_clauses:
                combined = " AND ".join(filter_clauses)
                outer_where = (
                    f"{where_sql} AND {combined}"
                    if where_sql
                    else f"WHERE {combined}"
                )
            query = f"""
                WITH RECURSIVE chain(root_id, cur_id) AS (
                    SELECT s.id, s.id FROM sessions s {where_sql}
                    UNION ALL
                    SELECT c.root_id, child.id
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX({_sql_session_last_active_by_id("cur_id")}) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                )
                SELECT {select}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    {_sql_session_last_active("s")} AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {prompt_join}
                {outer_where}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
                LIMIT ? OFFSET ?
            """
            params = params + params + query_params + [limit, offset]
        else:
            query = f"""
                SELECT {select}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    {_sql_session_last_active("s")} AS last_active
                FROM sessions s
                {prompt_join}
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

        connection = await self._get_connection()
        rows = await (await connection.execute(query, params)).fetchall()
        sessions = []
        for row in rows:
            session = self._session_row_dict(row)
            session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
            session.pop("_effective_last_active", None)
            sessions.append(session)

        if include_pinned:
            seen_ids = {session["id"] for session in sessions}
            pinned_where = (
                f"{where_sql} AND s.pinned = 1"
                if where_sql
                else "WHERE s.pinned = 1"
            )
            pinned_query = f"""
                SELECT {select}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                FROM sessions s
                {prompt_join}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
            pinned_rows = await (
                await connection.execute(pinned_query, base_where_params)
            ).fetchall()
            for row in pinned_rows:
                session = self._session_row_dict(row)
                if session["id"] in seen_ids:
                    continue
                session["preview"] = _shape_preview(
                    session.pop("_preview_raw", "")
                )
                seen_ids.add(session["id"])
                sessions.append(session)

        if project_compression_tips and not include_children:
            tip_ids_by_root: Dict[str, str] = {}
            for session in sessions:
                if session.get("end_reason") != "compression":
                    continue
                tip_id = await self.get_compression_tip(session["id"])
                if tip_id != session["id"]:
                    tip_ids_by_root[session["id"]] = tip_id

            tip_rows = (
                await self._get_session_rich_rows_batch(
                    set(tip_ids_by_root.values()), compact_rows=compact_rows
                )
                if tip_ids_by_root
                else {}
            )
            projected = []
            for session in sessions:
                tip_id = tip_ids_by_root.get(session["id"])
                tip_row = tip_rows.get(tip_id) if tip_id else None
                if not tip_row:
                    projected.append(session)
                    continue
                merged = dict(session)
                for key in (
                    "id",
                    "ended_at",
                    "end_reason",
                    "message_count",
                    "tool_call_count",
                    "title",
                    "last_active",
                    "preview",
                    "model",
                    "system_prompt",
                    "cwd",
                    "git_branch",
                    "git_repo_root",
                ):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged["_lineage_root_id"] = session["id"]
                projected.append(merged)
            sessions = projected

        return sessions

    async def get_compression_failure_cooldown(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_failure_cooldown_until, "
                "compression_failure_error FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        if row is None or row["compression_failure_cooldown_until"] is None:
            return None
        cooldown_until = float(row["compression_failure_cooldown_until"])
        remaining_seconds = cooldown_until - time.time()
        if remaining_seconds <= 0:
            return None
        return {
            "cooldown_until": cooldown_until,
            "remaining_seconds": remaining_seconds,
            "error": row["compression_failure_error"],
        }

    async def get_compression_failure_cooldown_row(
        self, session_id: str
    ) -> Dict[str, Any]:
        """Return the exact stored cooldown columns without expiry filtering."""
        if not session_id:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_failure_cooldown_until, "
                "compression_failure_error FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        if row is None:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        deadline = row["compression_failure_cooldown_until"]
        return {
            "session_exists": True,
            "cooldown_until": float(deadline) if deadline is not None else None,
            "error": row["compression_failure_error"],
        }

    async def restore_compression_failure_cooldown_row(
        self, session_id: str, snapshot: Dict[str, Any]
    ) -> None:
        """Restore and verify an exact cooldown-row snapshot."""
        expected_exists = bool(snapshot.get("session_exists", False))
        if not expected_exists:
            actual = await self.get_compression_failure_cooldown_row(session_id)
            if actual.get("session_exists", False):
                raise RuntimeError(
                    "cannot restore absent compression cooldown row: session now exists"
                )
            return

        deadline = snapshot.get("cooldown_until")
        error = snapshot.get("error")

        async def _restore(connection):
            cursor = await connection.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (deadline, error, session_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"compression cooldown rollback session missing: {session_id}"
                )

        await self._write(_restore)
        actual = await self.get_compression_failure_cooldown_row(session_id)
        expected = {
            "session_exists": True,
            "cooldown_until": float(deadline) if deadline is not None else None,
            "error": error,
        }
        if actual != expected:
            raise RuntimeError(
                "compression cooldown rollback verification failed: "
                f"expected={expected!r}, actual={actual!r}"
            )

    async def record_compression_failure_cooldown(
        self, session_id: str, cooldown_until: float, error: Optional[str] = None
    ) -> None:
        if not session_id:
            return

        async def _record(connection):
            await connection.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (cooldown_until, error, session_id),
            )

        await self._write(_record)

    async def clear_compression_failure_cooldown(self, session_id: str) -> None:
        if not session_id:
            return

        async def _clear(connection):
            await connection.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
                "compression_failure_error = NULL WHERE id = ?",
                (session_id,),
            )

        await self._write(_clear)

    async def get_compression_fallback_streak(self, session_id: str) -> int:
        if not session_id:
            return 0
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_fallback_streak FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        try:
            return max(0, int(row["compression_fallback_streak"] or 0)) if row else 0
        except (TypeError, ValueError):
            return 0

    async def set_compression_fallback_streak(
        self, session_id: str, streak: int
    ) -> None:
        if not session_id:
            return

        async def _set(connection):
            await connection.execute(
                "UPDATE sessions SET compression_fallback_streak = ? WHERE id = ?",
                (max(0, int(streak)), session_id),
            )

        await self._write(_set)

    async def get_compression_ineffective_count(self, session_id: str) -> int:
        if not session_id:
            return 0
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT compression_ineffective_count FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
        try:
            return max(0, int(row["compression_ineffective_count"] or 0)) if row else 0
        except (TypeError, ValueError):
            return 0

    async def set_compression_ineffective_count(
        self, session_id: str, count: int
    ) -> None:
        if not session_id:
            return

        async def _set(connection):
            await connection.execute(
                "UPDATE sessions SET compression_ineffective_count = ? WHERE id = ?",
                (max(0, int(count)), session_id),
            )

        await self._write(_set)

    async def get_conversation_root(self, session_id: str) -> str:
        """Resolve a session lineage root without using ``SessionDB``'s lock."""
        if not session_id:
            return session_id
        connection = await self._get_connection()
        current = session_id
        seen = set()
        for _ in range(100):
            if not current or current in seen:
                break
            seen.add(current)
            row = await (
                await connection.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                )
            ).fetchone()
            if row is None or not row["parent_session_id"]:
                break
            current = row["parent_session_id"]
        return current or session_id

    async def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
        repair_alternation: bool = False,
        include_row_ids: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load a conversation through the native async SQLite connection."""
        if not session_id:
            return []
        connection = await self._get_connection()
        session_ids = [session_id]
        if include_ancestors:
            current = session_id
            seen = set()
            while current and current not in seen:
                seen.add(current)
                row = await (
                    await connection.execute(
                        "SELECT parent_session_id FROM sessions WHERE id = ?",
                        (current,),
                    )
                ).fetchone()
                parent = row["parent_session_id"] if row is not None else None
                if not parent:
                    break
                session_ids.insert(0, parent)
                current = parent

        placeholders = ",".join("?" for _ in session_ids)
        active_clause = "" if include_inactive else " AND active = 1"
        rows = await (
            await connection.execute(
                f"SELECT {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders})"
                f"{active_clause} ORDER BY id",
                tuple(session_ids),
            )
        ).fetchall()
        return self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=include_ancestors,
            repair_alternation=repair_alternation,
            include_row_ids=include_row_ids,
        )

    async def get_resume_conversations(
        self, session_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return ``(model_history, display_history)`` for a session resume in ONE SELECT.

        ``session.resume`` needs two projections of the same lineage:

        - ``model_history`` — the tip session's active rows, alternation-repaired
          (the live-replay working conversation). Equivalent to
          ``get_messages_as_conversation(session_id, repair_alternation=True)``.
        - ``display_history`` — the full lineage (ancestors → tip), verbatim, with
          replayed-user dedup. Equivalent to
          ``get_messages_as_conversation(session_id, include_ancestors=True)``.

        The display fetch already reads a superset of the model fetch (the tip
        rows are part of the lineage), so serving both from one lineage SELECT
        halves the resume's DB work versus two separate calls, with byte-identical
        output (see test_get_resume_conversations_matches_separate_reads).
        """
        session_ids = await self._session_lineage_root_to_tip(session_id)
        connection = await self._get_connection()
        placeholders = ",".join("?" for _ in session_ids)
        rows = await (
            await connection.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) "
                "AND active = 1 "
                # ORDER BY id (insertion order) — see get_messages_as_conversation
                # for why timestamp ordering is unsafe.
                "ORDER BY id",
                tuple(session_ids),
            )
        ).fetchall()

        # Tip rows are exactly the model-fed set (get_messages_as_conversation
        # with session_ids=[session_id]); filtering the lineage fetch preserves
        # their relative id order.
        tip_rows = [r for r in rows if r["session_id"] == session_id]
        model_history = self._rows_to_conversation(
            tip_rows,
            session_id=session_id,
            include_ancestors=False,
            repair_alternation=True,
            include_row_ids=True,
        )
        display_history = self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
            include_row_ids=True,
        )
        return model_history, display_history

    async def get_ancestor_display_prefix(
        self, session_id: str
    ) -> List[Dict[str, Any]]:
        """Return the ancestor-only display messages for a session lineage.

        These are messages from parent/grandparent sessions (compression
        ancestors) that appear in the display transcript but NOT in the
        tip session's model-fed history. Used by ``session.resume`` to
        build the ``display_history_prefix`` that ``_live_session_payload``
        prepends to the live model history.

        Previously the prefix was calculated as
        ``display_history[:len(display) - len(raw)]``, but that overcounts
        when ``repair_message_sequence`` removes messages from the MIDDLE
        of the tip history (e.g. verification candidates collapsed by the
        consecutive-assistant merge) — the length difference includes both
        ancestor messages AND repair-removed tip messages, but the slice
        only captures the first N display messages (which are tip messages
        when there are no ancestors), causing duplication. This method
        returns ONLY the genuine ancestor messages, identified by
        ``session_id != tip_session_id``. (#65919)
        """
        session_ids = await self._session_lineage_root_to_tip(session_id)
        if len(session_ids) <= 1:
            return []
        connection = await self._get_connection()
        placeholders = ",".join("?" for _ in session_ids)
        rows = await (
            await connection.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) "
                "AND active = 1 ORDER BY id",
                tuple(session_ids),
            )
        ).fetchall()
        ancestor_rows = [r for r in rows if r["session_id"] != session_id]
        if not ancestor_rows:
            return []
        return self._rows_to_conversation(
            ancestor_rows,
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
        )

    async def _session_lineage_root_to_tip(
        self, session_id: str
    ) -> List[str]:
        if not session_id:
            return [session_id]

        chain = []
        current = session_id
        seen = set()
        connection = await self._get_connection()
        for _ in range(100):
            if not current or current in seen:
                break
            seen.add(current)
            chain.append(current)
            row = await (
                await connection.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                )
            ).fetchone()
            if row is None:
                break
            current = (
                row["parent_session_id"]
                if hasattr(row, "keys")
                else row[0]
            )
        return list(reversed(chain)) or [session_id]

    async def find_live_compression_child(
        self, parent_session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the one unambiguous live compression child, if it exists."""
        if not parent_session_id:
            return None
        connection = await self._get_connection()
        parent = await (
            await connection.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (parent_session_id,),
            )
        ).fetchone()
        if (
            parent is None
            or parent["ended_at"] is None
            or parent["end_reason"] != "compression"
        ):
            return None
        rows = await (
            await connection.execute(
                """SELECT s.*, COALESCE(sp.prompt, s.system_prompt)
                          AS _system_prompt_resolved
                   FROM sessions s LEFT JOIN system_prompts sp
                     ON sp.hash = s.system_prompt_hash
                   WHERE s.parent_session_id = ?
                     AND s.ended_at IS NULL
                     AND json_extract(COALESCE(s.model_config, '{}'), '$._branched_from') IS NULL
                     AND json_extract(COALESCE(s.model_config, '{}'), '$._delegate_from') IS NULL
                     AND COALESCE(s.source, '') != 'tool'
                   ORDER BY s.started_at ASC
                   LIMIT 2""",
                (parent_session_id,),
            )
        ).fetchall()
        return self._session_row_dict(rows[0]) if len(rows) == 1 else None

    async def get_session_title(self, session_id: str) -> Optional[str]:
        """Read a title without using the synchronous connection."""
        connection = await self._get_connection()
        row = await (
            await connection.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
        ).fetchone()
        return row["title"] if row is not None else None

    async def get_next_title_in_lineage(self, base_title: str) -> str:
        """Return the next generated continuation title."""
        match = re.match(r"^(.*?) #(\d+)$", base_title)
        base = match.group(1) if match else base_title
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        connection = await self._get_connection()
        rows = await (
            await connection.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
        ).fetchall()
        titles = [row["title"] for row in rows]
        if not titles:
            return base
        suffixes = [
            int(found.group(1))
            for title in titles
            if (found := re.match(rf"^{re.escape(base)} #(\d+)$", title or ""))
        ]
        return f"{base} #{max(suffixes, default=1) + 1}"

    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Lone surrogates cannot be bound by sqlite3 (UnicodeEncodeError at
        # UTF-8 encode time) — scrub them like every other write path here.
        title = _sanitize_surrogates(title)

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars so the whitespace collapsing step normalizes them.
        cleaned = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", title
        )

        # Remove zero-width, directional, object-replacement, and interlinear
        # annotation control characters.
        cleaned = re.sub(
            r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]",
            "",
            cleaned,
        )

        # Collapse internal whitespace runs and strip.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, "
                f"max {SessionDB.MAX_TITLE_LENGTH})"
            )
        return cleaned

    async def _is_compression_ancestor(
        self, conn, *, ancestor_id: str, descendant_id: str
    ) -> bool:
        """Return whether *ancestor_id* is a compression predecessor."""
        if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
            return False
        edge = _COMPRESSION_CHILD_SQL.format(a="child")
        row = await (
            await conn.execute(
                f"""
                WITH RECURSIVE ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT parent.id
                    FROM ancestors a
                    JOIN sessions child ON child.id = a.id
                    JOIN sessions parent ON parent.id = child.parent_session_id
                    WHERE {edge}
                )
                SELECT 1 FROM ancestors WHERE id = ? AND id != ? LIMIT 1
                """,
                (descendant_id, ancestor_id, descendant_id),
            )
        ).fetchone()
        return row is not None

    async def _set_session_title(
        self,
        session_id: str,
        title: str,
        *,
        only_if_empty: bool,
    ) -> bool:
        title = self.sanitize_title(title)

        async def _do(conn):
            if only_if_empty:
                current = await (
                    await conn.execute(
                        "SELECT title FROM sessions WHERE id = ?",
                        (session_id,),
                    )
                ).fetchone()
                if current is None or current["title"] is not None:
                    return 0

            if title:
                cursor = await conn.execute(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    (title, session_id),
                )
                conflict = await cursor.fetchone()
                if conflict:
                    conflict_id = conflict["id"]
                    if await self._is_compression_ancestor(
                        conn,
                        ancestor_id=conflict_id,
                        descendant_id=session_id,
                    ):
                        await conn.execute(
                            "UPDATE sessions SET title = NULL WHERE id = ?",
                            (conflict_id,),
                        )
                    else:
                        raise ValueError(
                            f"Title '{title}' is already in use by session "
                            f"{conflict_id}"
                        )

            predicate = " AND title IS NULL" if only_if_empty else ""
            cursor = await conn.execute(
                f"UPDATE sessions SET title = ? WHERE id = ?{predicate}",
                (title, session_id),
            )
            return cursor.rowcount

        rowcount = await self._write(_do)
        return rowcount > 0

    async def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title."""
        return await self._set_session_title(
            session_id, title, only_if_empty=False
        )

    async def set_auto_title_if_empty(
        self, session_id: str, title: str
    ) -> bool:
        """Set an auto-generated title only when the current title is null."""
        return await self._set_session_title(
            session_id, title, only_if_empty=True
        )

    async def set_session_archived(
        self, session_id: str, archived: bool
    ) -> bool:
        """Archive or unarchive a session.

        Archived sessions are hidden from the default session list but keep all
        their messages. Compression chains are updated as one conversation.
        Returns True when at least one row was updated.
        """
        async def _do(conn):
            cursor = await conn.execute(
                """
                WITH RECURSIVE
                  ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT parent.id
                    FROM ancestors a
                    JOIN sessions child ON child.id = a.id
                    JOIN sessions parent ON parent.id = child.parent_session_id
                    WHERE parent.end_reason = 'compression'
                  ),
                  descendants(id) AS (
                    SELECT ?
                    UNION
                    SELECT child.id
                    FROM descendants d
                    JOIN sessions parent ON parent.id = d.id
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.end_reason = 'compression'
                  ),
                  lineage(id) AS (
                    SELECT id FROM ancestors
                    UNION
                    SELECT id FROM descendants
                  )
                UPDATE sessions
                SET archived = ?
                WHERE id IN (SELECT id FROM lineage)
                """,
                (session_id, session_id, 1 if archived else 0),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = (
                    await (await conn.execute("SELECT changes()")).fetchone()
                )[0]
            return rowcount

        rowcount = await self._write(_do)
        return rowcount > 0

    async def set_session_pinned(
        self, session_id: str, pinned: bool
    ) -> bool:
        """Pin or unpin a session and its whole compression lineage.

        Pinned sessions are exempt from stale-session archival. Returns True
        when at least one row was updated.
        """
        async def _do(conn):
            cursor = await conn.execute(
                """
                WITH RECURSIVE
                  ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT parent.id
                    FROM ancestors a
                    JOIN sessions child ON child.id = a.id
                    JOIN sessions parent ON parent.id = child.parent_session_id
                    WHERE parent.end_reason = 'compression'
                  ),
                  descendants(id) AS (
                    SELECT ?
                    UNION
                    SELECT child.id
                    FROM descendants d
                    JOIN sessions parent ON parent.id = d.id
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.end_reason = 'compression'
                  ),
                  lineage(id) AS (
                    SELECT id FROM ancestors
                    UNION
                    SELECT id FROM descendants
                  )
                UPDATE sessions
                SET pinned = ?
                WHERE id IN (SELECT id FROM lineage)
                """,
                (session_id, session_id, 1 if pinned else 0),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = (
                    await (await conn.execute("SELECT changes()")).fetchone()
                )[0]
            return rowcount

        rowcount = await self._write(_do)
        return rowcount > 0

    async def publish_compression_child(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        source: str,
        messages: List[Dict[str, Any]],
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        cwd: str = None,
        profile_name: str = None,
        compression_lock_holder: str = None,
        require_compression_lease: bool = True,
    ) -> None:
        """Publish a compressed continuation without a sync SQLite bridge."""
        if not messages:
            raise RuntimeError("Compression child handoff must not be empty")

        async def _publish(connection):
            lock_row = await (
                await connection.execute(
                    "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                    (parent_session_id,),
                )
            ).fetchone()
            if require_compression_lease and (
                lock_row is None
                or not compression_lock_holder
                or lock_row["holder"] != compression_lock_holder
                or float(lock_row["expires_at"]) <= time.time()
            ):
                raise CompressionSessionBusyError(
                    f"Compression lease lost before publication: {parent_session_id}"
                )
            parent = await (
                await connection.execute(
                    """SELECT ended_at, cwd, git_branch, git_repo_root,
                              user_id, session_key, chat_id, chat_type,
                              thread_id, display_name, origin_json, profile_name
                       FROM sessions WHERE id = ?""",
                    (parent_session_id,),
                )
            ).fetchone()
            if parent is None:
                raise RuntimeError(f"Compression parent not found: {parent_session_id}")
            if parent["ended_at"] is not None:
                raise RuntimeError(
                    f"Compression parent already ended: {parent_session_id}"
                )
            system_prompt_hash = await self._store_system_prompt(
                connection, system_prompt
            )
            await connection.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt,
                   system_prompt_hash,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, started_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    child_session_id,
                    source,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    cwd or parent["cwd"],
                    parent["git_branch"],
                    parent["git_repo_root"],
                    profile_name or parent["profile_name"],
                    parent["user_id"],
                    parent["session_key"],
                    parent["chat_id"],
                    parent["chat_type"],
                    parent["thread_id"],
                    parent["display_name"],
                    parent["origin_json"],
                    time.time(),
                ),
            )
            now_ts = time.time()
            tool_calls_total = 0
            for message in messages:
                role = message.get("role", "unknown")
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, str):
                    try:
                        tool_calls = json.loads(tool_calls)
                    except (json.JSONDecodeError, TypeError):
                        tool_calls = []
                timestamp = message.get("timestamp", now_ts)
                try:
                    timestamp = float(
                        timestamp.timestamp()
                        if hasattr(timestamp, "timestamp")
                        else timestamp
                    )
                except (TypeError, ValueError):
                    timestamp = now_ts
                reasoning_details = (
                    message.get("reasoning_details") if role == "assistant" else None
                )
                codex_reasoning_items = (
                    message.get("codex_reasoning_items") if role == "assistant" else None
                )
                codex_message_items = (
                    message.get("codex_message_items") if role == "assistant" else None
                )
                await connection.execute(
                    """INSERT INTO messages (session_id, role, content, tool_call_id,
                       tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        child_session_id, role,
                        self._encode_content(message.get("content")),
                        message.get("tool_call_id"),
                        json.dumps(tool_calls) if tool_calls else None,
                        _scrub_surrogates(message.get("tool_name")),
                        message.get("effect_disposition"), timestamp,
                        message.get("token_count"), message.get("finish_reason"),
                        _scrub_surrogates(message.get("reasoning")) if role == "assistant" else None,
                        _scrub_surrogates(message.get("reasoning_content")) if role == "assistant" else None,
                        json.dumps(reasoning_details) if reasoning_details else None,
                        json.dumps(codex_reasoning_items) if codex_reasoning_items else None,
                        json.dumps(codex_message_items) if codex_message_items else None,
                        message.get("platform_message_id") or message.get("message_id"),
                        1 if message.get("observed") else 0, 1,
                        _scrub_surrogates(message.get("api_content"))
                        if isinstance(message.get("api_content"), str) else None,
                        _scrub_surrogates(message.get("display_kind"))
                        if isinstance(message.get("display_kind"), str) else None,
                        self._encode_display_metadata(message.get("display_metadata")),
                    ),
                )
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else int(tool_calls is not None)
                )
                now_ts = max(now_ts + 1e-6, timestamp + 1e-6)
            await connection.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (len(messages), tool_calls_total, child_session_id),
            )
            updated = await connection.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'compression' "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), parent_session_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    f"Compression parent changed during publication: {parent_session_id}"
                )

        await self._write(_publish)

    async def logical_size_bytes(self) -> Optional[int]:
        """Return SQLite's logical database size in bytes."""
        try:
            async with self._get_write_lock():
                connection = await self._get_connection()
                page_count_cursor = await connection.execute("PRAGMA page_count")
                try:
                    page_count_row = await page_count_cursor.fetchone()
                finally:
                    await page_count_cursor.close()
                page_size_cursor = await connection.execute("PRAGMA page_size")
                try:
                    page_size_row = await page_size_cursor.fetchone()
                finally:
                    await page_size_cursor.close()
            if page_count_row is None or page_size_row is None:
                return None
            return int(page_count_row[0]) * int(page_size_row[0])
        except Exception as exc:
            logger.debug("Could not read logical DB size: %s", exc)
            return None

    async def _fts_table_exists(self, name: str) -> bool:
        """Return whether an FTS5 virtual table is queryable."""
        connection = await self._get_connection()
        try:
            cursor = await connection.execute(f"SELECT 1 FROM {name} LIMIT 0")
            await cursor.close()
            return True
        except sqlite3.DatabaseError:
            return False

    async def optimize_fts(self) -> int:
        """Merge fragmented FTS5 b-tree segments into one per index."""
        optimized = 0
        async with self._get_write_lock():
            connection = await self._get_connection()
            for table_name in self._FTS_TABLES:
                if not await self._fts_table_exists(table_name):
                    continue
                try:
                    cursor = await connection.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('optimize')"
                    )
                    await cursor.close()
                    optimized += 1
                except sqlite3.OperationalError as exc:
                    logger.warning(
                        "FTS optimize failed for %s: %s", table_name, exc
                    )
        return optimized

    async def rebuild_fts(self) -> int:
        """Rebuild present FTS5 indexes from their canonical content tables."""
        rebuilt = 0
        async with self._get_write_lock():
            connection = await self._get_connection()
            for table_name in self._FTS_TABLES:
                if not await self._fts_table_exists(table_name):
                    continue
                try:
                    cursor = await connection.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                    )
                    await cursor.close()
                    await connection.commit()
                    rebuilt += 1
                except sqlite3.OperationalError as exc:
                    await connection.rollback()
                    logger.warning("FTS rebuild failed for %s: %s", table_name, exc)
        return rebuilt

    async def vacuum(self) -> int:
        """Run FTS optimization and VACUUM to reclaim database pages."""
        optimized = 0
        try:
            optimized = await self.optimize_fts()
        except Exception as exc:
            logger.warning("FTS optimize before VACUUM failed: %s", exc)
        async with self._get_write_lock():
            connection = await self._get_connection()
            try:
                checkpoint = await connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                )
                try:
                    await checkpoint.fetchall()
                finally:
                    await checkpoint.close()
            except Exception as exc:
                logger.debug(
                    "WAL checkpoint (TRUNCATE) before VACUUM failed: %s", exc
                )
            vacuum_cursor = await connection.execute("VACUUM")
            await vacuum_cursor.close()
        return optimized

    async def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
        sessions_dir: Optional[Path] = None,
        min_vacuum_interval_days: int = 30,
    ) -> Dict[str, Any]:
        """Run throttled session pruning and optional VACUUM maintenance."""
        result: Dict[str, Any] = {
            "skipped": False,
            "pruned": 0,
            "vacuumed": False,
        }
        try:
            last_raw = await self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3_600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass

            pruned = await self.prune_sessions(
                older_than_days=retention_days,
                sessions_dir=sessions_dir,
            )
            result["pruned"] = pruned

            last_vacuum_raw = await self.get_meta("last_vacuum")
            vacuum_due = True
            if last_vacuum_raw:
                try:
                    vacuum_due = (
                        now - float(last_vacuum_raw)
                    ) >= min_vacuum_interval_days * 86_400
                except (TypeError, ValueError):
                    vacuum_due = True
            if vacuum and pruned > 0 and vacuum_due:
                try:
                    await self.vacuum()
                    result["vacuumed"] = True
                    await self.set_meta("last_vacuum", str(now))
                except Exception as exc:
                    logger.warning("state.db VACUUM failed: %s", exc)

            await self.set_meta("last_auto_prune", str(now))
            if pruned > 0:
                logger.info(
                    "state.db auto-maintenance: pruned %d session(s) inactive "
                    "for %d days%s",
                    pruned,
                    retention_days,
                    " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)
        return result

    async def maybe_auto_archive(
        self,
        idle_days: float = 3,
        min_interval_hours: int = 24,
        exclude_pinned: bool = True,
    ) -> Dict[str, Any]:
        """Run throttled soft archival for sessions idle beyond the cutoff."""
        result: Dict[str, Any] = {"skipped": False, "archived": 0}
        try:
            last_raw = await self.get_meta("last_auto_archive")
            now = time.time()
            if last_raw:
                try:
                    if now - float(last_raw) < min_interval_hours * 3_600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass

            archived = await self.archive_stale_sessions(
                idle_days, exclude_pinned=exclude_pinned
            )
            result["archived"] = archived
            await self.set_meta("last_auto_archive", str(now))
            if archived > 0:
                logger.info(
                    "state.db auto-archive: archived %d session(s) idle >= %s days",
                    archived,
                    idle_days,
                )
        except Exception as exc:
            logger.warning("state.db auto-archive failed: %s", exc)
            result["error"] = str(exc)
        return result

    async def flush_token_counts(self, timeout: float = 5.0) -> bool:
        """Token deltas still use their own async accounting path; no turn-blocking drain."""
        del timeout
        return True

    async def close(self) -> None:
        self._closed = True
        connection = self._connection
        self._connection = None
        tracking_key = self._connection_tracking_key
        self._connection_tracking_key = None
        if connection is None:
            return
        close_task = asyncio.create_task(connection.close())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await asyncio.shield(close_task)
            raise
        finally:
            if tracking_key is not None:
                remaining = _live_connection_counts.get(tracking_key, 1) - 1
                if remaining > 0:
                    _live_connection_counts[tracking_key] = remaining
                else:
                    _live_connection_counts.pop(tracking_key, None)
