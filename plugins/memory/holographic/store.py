"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import asyncio
import os
import re
import weakref
from pathlib import Path

import aiofiles.os
import aiosqlite

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)


async def _finish_owned_task(task: asyncio.Task):
    """Finish one accepted store operation before propagating cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # --- Event-loop-local shared connection registry ---------------------
    # SQLite permits only one writer at a time. Each MemoryStore instance used
    # to open its own connection guarded by its own RLock, so the several
    # providers that coexist on one event loop (the main agent plus every
    # delegate_task subagent) raced as independent WAL writers. Combined with
    # writes that were not rolled back on error, one connection could leave an
    # open write transaction that pinned the write lock and made every other
    # connection's write fail with "database is locked" for the full busy
    # timeout. All same-loop instances for the same database now share ONE
    # connection and ONE async lock, so access is fully serialized and
    # cross-connection contention is impossible. The shared connection is
    # refcounted, so closing
    # one instance never tears the connection out from under a live sibling.
    _shared: dict = {}
    _shared_guards: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, weakref.ReferenceType[asyncio.Lock]]" = (
        weakref.WeakKeyDictionary()
    )

    @classmethod
    def _shared_guard_for_loop(
        cls,
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Lock:
        guard_ref = cls._shared_guards.get(loop)
        guard = guard_ref() if guard_ref is not None else None
        if guard is None:
            guard = asyncio.Lock()
            cls._shared_guards[loop] = weakref.ref(guard)
        return guard

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY
        self._key = ""
        self._registry_key = None
        self._entry = None
        self._conn: aiosqlite.Connection | None = None
        self._lock: asyncio.Lock | None = None
        self._initialize_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def _initialize(self) -> None:
        """Open the loop-local shared connection and initialize its schema."""
        if self._entry is not None:
            return
        if self._closed:
            raise RuntimeError("MemoryStore is closed")

        async with self._initialize_lock:
            if self._entry is not None:
                return
            if self._closed:
                raise RuntimeError("MemoryStore is closed")
            await aiofiles.os.makedirs(self.db_path.parent, exist_ok=True)
            self._key = await aiofiles.os.wrap(os.path.realpath)(self.db_path)
            loop = asyncio.get_running_loop()
            registry_key = (loop, self._key)
            guard = MemoryStore._shared_guard_for_loop(loop)

            async with guard:
                entry = MemoryStore._shared.get(registry_key)
                created = entry is None
                if created:
                    conn = await aiosqlite.connect(
                        self._key,
                        timeout=10.0,
                        # Autocommit: every statement is its own transaction, so a
                        # write that raises mid-method can never leave a dangling
                        # transaction (and its write lock) open. The explicit
                        # commit() calls below become harmless no-ops.
                        isolation_level=None,
                    )
                    conn.row_factory = aiosqlite.Row
                    entry = {
                        "conn": conn,
                        "lock": asyncio.Lock(),
                        "refs": 0,
                        "ready": False,
                        "fact_count": 0,
                    }
                    MemoryStore._shared[registry_key] = entry
                entry["refs"] += 1
                self._registry_key = registry_key
                self._entry = entry
                self._conn = entry["conn"]
                self._lock = entry["lock"]

                try:
                    if not entry["ready"]:
                        async with entry["lock"]:
                            await self._init_db()
                            entry["ready"] = True
                except BaseException:
                    entry["refs"] -= 1
                    self._entry = None
                    self._conn = None
                    self._lock = None
                    self._registry_key = None
                    if created:
                        MemoryStore._shared.pop(registry_key, None)
                        from hermes_state import _close_owned_connection

                        await _close_owned_connection(entry["conn"])
                    raise

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        await apply_wal_with_fallback(
            self._conn,
            db_label="memory_store.db (holographic)",
        )
        await self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {
            row[1]
            for row in await self._conn.execute_fetchall("PRAGMA table_info(facts)")
        }
        if "hrr_vector" not in columns:
            await self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        count_row = await self._fetchone(
            self._conn,
            "SELECT COUNT(*) FROM facts",
        )
        self._entry["fact_count"] = int(count_row[0])
        await self._conn.commit()

    async def _ready(self) -> tuple[aiosqlite.Connection, asyncio.Lock]:
        await self._initialize()
        if self._conn is None or self._lock is None:
            raise RuntimeError("MemoryStore is not initialized")
        return self._conn, self._lock

    @staticmethod
    async def _fetchone(
        conn: aiosqlite.Connection,
        query: str,
        parameters=(),
    ):
        cursor = await conn.execute(query, parameters)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        task = asyncio.create_task(
            self._add_fact(content, category=category, tags=tags),
            name="holographic-add-fact",
        )
        return await _finish_owned_task(task)

    async def _add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        conn, lock = await self._ready()
        async with lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = await conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust),
                )
                await conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
                await cur.close()
                self._entry["fact_count"] += 1
            except aiosqlite.IntegrityError:
                # Duplicate content — return existing id
                row = await self._fetchone(
                    conn,
                    "SELECT fact_id FROM facts WHERE content = ?",
                    (content,),
                )
                return int(row["fact_id"])

            # Entity extraction and linking
            for name in self._extract_entities(content):
                entity_id = await self._resolve_entity(conn, name)
                await self._link_fact_entity(conn, fact_id, entity_id)

            # Compute HRR vector after entity linking
            await self._compute_hrr_vector(conn, fact_id, content)
            await self._rebuild_bank(conn, category)

            return fact_id

    async def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        task = asyncio.create_task(
            self._search_facts(
                query,
                category=category,
                min_trust=min_trust,
                limit=limit,
            ),
            name="holographic-search-facts",
        )
        return await _finish_owned_task(task)

    async def _search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        conn, lock = await self._ready()
        async with lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from plugins.memory.holographic.retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = await conn.execute_fetchall(sql, params)
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                await conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                await conn.commit()

            return results

    async def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        task = asyncio.create_task(
            self._update_fact(
                fact_id,
                content=content,
                trust_delta=trust_delta,
                tags=tags,
                category=category,
            ),
            name="holographic-update-fact",
        )
        return await _finish_owned_task(task)

    async def _update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        conn, lock = await self._ready()
        async with lock:
            row = await self._fetchone(
                conn,
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?",
                (fact_id,),
            )
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            await conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            await conn.commit()

            # If content changed, re-extract entities
            if content is not None:
                await conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = await self._resolve_entity(conn, name)
                    await self._link_fact_entity(conn, fact_id, entity_id)
                await conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                await self._compute_hrr_vector(conn, fact_id, content)
            # Rebuild bank for relevant category
            category_row = await self._fetchone(
                conn,
                "SELECT category FROM facts WHERE fact_id = ?",
                (fact_id,),
            )
            cat = category or category_row["category"]
            await self._rebuild_bank(conn, cat)

            return True

    async def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        task = asyncio.create_task(
            self._remove_fact(fact_id),
            name="holographic-remove-fact",
        )
        return await _finish_owned_task(task)

    async def _remove_fact(self, fact_id: int) -> bool:
        conn, lock = await self._ready()
        async with lock:
            row = await self._fetchone(
                conn,
                "SELECT fact_id, category FROM facts WHERE fact_id = ?",
                (fact_id,),
            )
            if row is None:
                return False

            await conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            await conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            await conn.commit()
            self._entry["fact_count"] -= 1
            await self._rebuild_bank(conn, row["category"])
            return True

    async def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        conn, lock = await self._ready()
        async with lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = await conn.execute_fetchall(sql, params)
            return [self._row_to_dict(r) for r in rows]

    def _current_fact_count(self) -> int:
        """Return the shared fact count without performing synchronous I/O."""
        if self._entry is None:
            return 0
        return int(self._entry["fact_count"])

    async def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        task = asyncio.create_task(
            self._record_feedback(fact_id, helpful),
            name="holographic-record-feedback",
        )
        return await _finish_owned_task(task)

    async def _record_feedback(self, fact_id: int, helpful: bool) -> dict:
        conn, lock = await self._ready()
        async with lock:
            row = await self._fetchone(
                conn,
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            )
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            await conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            await conn.commit()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if stripped and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    async def _resolve_entity(
        self,
        conn: aiosqlite.Connection,
        name: str,
    ) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = await self._fetchone(
            conn,
            "SELECT entity_id FROM entities WHERE name LIKE ?",
            (name,),
        )
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = await self._fetchone(
            conn,
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        )
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = await conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        try:
            await conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]
        finally:
            await cur.close()

    async def _link_fact_entity(
        self,
        conn: aiosqlite.Connection,
        fact_id: int,
        entity_id: int,
    ) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        await conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        await conn.commit()

    async def _compute_hrr_vector(
        self,
        conn: aiosqlite.Connection,
        fact_id: int,
        content: str,
    ) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        if not self._hrr_available:
            return

        # Get entities linked to this fact
        rows = await conn.execute_fetchall(
            """
            SELECT e.name FROM entities e
            JOIN fact_entities fe ON fe.entity_id = e.entity_id
            WHERE fe.fact_id = ?
            """,
            (fact_id,),
        )
        entities = [row["name"] for row in rows]

        vector = hrr.encode_fact(content, entities, self.hrr_dim)
        await conn.execute(
            "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
            (hrr.phases_to_bytes(vector), fact_id),
        )
        await conn.commit()

    async def _rebuild_bank(
        self,
        conn: aiosqlite.Connection,
        category: str,
    ) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        if not self._hrr_available:
            return

        bank_name = f"cat:{category}"
        rows = await conn.execute_fetchall(
            "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
            (category,),
        )

        if not rows:
            await conn.execute(
                "DELETE FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            )
            await conn.commit()
            return

        vectors = [
            hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim)
            for row in rows
        ]
        bank_vector = hrr.bundle(*vectors)
        fact_count = len(vectors)

        # Check SNR
        hrr.snr_estimate(self.hrr_dim, fact_count)

        await conn.execute(
            """
            INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(bank_name) DO UPDATE SET
                vector = excluded.vector,
                dim = excluded.dim,
                fact_count = excluded.fact_count,
                updated_at = excluded.updated_at
            """,
            (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
        )
        await conn.commit()

    async def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        task = asyncio.create_task(
            self._rebuild_all_vectors(dim),
            name="holographic-rebuild-vectors",
        )
        return await _finish_owned_task(task)

    async def _rebuild_all_vectors(self, dim: int | None = None) -> int:
        conn, lock = await self._ready()
        async with lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = await conn.execute_fetchall(
                "SELECT fact_id, content, category FROM facts"
            )

            categories: set[str] = set()
            for row in rows:
                await self._compute_hrr_vector(
                    conn,
                    row["fact_id"],
                    row["content"],
                )
                categories.add(row["category"])

            for category in categories:
                await self._rebuild_bank(conn, category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: aiosqlite.Row) -> dict:
        """Convert an aiosqlite.Row to a plain dict."""
        return dict(row)

    async def close(self) -> None:
        """Release this instance's reference to the shared connection.

        The underlying connection is closed only when the last MemoryStore
        referencing the same database is closed, so closing one instance can
        never break sibling instances that still hold it. Idempotent.
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_owned(),
                name="holographic-store-close",
            )
        await _finish_owned_task(self._close_task)

    async def _close_owned(self) -> None:
        async with self._initialize_lock:
            entry = self._entry
            registry_key = self._registry_key
            if entry is None or registry_key is None:
                self._closed = True
                return
            loop = asyncio.get_running_loop()
            guard = MemoryStore._shared_guard_for_loop(loop)
            async with guard:
                if self._entry is None:
                    self._closed = True
                    return
                entry["refs"] -= 1
                try:
                    if entry["refs"] <= 0:
                        from hermes_state import _close_owned_connection

                        try:
                            await _close_owned_connection(entry["conn"])
                        finally:
                            MemoryStore._shared.pop(registry_key, None)
                finally:
                    self._entry = None
                    self._registry_key = None
                    self._conn = None
                    self._lock = None
                    self._closed = True

    async def __aenter__(self) -> "MemoryStore":
        await self._initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
