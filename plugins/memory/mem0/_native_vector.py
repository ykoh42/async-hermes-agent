"""Native-async vector stores for the retained Mem0 OSS backend."""

from __future__ import annotations

import asyncio
import builtins
from contextlib import asynccontextmanager
import inspect
import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from ._native_local_qdrant import NativeLocalQdrantClient
from ._native_oss import _finish_cleanup
from ._native_sparse import NativeSparseEncoder, SparseEncoding

logger = logging.getLogger(__name__)

_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"([T ]\d{2}:\d{2}(:\d{2})?"
    r"(\.\d+)?"
    r"(Z|[+-]\d{2}:?\d{2})?"
    r")?$"
)

_PGVECTOR_OPERATOR_SQL = {
    "eq": ("payload->>%s = %s", False),
    "ne": ("payload->>%s != %s", False),
    "gt": ("(payload->>%s)::numeric > %s", True),
    "gte": ("(payload->>%s)::numeric >= %s", True),
    "lt": ("(payload->>%s)::numeric < %s", True),
    "lte": ("(payload->>%s)::numeric <= %s", True),
    "in": ("payload->>%s = ANY(%s)", False),
    "nin": ("NOT (payload->>%s = ANY(%s))", False),
    "contains": ("payload->>%s LIKE %s", False),
    "icontains": ("payload->>%s ILIKE %s", False),
}

_PGVECTOR_CONFIG_FIELDS = {
    "dbname",
    "collection_name",
    "embedding_model_dims",
    "user",
    "password",
    "host",
    "port",
    "diskann",
    "hnsw",
    "minconn",
    "maxconn",
    "sslmode",
    "connection_string",
    "connection_pool",
}


def _validate_pgvector_config(config: dict[str, Any]) -> None:
    """Match Mem0 2.0.10's PGVectorConfig validation."""
    extra_fields = set(config) - _PGVECTOR_CONFIG_FIELDS
    if extra_fields:
        extras = ", ".join(extra_fields)
        allowed = ", ".join(_PGVECTOR_CONFIG_FIELDS)
        raise ValueError(
            f"Extra fields not allowed: {extras}. Please input only the "
            f"following fields: {allowed}"
        )
    if config.get("connection_pool") is not None:
        return
    if config.get("connection_string") is not None:
        return
    if not config.get("user") and not config.get("password"):
        raise ValueError(
            "Both 'user' and 'password' must be provided when not using "
            "connection_string."
        )
    if not config.get("host") and not config.get("port"):
        raise ValueError(
            "Both 'host' and 'port' must be provided when not using "
            "connection_string."
        )


def _build_filter_conditions(
    filters: dict[str, Any] | None,
) -> tuple[builtins.list[str], builtins.list[Any]]:
    """Translate Mem0's processed metadata filters to parameterized SQL."""
    conditions: builtins.list[str] = []
    params: builtins.list[Any] = []
    if not filters:
        return conditions, params

    for key, value in filters.items():
        if key == "$or":
            groups = []
            for nested_filter in value:
                nested_conditions, nested_params = _build_filter_conditions(
                    nested_filter
                )
                if nested_conditions:
                    groups.append("(" + " AND ".join(nested_conditions) + ")")
                    params.extend(nested_params)
            if groups:
                conditions.append("(" + " OR ".join(groups) + ")")
            continue
        if key == "$not":
            groups = []
            for nested_filter in value:
                nested_conditions, nested_params = _build_filter_conditions(
                    nested_filter
                )
                if nested_conditions:
                    groups.append("(" + " AND ".join(nested_conditions) + ")")
                    params.extend(nested_params)
            if groups:
                conditions.append("NOT (" + " OR ".join(groups) + ")")
            continue
        if value == "*":
            conditions.append("payload ? %s")
            params.append(key)
            continue
        if isinstance(value, dict):
            for operator, operator_value in value.items():
                if operator not in _PGVECTOR_OPERATOR_SQL:
                    raise ValueError(
                        f"Unsupported filter operator: {operator}"
                    )
                template, numeric = _PGVECTOR_OPERATOR_SQL[operator]
                if operator in {"in", "nin"}:
                    conditions.append(template)
                    params.extend([key, [str(item) for item in operator_value]])
                elif operator in {"contains", "icontains"}:
                    escaped = (
                        str(operator_value)
                        .replace("\\", "\\\\")
                        .replace("%", "\\%")
                        .replace("_", "\\_")
                    )
                    conditions.append(template + " ESCAPE '\\'")
                    params.extend([key, f"%{escaped}%"])
                else:
                    conditions.append(template)
                    params.extend(
                        [key, float(operator_value) if numeric else str(operator_value)]
                    )
        elif isinstance(value, builtins.list):
            conditions.append("payload->>%s = ANY(%s)")
            params.extend([key, [str(item) for item in value]])
        else:
            conditions.append("payload->>%s = %s")
            params.extend(
                [key, json.dumps(value) if isinstance(value, bool) else str(value)]
            )
    return conditions, params


class OutputData(BaseModel):
    id: str | None
    score: float | None
    payload: dict[str, Any] | None


@asynccontextmanager
async def _pool_cursor(pool: Any):
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            yield cursor


def _vector_literal(vector: builtins.list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


class Qdrant:
    """Native-async Qdrant adapter matching Mem0 2.0.10 contracts."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.collection_name = self.config.get("collection_name", "mem0")
        self.embedding_model_dims = self.config.get("embedding_model_dims", 1536)
        self.on_disk = bool(self.config.get("on_disk", False))
        self._client: Any = None
        self._models: Any = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False
        self._has_bm25_slot = False
        self._bm25_encoder = NativeSparseEncoder()
        self._configured_client = self.config.get("client")
        self._owns_client = self._configured_client is None
        self._remote_options: dict[str, Any] = {}
        if self.config.get("api_key"):
            self._remote_options["api_key"] = self.config["api_key"]
        if self.config.get("url"):
            self._remote_options["url"] = self.config["url"]
        if self.config.get("host") and self.config.get("port"):
            self._remote_options["host"] = self.config["host"]
            self._remote_options["port"] = self.config["port"]
        if self.config.get("https") is not None:
            self._remote_options["https"] = self.config["https"]
        self._is_local = not self._configured_client and not self._remote_options
        self._use_embedded = (
            self._is_local and self.config.get("path") is not None
        )

    @property
    def has_bm25_slot(self) -> bool:
        return self._has_bm25_slot

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._initialize_lock:
            if self._client is not None:
                return self._client
            if self._closed:
                raise RuntimeError("Cannot use a closed Qdrant")
            from qdrant_client import AsyncQdrantClient, models

            if self._configured_client:
                client = self._configured_client
                if not inspect.iscoroutinefunction(
                    getattr(client, "get_collections", None)
                ):
                    raise RuntimeError(
                        "Mem0 OSS Qdrant requires a native async configured "
                        "client; synchronous QdrantClient instances are not "
                        "supported."
                    )
            elif self._use_embedded:
                client = NativeLocalQdrantClient(
                    self.config.get("path"),
                    models,
                )
            else:
                client_options: dict[str, Any] = {
                    **self._remote_options,
                    "check_compatibility": False,
                }
                client = AsyncQdrantClient(**client_options)
            try:
                await self._initialize_collection(client, models)
            except BaseException:
                if self._owns_client:
                    try:
                        await _finish_cleanup(
                            client.close(),
                            error_message=(
                                "Mem0 Qdrant cleanup failed during "
                                "initialization cancellation"
                            ),
                        )
                    except Exception:
                        logger.exception("Failed to close Mem0 Qdrant client")
                raise
            self._models = models
            self._client = client
            return client

    async def _initialize(self) -> None:
        await self._get_client()

    async def _initialize_collection(self, client: Any, models: Any) -> None:
        collections = await client.get_collections()
        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )
        if exists:
            info = await client.get_collection(self.collection_name)
            sparse = info.config.params.sparse_vectors
            self._has_bm25_slot = bool(sparse and "bm25" in sparse)
            if not self._has_bm25_slot:
                logger.warning(
                    "Collection '%s' predates v3 hybrid search (no 'bm25' "
                    "sparse slot). BM25 keyword scoring will be disabled for "
                    "this collection; semantic search works normally. To "
                    "enable hybrid search, use a fresh collection.",
                    self.collection_name,
                )
        else:
            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model_dims,
                    distance=models.Distance.COSINE,
                    on_disk=self.on_disk,
                ),
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
            self._has_bm25_slot = True

        if self._is_local:
            return
        for field_name in ("user_id", "agent_id", "run_id", "actor_id"):
            try:
                await client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )
            except Exception as exc:
                logger.debug(
                    "Qdrant index for %s may already exist: %s",
                    field_name,
                    exc,
                )

    @staticmethod
    def _is_datetime_range(range_values: dict[str, Any]) -> bool:
        return all(
            isinstance(value, str) and _ISO_DATETIME_RE.match(value)
            for value in range_values.values()
        )

    def _build_field_condition(self, key: str, value: Any) -> Any:
        models = self._models
        if not isinstance(value, dict):
            if value == "*":
                return None
            if isinstance(value, builtins.list):
                return models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value),
                )
            return models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )

        operators = set(value)
        range_operators = {"gt", "gte", "lt", "lte"}
        non_range_operators = operators - range_operators
        if operators & range_operators:
            if non_range_operators:
                raise ValueError(
                    f"Cannot mix range operators "
                    f"({operators & range_operators}) with non-range "
                    f"operators ({non_range_operators}) for field '{key}'. "
                    "Use AND to combine them as separate conditions."
                )
            range_values = {
                operator: value[operator]
                for operator in range_operators
                if operator in value
            }
            range_type = (
                models.DatetimeRange
                if self._is_datetime_range(range_values)
                else models.Range
            )
            try:
                return models.FieldCondition(
                    key=key,
                    range=range_type(**range_values),
                )
            except (ValueError, TypeError) as exc:
                if range_type is models.DatetimeRange:
                    raise ValueError(
                        f"Invalid datetime value in range filter for field "
                        f"'{key}': {exc}"
                    ) from exc
                raise
        if "eq" in value:
            return models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value["eq"]),
            )
        if "ne" in value:
            return models.FieldCondition(
                key=key,
                match=models.MatchExcept(**{"except": [value["ne"]]}),
            )
        if "in" in value:
            return models.FieldCondition(
                key=key,
                match=models.MatchAny(any=value["in"]),
            )
        if "nin" in value:
            return models.FieldCondition(
                key=key,
                match=models.MatchExcept(**{"except": value["nin"]}),
            )
        if "contains" in value or "icontains" in value:
            operator = "icontains" if "icontains" in value else "contains"
            if operator == "icontains":
                logger.debug(
                    "icontains on field '%s': Qdrant MatchText case "
                    "sensitivity depends on full-text index configuration. "
                    "Without a full-text index this behaves as a "
                    "case-sensitive substring match (same as 'contains').",
                    key,
                )
            return models.FieldCondition(
                key=key,
                match=models.MatchText(text=value[operator]),
            )
        supported = {
            "eq",
            "ne",
            "gt",
            "gte",
            "lt",
            "lte",
            "in",
            "nin",
            "contains",
            "icontains",
        }
        raise ValueError(
            f"Unsupported filter operator(s) for field '{key}': "
            f"{operators}. Supported operators: {supported}"
        )

    def _create_filter(self, filters: dict[str, Any] | None) -> Any:
        if not filters:
            return None
        normalized = {}
        key_map = {"$or": "OR", "$not": "NOT", "$and": "AND"}
        for key, value in filters.items():
            normalized.setdefault(key_map.get(key, key), value)

        must = []
        should = []
        must_not = []
        for key, value in normalized.items():
            if key in {"AND", "OR", "NOT"}:
                if not isinstance(value, builtins.list):
                    raise ValueError(
                        f"{key} filter value must be a list of filter dicts, "
                        f"got {type(value).__name__}"
                    )
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{key} filter list item at index {index} must be "
                            f"a dict, got {type(item).__name__}: {item!r}"
                        )

            if key == "AND":
                for nested in value:
                    condition = self._create_filter(nested)
                    if condition:
                        must.append(condition)
            elif key == "OR":
                for nested in value:
                    condition = self._create_filter(nested)
                    if condition:
                        should.append(condition)
            elif key == "NOT":
                for nested in value:
                    condition = self._create_filter(nested)
                    if condition:
                        must_not.append(condition)
            else:
                condition = self._build_field_condition(key, value)
                if condition is not None:
                    must.append(condition)

        if not any((must, should, must_not)):
            return None
        return self._models.Filter(
            must=must or None,
            should=should or None,
            must_not=must_not or None,
        )

    async def insert(
        self,
        vectors: builtins.list[builtins.list[float]],
        payloads: builtins.list[dict[str, Any]] | None = None,
        ids: builtins.list[Any] | None = None,
    ) -> None:
        client = await self._get_client()
        sparse_vectors: builtins.list[Any | None] = [None] * len(vectors)
        if self._has_bm25_slot and payloads:
            texts: builtins.list[str] = []
            text_indices: builtins.list[int] = []
            for index, payload in enumerate(payloads):
                text = payload.get("text_lemmatized") or payload.get("data", "")
                if text:
                    texts.append(text)
                    text_indices.append(index)
            if texts:
                try:
                    encodings = await self._bm25_encoder.encode_batch(texts)
                    if encodings is not None:
                        if len(encodings) != len(texts):
                            logger.warning(
                                "BM25 batch returned %s results for %s texts; "
                                "falling back to per-row encoding",
                                len(encodings),
                                len(texts),
                            )
                            raise ValueError("count mismatch")
                        for index, encoding in zip(
                            text_indices,
                            encodings,
                            strict=True,
                        ):
                            sparse_vectors[index] = self._sparse_vector(encoding)
                except Exception as exc:
                    logger.debug(
                        "Batch BM25 encoding failed, falling back to per-row: %s",
                        exc,
                    )
                    for index, text in zip(text_indices, texts, strict=True):
                        sparse_vectors[index] = await self._encode_bm25(text)

        points = []
        for index, vector in enumerate(vectors):
            payload = payloads[index] if payloads else {}
            point_id = index if ids is None else ids[index]
            named_vectors: dict[str, Any] = {"": vector}
            if self._has_bm25_slot and sparse_vectors[index] is not None:
                named_vectors["bm25"] = sparse_vectors[index]
            points.append(
                self._models.PointStruct(
                    id=point_id,
                    vector=named_vectors,
                    payload=payload,
                )
            )
        await client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search(
        self,
        query: str,  # noqa: ARG002
        vectors: builtins.list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[Any]:
        client = await self._get_client()
        response = await client.query_points(
            collection_name=self.collection_name,
            query=vectors,
            query_filter=self._create_filter(filters),
            limit=top_k,
        )
        return response.points

    async def search_batch(
        self,
        queries: builtins.list[str],
        vectors_list: builtins.list[builtins.list[float]],
        top_k: int = 1,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[builtins.list[Any]]:
        client = await self._get_client()
        query_filter = self._create_filter(filters)
        requests = [
            self._models.QueryRequest(
                query=vector,
                filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            for vector in vectors_list
        ]
        try:
            responses = await client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )
            return [response.points for response in responses]
        except Exception as exc:
            logger.warning(
                "Qdrant batch search failed, falling back to sequential: %s",
                exc,
            )
            return [
                await self.search(query, vector, top_k=top_k, filters=filters)
                for query, vector in zip(queries, vectors_list, strict=False)
            ]

    def _sparse_vector(self, encoding: SparseEncoding) -> Any:
        indices, values = encoding
        return self._models.SparseVector(indices=indices, values=values)

    async def _encode_bm25(self, text: str) -> Any | None:
        try:
            encodings = await self._bm25_encoder.encode_batch([text])
            if encodings:
                return self._sparse_vector(encodings[0])
        except Exception as exc:
            logger.debug("BM25 encoding failed: %s", exc)
        return None

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[Any] | None:
        client = await self._get_client()
        if not self._has_bm25_slot:
            return None
        sparse_query = await self._encode_bm25(query)
        if sparse_query is None:
            return None
        try:
            response = await client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using="bm25",
                query_filter=self._create_filter(filters),
                limit=top_k,
            )
            return response.points
        except Exception as exc:
            logger.debug("BM25 keyword search failed: %s", exc)
            return None

    async def delete(self, vector_id: Any) -> None:
        client = await self._get_client()
        await client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.PointIdsList(points=[vector_id]),
        )

    async def update(
        self,
        vector_id: Any,
        vector: builtins.list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        client = await self._get_client()
        if vector is not None and payload is not None:
            named_vectors: dict[str, Any] = {"": vector}
            if self._has_bm25_slot:
                text = payload.get("text_lemmatized") or payload.get("data", "")
                if text:
                    sparse = await self._encode_bm25(text)
                    if sparse is not None:
                        named_vectors["bm25"] = sparse
            await client.upsert(
                collection_name=self.collection_name,
                points=[
                    self._models.PointStruct(
                        id=vector_id,
                        vector=named_vectors,
                        payload=payload,
                    )
                ],
            )
            return
        if payload is not None:
            await client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[vector_id],
            )
        if vector is not None:
            await client.update_vectors(
                collection_name=self.collection_name,
                points=[
                    self._models.PointVectors(id=vector_id, vector=vector)
                ],
            )

    async def get(self, vector_id: Any) -> Any:
        client = await self._get_client()
        results = await client.retrieve(
            collection_name=self.collection_name,
            ids=[vector_id],
            with_payload=True,
        )
        return results[0] if results else None

    async def list_cols(self) -> Any:
        client = await self._get_client()
        return await client.get_collections()

    async def delete_col(self) -> None:
        client = await self._get_client()
        await client.delete_collection(collection_name=self.collection_name)

    async def col_info(self) -> Any:
        client = await self._get_client()
        return await client.get_collection(self.collection_name)

    async def list(
        self,
        filters: dict[str, Any] | None = None,
        top_k: int = 100,
    ) -> Any:
        client = await self._get_client()
        return await client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._create_filter(filters),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    async def reset(self) -> None:
        client = await self._get_client()
        logger.warning("Resetting index %s...", self.collection_name)
        await client.delete_collection(collection_name=self.collection_name)
        self._has_bm25_slot = False
        await self._initialize_collection(client, self._models)

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            encoder = self._bm25_encoder
            self._client = None
            self._models = None
            close_tasks = [asyncio.create_task(encoder.close())]
            if client is not None and self._owns_client:
                close_tasks.append(asyncio.create_task(client.close()))
            close_group = asyncio.gather(*close_tasks, return_exceptions=True)
            try:
                results = await asyncio.shield(close_group)
            except asyncio.CancelledError:
                results = await close_group
                for result in results:
                    if isinstance(result, Exception):
                        logger.error("Mem0 Qdrant cleanup failed: %s", result)
                raise
            errors = [
                result for result in results if isinstance(result, BaseException)
            ]
            if errors:
                raise errors[0]


class PGVector:
    """Native-async psycopg 3 adapter matching Mem0 2.0.10 PGVector."""

    def __init__(self, config: dict[str, Any]) -> None:
        _validate_pgvector_config(config)
        self.config = dict(config)
        self.collection_name = self.config.get("collection_name", "mem0")
        self.embedding_model_dims = self.config.get(
            "embedding_model_dims",
            1536,
        )
        self.use_diskann = bool(self.config.get("diskann", False))
        self.use_hnsw = bool(self.config.get("hnsw", True))
        self._pool: Any = None
        self._sql: Any = None
        self._json_adapter: Any = None
        self._initialize_lock = asyncio.Lock()
        self._collection_lock = asyncio.Lock()
        self._collection_ensured = False
        self._closed = False
        self._owns_pool = self.config.get("connection_pool") is None

    async def _initialize(self) -> None:
        await self._get_pool()

    async def _get_pool(self) -> Any:
        if self._closed:
            raise RuntimeError("Cannot use a closed PGVector")
        if self._pool is not None:
            return self._pool
        async with self._initialize_lock:
            if self._closed:
                raise RuntimeError("Cannot use a closed PGVector")
            if self._pool is not None:
                return self._pool

            from psycopg import capabilities, conninfo, sql
            from psycopg.types.json import Json
            from psycopg_pool import AsyncConnectionPool

            if not capabilities.has_cancel_safe(check=True):
                raise RuntimeError(
                    "Mem0 PGVector requires libpq 17 or newer for native-async "
                    "query cancellation. Install the pinned "
                    "psycopg[binary,pool] extra."
                )

            configured_pool = self.config.get("connection_pool")
            if configured_pool is not None:
                if not isinstance(configured_pool, AsyncConnectionPool):
                    raise RuntimeError(
                        "Mem0 PGVector connection_pool must be a native-async "
                        "psycopg_pool.AsyncConnectionPool."
                    )
                pool = configured_pool
            else:
                connection_string = self.config.get("connection_string")
                sslmode = self.config.get("sslmode")
                if connection_string:
                    connection_options = {"sslmode": sslmode} if sslmode else {}
                    connection_string = conninfo.make_conninfo(
                        connection_string,
                        **connection_options,
                    )
                else:
                    connection_options = {
                        key: self.config.get(key)
                        for key in (
                            "dbname",
                            "user",
                            "password",
                            "host",
                            "port",
                            "sslmode",
                        )
                        if self.config.get(key) is not None
                    }
                    connection_options.setdefault("dbname", "postgres")
                    connection_string = conninfo.make_conninfo(
                        "",
                        **connection_options,
                    )
                pool = AsyncConnectionPool(
                    conninfo=connection_string,
                    min_size=self.config.get("minconn", 1),
                    max_size=self.config.get("maxconn", 5),
                    open=False,
                )

            try:
                if self._owns_pool:
                    await pool.open(wait=True)
                await self._ensure_collection_with_pool(pool, sql)
            except BaseException:
                if self._owns_pool:
                    try:
                        await _finish_cleanup(
                            pool.close(),
                            error_message=(
                                "Mem0 PGVector cleanup failed during "
                                "initialization cancellation"
                            ),
                        )
                    except Exception:
                        logger.exception("Failed to close Mem0 PGVector pool")
                raise

            self._sql = sql
            self._json_adapter = Json
            self._pool = pool
            return pool

    def _collection(self) -> Any:
        return self._sql.Identifier(self.collection_name)

    async def _list_cols_with_pool(self, pool: Any, sql: Any) -> builtins.list[str]:
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            return [row[0] for row in await cursor.fetchall()]

    async def _ensure_collection_with_pool(self, pool: Any, sql: Any) -> None:
        async with self._collection_lock:
            if self._collection_ensured:
                return
            collections = await self._list_cols_with_pool(pool, sql)
            if self.collection_name in collections:
                async with _pool_cursor(pool) as cursor:
                    await cursor.execute(
                        "SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid = %s::regclass AND attname = 'vector'",
                        (self.collection_name,),
                    )
                    dimension = await cursor.fetchone()
                if (
                    dimension
                    and dimension[0] > 0
                    and dimension[0] != self.embedding_model_dims
                ):
                    async with _pool_cursor(pool) as cursor:
                        await cursor.execute(
                            sql.SQL("DROP TABLE IF EXISTS {}").format(
                                sql.Identifier(self.collection_name)
                            )
                        )
                    await self._create_col_with_pool(pool, sql)
            else:
                await self._create_col_with_pool(pool, sql)
            self._collection_ensured = True

    async def _ensure_collection(self) -> Any:
        pool = await self._get_pool()
        if not self._collection_ensured:
            await self._ensure_collection_with_pool(pool, self._sql)
        return pool

    async def _create_col_with_pool(self, pool: Any, sql: Any) -> None:
        collection = sql.Identifier(self.collection_name)
        async with _pool_cursor(pool) as cursor:
            await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id UUID PRIMARY KEY,
                        vector vector({}),
                        payload JSONB
                    );
                    """
                ).format(
                    collection,
                    sql.Literal(self.embedding_model_dims),
                )
            )
            if self.use_diskann and self.embedding_model_dims < 2000:
                await cursor.execute(
                    "SELECT * FROM pg_extension WHERE extname = 'vectorscale'"
                )
                if await cursor.fetchone():
                    await cursor.execute(
                        sql.SQL(
                            """
                            CREATE INDEX IF NOT EXISTS {} ON {}
                            USING diskann (vector);
                            """
                        ).format(
                            sql.Identifier(
                                f"{self.collection_name}_diskann_idx"
                            ),
                            collection,
                        )
                    )
            elif self.use_hnsw:
                await cursor.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {} ON {}
                        USING hnsw (vector vector_cosine_ops)
                        """
                    ).format(
                        sql.Identifier(f"{self.collection_name}_hnsw_idx"),
                        collection,
                    )
                )
            await cursor.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {} ON {}
                    USING gin(to_tsvector(
                        'simple', payload->>'text_lemmatized'
                    ));
                    """
                ).format(
                    sql.Identifier(
                        f"{self.collection_name}_text_lemmatized_idx"
                    ),
                    collection,
                )
            )

    async def create_col(self) -> None:
        pool = await self._get_pool()
        await self._create_col_with_pool(pool, self._sql)

    async def insert(
        self,
        vectors: builtins.list[builtins.list[float]],
        payloads: builtins.list[dict[str, Any]] | None = None,
        ids: builtins.list[Any] | None = None,
    ) -> None:
        pool = await self._ensure_collection()
        if payloads is None or ids is None:
            raise TypeError("'NoneType' object is not iterable")
        json_payloads = [json.dumps(payload) for payload in payloads]
        rows = [
            (vector_id, _vector_literal(vector), payload)
            for vector_id, vector, payload in zip(
                ids,
                vectors,
                json_payloads,
                strict=False,
            )
        ]
        async with _pool_cursor(pool) as cursor:
            await cursor.executemany(
                self._sql.SQL(
                    "INSERT INTO {} (id, vector, payload) VALUES (%s, %s, %s)"
                ).format(self._collection()),
                rows,
            )

    async def search(
        self,
        query: str,  # noqa: ARG002
        vectors: builtins.list[float],
        top_k: int | None = 5,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[OutputData]:
        pool = await self._ensure_collection()
        conditions, filter_params = _build_filter_conditions(filters)
        filter_clause = self._sql.SQL(
            "WHERE " + " AND ".join(conditions) if conditions else ""
        )
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                self._sql.SQL(
                    """
                    SELECT id, vector <=> %s::vector AS distance, payload
                    FROM {}
                    {}
                    ORDER BY distance
                    LIMIT %s
                    """
                ).format(self._collection(), filter_clause),
                (_vector_literal(vectors), *filter_params, top_k),
            )
            rows = await cursor.fetchall()
        return [
            OutputData(
                id=str(row[0]),
                score=max(0.0, 1.0 - float(row[1])),
                payload=row[2],
            )
            for row in rows
        ]

    async def search_batch(
        self,
        queries: builtins.list[str],
        vectors_list: builtins.list[builtins.list[float]],
        top_k: int = 1,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[builtins.list[OutputData]]:
        return [
            await self.search(query, vector, top_k=top_k, filters=filters)
            for query, vector in zip(queries, vectors_list, strict=False)
        ]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[OutputData] | None:
        pool = await self._ensure_collection()
        conditions, filter_params = _build_filter_conditions(filters)
        filter_clause = self._sql.SQL(
            "AND " + " AND ".join(conditions) if conditions else ""
        )
        try:
            async with _pool_cursor(pool) as cursor:
                await cursor.execute(
                    self._sql.SQL(
                        """
                        SELECT id, ts_rank_cd(
                            to_tsvector('simple', payload->>'text_lemmatized'),
                            plainto_tsquery('simple', %s)
                        ) AS score, payload
                        FROM {}
                        WHERE to_tsvector(
                            'simple', payload->>'text_lemmatized'
                        ) @@ plainto_tsquery('simple', %s)
                        {}
                        ORDER BY score DESC
                        LIMIT %s
                        """
                    ).format(self._collection(), filter_clause),
                    (query, query, *filter_params, top_k),
                )
                rows = await cursor.fetchall()
            return [
                OutputData(
                    id=str(row[0]),
                    score=float(row[1]),
                    payload=row[2],
                )
                for row in rows
            ]
        except Exception as exc:
            logger.debug("Keyword search failed: %s", exc)
            return None

    async def delete(self, vector_id: str) -> None:
        pool = await self._ensure_collection()
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                self._sql.SQL("DELETE FROM {} WHERE id = %s").format(
                    self._collection()
                ),
                (vector_id,),
            )

    async def update(
        self,
        vector_id: str,
        vector: builtins.list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        pool = await self._ensure_collection()
        async with _pool_cursor(pool) as cursor:
            if vector is not None:
                await cursor.execute(
                    self._sql.SQL(
                        "UPDATE {} SET vector = %s WHERE id = %s"
                    ).format(self._collection()),
                    (_vector_literal(vector), vector_id),
                )
            if payload is not None:
                await cursor.execute(
                    self._sql.SQL(
                        "UPDATE {} SET payload = %s WHERE id = %s"
                    ).format(self._collection()),
                    (self._json_adapter(payload), vector_id),
                )

    async def get(self, vector_id: str) -> OutputData | None:
        pool = await self._ensure_collection()
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                self._sql.SQL(
                    "SELECT id, vector, payload FROM {} WHERE id = %s"
                ).format(self._collection()),
                (vector_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return OutputData(id=str(row[0]), score=None, payload=row[2])

    async def list_cols(self) -> builtins.list[str]:
        pool = await self._get_pool()
        return await self._list_cols_with_pool(pool, self._sql)

    async def delete_col(self) -> None:
        pool = await self._get_pool()
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                self._sql.SQL("DROP TABLE IF EXISTS {}").format(
                    self._collection()
                )
            )

    async def col_info(self) -> dict[str, Any]:
        pool = await self._ensure_collection()
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                self._sql.SQL(
                    """
                    SELECT table_name,
                        (SELECT COUNT(*) FROM {}) AS row_count,
                        (SELECT pg_size_pretty(
                            pg_total_relation_size({}::regclass)
                        )) AS total_size
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                    """
                ).format(
                    self._collection(),
                    self._sql.Literal(self.collection_name),
                ),
                (self.collection_name,),
            )
            row = await cursor.fetchone()
        return {"name": row[0], "count": row[1], "size": row[2]}

    async def list(
        self,
        filters: dict[str, Any] | None = None,
        top_k: int | None = 100,
    ) -> builtins.list[builtins.list[OutputData]]:
        pool = await self._ensure_collection()
        conditions, filter_params = _build_filter_conditions(filters)
        filter_clause = self._sql.SQL(
            "WHERE " + " AND ".join(conditions) if conditions else ""
        )
        async with _pool_cursor(pool) as cursor:
            await cursor.execute(
                self._sql.SQL(
                    "SELECT id, vector, payload FROM {} {} LIMIT %s"
                ).format(self._collection(), filter_clause),
                (*filter_params, top_k),
            )
            rows = await cursor.fetchall()
        return [
            [
                OutputData(id=str(row[0]), score=None, payload=row[2])
                for row in rows
            ]
        ]

    async def reset(self) -> None:
        await self._ensure_collection()
        logger.warning("Resetting index %s...", self.collection_name)
        await self.delete_col()
        await self.create_col()

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            pool = self._pool
            self._pool = None
            self._sql = None
            self._json_adapter = None
            if pool is not None and self._owns_pool:
                await _finish_cleanup(
                    pool.close(),
                    error_message=(
                        "Mem0 PGVector cleanup failed during cancellation"
                    ),
                )
