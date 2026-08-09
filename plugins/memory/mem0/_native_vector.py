"""Native-async vector stores for the retained Mem0 OSS backend."""

from __future__ import annotations

import asyncio
import builtins
import logging
from typing import Any

from ._native_oss import _finish_cleanup

logger = logging.getLogger(__name__)


class Qdrant:
    """Native-async remote Qdrant adapter matching Mem0 2.0.10 contracts."""

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
            if self.config.get("path"):
                raise RuntimeError(
                    "Mem0 OSS embedded Qdrant is not native async: "
                    "qdrant-client performs blocking file I/O in local mode."
                )
            from qdrant_client import AsyncQdrantClient, models

            client_options: dict[str, Any] = {"check_compatibility": False}
            if self.config.get("api_key"):
                client_options["api_key"] = self.config["api_key"]
            if self.config.get("url"):
                client_options["url"] = self.config["url"]
            elif self.config.get("host") and self.config.get("port"):
                client_options["host"] = self.config["host"]
                client_options["port"] = self.config["port"]
            else:
                raise RuntimeError(
                    "Mem0 OSS Qdrant requires a remote url or host/port for "
                    "native-async operation."
                )
            if self.config.get("https") is not None:
                client_options["https"] = self.config["https"]

            client = AsyncQdrantClient(**client_options)
            try:
                await self._initialize_collection(client, models)
            except BaseException:
                try:
                    await _finish_cleanup(
                        client.close(),
                        error_message=(
                            "Mem0 Qdrant cleanup failed during initialization "
                            "cancellation"
                        ),
                    )
                except Exception:
                    logger.exception("Failed to close Mem0 Qdrant client")
                raise
            self._models = models
            self._client = client
            return client

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
    def _create_filter(filters: dict[str, Any] | None) -> Any:
        if not filters:
            return None
        from mem0.vector_stores.qdrant import Qdrant as SyncQdrant

        builder = SyncQdrant.__new__(SyncQdrant)
        return builder._create_filter(filters)

    async def insert(
        self,
        vectors: builtins.list[builtins.list[float]],
        payloads: builtins.list[dict[str, Any]] | None = None,
        ids: builtins.list[Any] | None = None,
    ) -> None:
        client = await self._get_client()
        points = []
        for index, vector in enumerate(vectors):
            payload = payloads[index] if payloads else {}
            point_id = index if ids is None else ids[index]
            points.append(
                self._models.PointStruct(
                    id=point_id,
                    vector={"": vector},
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

    async def keyword_search(
        self,
        query: str,  # noqa: ARG002
        top_k: int = 5,  # noqa: ARG002
        filters: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        await self._get_client()
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
            await client.upsert(
                collection_name=self.collection_name,
                points=[
                    self._models.PointStruct(
                        id=vector_id,
                        vector={"": vector},
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

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            self._models = None
            if client is not None:
                await _finish_cleanup(
                    client.close(),
                    error_message="Mem0 Qdrant cleanup failed during cancellation",
                )
