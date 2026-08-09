"""Native-async proxy for qdrant-client's blocking embedded mode."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic_core import to_jsonable_python

from ._native_worker import NativeWorker

_MISSING_DEPENDENCY = object()


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return to_jsonable_python(value)


class NativeLocalQdrantClient:
    """Expose the retained low-level Qdrant calls through one subprocess."""

    def __init__(self, path: str | None, models: Any) -> None:
        self._path = path
        self._models = models
        self._worker = NativeWorker(
            "qdrant_client",
            worker_filename="_local_qdrant_worker.py",
        )
        self._closed = False

    async def _request(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("Cannot use a closed embedded Qdrant client")
        result = await self._worker.request(
            operation,
            fallback=_MISSING_DEPENDENCY,
            path=_dump(self._path),
            **payload,
        )
        if result is _MISSING_DEPENDENCY:
            raise RuntimeError("qdrant-client is required for embedded Qdrant")
        return result

    async def get_collections(self) -> Any:
        result = await self._request("get_collections")
        return self._models.CollectionsResponse.model_validate(result)

    async def get_collection(self, collection_name: str) -> Any:
        result = await self._request(
            "get_collection",
            collection_name=collection_name,
        )
        return self._models.CollectionInfo.model_validate(result)

    async def collection_exists(self, collection_name: str) -> bool:
        return bool(
            await self._request(
                "collection_exists",
                collection_name=collection_name,
            )
        )

    async def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: dict[str, Any],
    ) -> bool:
        return bool(
            await self._request(
                "create_collection",
                collection_name=collection_name,
                vectors_config=_dump(vectors_config),
                sparse_vectors_config={
                    name: _dump(config)
                    for name, config in sparse_vectors_config.items()
                },
            )
        )

    async def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        await self._request(
            "upsert",
            collection_name=collection_name,
            points=[_dump(point) for point in points],
        )

    async def query_points(
        self,
        *,
        collection_name: str,
        query: Any,
        query_filter: Any,
        limit: int,
        using: str | None = None,
    ) -> Any:
        sparse = isinstance(query, self._models.SparseVector)
        result = await self._request(
            "query_points",
            collection_name=collection_name,
            query=_dump(query),
            sparse=sparse,
            using=using,
            query_filter=_dump(query_filter),
            limit=limit,
        )
        return SimpleNamespace(
            points=[
                self._models.ScoredPoint.model_validate(point)
                for point in result["points"]
            ]
        )

    async def query_batch_points(
        self,
        *,
        collection_name: str,
        requests: list[Any],
    ) -> list[Any]:
        result = await self._request(
            "query_batch_points",
            collection_name=collection_name,
            requests=[_dump(request) for request in requests],
        )
        return [
            SimpleNamespace(
                points=[
                    self._models.ScoredPoint.model_validate(point)
                    for point in response["points"]
                ]
            )
            for response in result
        ]

    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: Any,
    ) -> None:
        await self._request(
            "delete",
            collection_name=collection_name,
            points_selector=_dump(points_selector),
        )

    async def set_payload(
        self,
        *,
        collection_name: str,
        payload: dict[str, Any],
        points: list[Any],
    ) -> None:
        await self._request(
            "set_payload",
            collection_name=collection_name,
            payload=_dump(payload),
            points=_dump(points),
        )

    async def update_vectors(
        self,
        *,
        collection_name: str,
        points: list[Any],
    ) -> None:
        await self._request(
            "update_vectors",
            collection_name=collection_name,
            points=[_dump(point) for point in points],
        )

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[Any],
        with_payload: bool,
    ) -> list[Any]:
        result = await self._request(
            "retrieve",
            collection_name=collection_name,
            ids=_dump(ids),
            with_payload=with_payload,
        )
        return [self._models.Record.model_validate(record) for record in result]

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[Any], Any]:
        records, offset = await self._request(
            "scroll",
            collection_name=collection_name,
            scroll_filter=_dump(scroll_filter),
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return (
            [self._models.Record.model_validate(record) for record in records],
            offset,
        )

    async def delete_collection(self, collection_name: str) -> bool:
        return bool(
            await self._request(
                "delete_collection",
                collection_name=collection_name,
            )
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._worker.close()
