"""Stdio worker owning one blocking embedded Qdrant client."""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any

_client: Any = None


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def _filter(models: Any, value: Any) -> Any:
    return models.Filter.model_validate(value) if value is not None else None


def _error(exc: Exception) -> dict[str, Any]:
    try:
        json.dumps(exc.args)
        args = list(exc.args)
    except (TypeError, ValueError):
        args = [str(exc)]
    return {
        "builtin_type": (
            type(exc).__name__ if type(exc).__module__ == "builtins" else None
        ),
        "args": args,
        "message": f"{type(exc).__name__}: {exc}",
    }


def _execute(request: dict[str, Any]) -> Any:
    global _client
    operation = request.get("operation")
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(path=request["path"])

    from qdrant_client import models

    if operation == "get_collections":
        return _dump(_client.get_collections())
    if operation == "get_collection":
        return _dump(_client.get_collection(request["collection_name"]))
    if operation == "collection_exists":
        return _client.collection_exists(request["collection_name"])
    if operation == "create_collection":
        vectors = models.VectorParams.model_validate(request["vectors_config"])
        sparse_vectors = {
            name: models.SparseVectorParams.model_validate(config)
            for name, config in request["sparse_vectors_config"].items()
        }
        return _client.create_collection(
            collection_name=request["collection_name"],
            vectors_config=vectors,
            sparse_vectors_config=sparse_vectors,
        )
    if operation == "upsert":
        points = [
            models.PointStruct.model_validate(point)
            for point in request["points"]
        ]
        _client.upsert(
            collection_name=request["collection_name"],
            points=points,
        )
        return None
    if operation == "query_points":
        query = request["query"]
        if request.get("sparse"):
            query = models.SparseVector.model_validate(query)
        response = _client.query_points(
            collection_name=request["collection_name"],
            query=query,
            using=request.get("using"),
            query_filter=_filter(models, request.get("query_filter")),
            limit=request["limit"],
        )
        return _dump(response)
    if operation == "query_batch_points":
        requests = [
            models.QueryRequest.model_validate(item)
            for item in request["requests"]
        ]
        return [
            _dump(response)
            for response in _client.query_batch_points(
                collection_name=request["collection_name"],
                requests=requests,
            )
        ]
    if operation == "delete":
        selector = models.PointIdsList.model_validate(request["points_selector"])
        _client.delete(
            collection_name=request["collection_name"],
            points_selector=selector,
        )
        return None
    if operation == "set_payload":
        _client.set_payload(
            collection_name=request["collection_name"],
            payload=request["payload"],
            points=request["points"],
        )
        return None
    if operation == "update_vectors":
        points = [
            models.PointVectors.model_validate(point)
            for point in request["points"]
        ]
        _client.update_vectors(
            collection_name=request["collection_name"],
            points=points,
        )
        return None
    if operation == "retrieve":
        return [
            _dump(record)
            for record in _client.retrieve(
                collection_name=request["collection_name"],
                ids=request["ids"],
                with_payload=request["with_payload"],
            )
        ]
    if operation == "scroll":
        records, offset = _client.scroll(
            collection_name=request["collection_name"],
            scroll_filter=_filter(models, request.get("scroll_filter")),
            limit=request["limit"],
            with_payload=request["with_payload"],
            with_vectors=request["with_vectors"],
        )
        if isinstance(offset, uuid.UUID):
            offset = str(offset)
        return [[_dump(record) for record in records], offset]
    if operation == "delete_collection":
        return _client.delete_collection(request["collection_name"])
    raise ValueError(f"Unsupported embedded Qdrant operation: {operation}")


def _serve_stdio() -> None:
    global _client
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("operation") == "close":
                    return
                response = {"result": _execute(request)}
            except Exception as exc:
                response = {"error": _error(exc)}
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        if _client is not None:
            _client.close()
            _client = None


if __name__ == "__main__" and sys.argv[1:] == ["--stdio"]:
    _serve_stdio()
