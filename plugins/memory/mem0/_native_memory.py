"""Native-async Mem0 OSS memory orchestration."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import aiofiles.os

from ._native_oss import (
    OllamaEmbedding,
    OllamaLLM,
    OpenAIEmbedding,
    OpenAILLM,
    SQLiteManager,
)
from ._native_vector import Qdrant

logger = logging.getLogger(__name__)

_PROMOTED_PAYLOAD_KEYS = (
    "user_id",
    "agent_id",
    "run_id",
    "actor_id",
    "role",
    "attributed_to",
    "expiration_date",
)
_CORE_PAYLOAD_KEYS = {
    "data",
    "hash",
    "created_at",
    "updated_at",
    "id",
    "text_lemmatized",
    "attributed_to",
    *_PROMOTED_PAYLOAD_KEYS,
}
_UNSET = object()


def _validate_entity_id(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. "
            "Provide a valid identifier."
        )
    if any(character.isspace() for character in trimmed):
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. "
            "Provide a valid identifier without spaces."
        )
    return trimmed


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed


def _normalize_timestamp(timestamp: str | None) -> str | None:
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _is_expired(payload: dict[str, Any] | None) -> bool:
    if not payload or not payload.get("expiration_date"):
        return False
    try:
        return date.fromisoformat(str(payload["expiration_date"])) < datetime.now(
            timezone.utc
        ).date()
    except ValueError:
        return False


def _normalize_expiration_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError(
                "expiration_date must be a valid date in YYYY-MM-DD format."
            ) from exc
    raise ValueError(
        "expiration_date must be a date string in YYYY-MM-DD format."
    )


class Memory:
    """Retained Mem0 OSS runtime with native coroutine boundaries."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = deepcopy(config)
        self.embedding_model: Any = None
        self.llm: Any = None
        self.vector_store: Any = None
        self.db: Any = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            if self._closed:
                raise RuntimeError("Cannot initialize a closed Mem0 Memory")

            embedder = self.config.get("embedder") or {}
            embedder_provider = embedder.get("provider")
            embedder_config = dict(embedder.get("config") or {})
            if embedder_provider == "openai":
                embedding_model = OpenAIEmbedding(embedder_config)
            elif embedder_provider == "ollama":
                embedding_model = OllamaEmbedding(embedder_config)
            else:
                raise ValueError(
                    f"Unsupported native Mem0 embedder provider: {embedder_provider}"
                )

            llm = self.config.get("llm") or {}
            llm_provider = llm.get("provider")
            llm_config = dict(llm.get("config") or {})
            if llm_provider == "openai":
                llm_client = OpenAILLM(llm_config)
            elif llm_provider == "ollama":
                llm_client = OllamaLLM(llm_config)
            else:
                raise ValueError(
                    f"Unsupported native Mem0 LLM provider: {llm_provider}"
                )

            vector = self.config.get("vector_store") or {}
            vector_provider = vector.get("provider")
            vector_config = dict(vector.get("config") or {})
            if vector_provider == "qdrant":
                if vector_config.get("path"):
                    raise RuntimeError(
                        "Mem0 OSS embedded Qdrant is not native async: "
                        "qdrant-client performs blocking file I/O in local mode."
                    )
                vector_store = Qdrant(vector_config)
            elif vector_provider == "pgvector":
                raise RuntimeError(
                    "Mem0 OSS PGVector native-async runtime is not implemented yet."
                )
            else:
                raise ValueError(
                    f"Unsupported native Mem0 vector provider: {vector_provider}"
                )

            history_path = self.config.get("history_db_path")
            if not history_path:
                mem0_dir = os.getenv("MEM0_DIR") or os.path.join(
                    os.path.expanduser("~"), ".mem0"
                )
                history_path = os.path.join(mem0_dir, "history.db")
            history_path = os.path.expanduser(str(history_path))
            if history_path != ":memory:":
                parent = str(Path(history_path).parent)
                await aiofiles.os.makedirs(parent, exist_ok=True)

            database = SQLiteManager(history_path)
            try:
                await vector_store._initialize()
                await database._initialize()
            except BaseException:
                await asyncio.gather(
                    vector_store.close(),
                    database.close(),
                    return_exceptions=True,
                )
                raise

            self.embedding_model = embedding_model
            self.llm = llm_client
            self.vector_store = vector_store
            self.db = database
            self._initialized = True

    async def _create_memory(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid.uuid4())
        payload = deepcopy(metadata)
        payload["data"] = text
        payload["hash"] = hashlib.md5(text.encode()).hexdigest()
        if "created_at" not in payload:
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
        payload["updated_at"] = payload["created_at"]
        # ``mem0ai`` returns the original text when its optional spaCy extra is
        # absent.  The pinned ``mem0`` extra intentionally has that core shape.
        payload["text_lemmatized"] = text
        await self.vector_store.insert(
            vectors=[embedding],
            ids=[memory_id],
            payloads=[payload],
        )
        await self.db.add_history(
            memory_id,
            None,
            text,
            "ADD",
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            actor_id=payload.get("actor_id"),
            role=payload.get("role"),
        )
        return memory_id

    async def add(
        self,
        messages: str | dict[str, Any] | list[dict[str, Any]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: Any = None,
        expiration_date: Any = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        await self.initialize()
        if timestamp is not None:
            raise ValueError(
                "The timestamp parameter is not supported by the OSS Memory SDK."
            )
        normalized_expiration_date = _normalize_expiration_date(expiration_date)
        user_id = _validate_entity_id(user_id, "user_id")
        agent_id = _validate_entity_id(agent_id, "agent_id")
        run_id = _validate_entity_id(run_id, "run_id")
        if not any((user_id, agent_id, run_id)):
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be provided."
            )
        if memory_type is not None and memory_type != "procedural_memory":
            raise ValueError(
                "Invalid 'memory_type'. Please pass procedural_memory to create "
                "procedural memories."
            )
        if isinstance(messages, str):
            normalized_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            normalized_messages = [messages]
        elif isinstance(messages, list):
            normalized_messages = messages
        else:
            raise ValueError("messages must be str, dict, or list[dict]")

        if memory_type == "procedural_memory":
            raise RuntimeError(
                "Mem0 OSS procedural memory pipeline is not native async yet; "
                "no memory was written."
            )
        if infer:
            raise RuntimeError(
                "Mem0 OSS infer=True pipeline is not native async yet; no memory "
                "was written."
            )

        base_metadata = deepcopy(metadata) if metadata else {}
        if normalized_expiration_date is not None:
            base_metadata["expiration_date"] = normalized_expiration_date
        for key, value in (
            ("user_id", user_id),
            ("agent_id", agent_id),
            ("run_id", run_id),
        ):
            if value:
                base_metadata[key] = value

        returned_memories = []
        for message in normalized_messages:
            if (
                not isinstance(message, dict)
                or message.get("role") is None
                or message.get("content") is None
                or message["role"] == "system"
            ):
                continue
            per_message_metadata = deepcopy(base_metadata)
            per_message_metadata["role"] = message["role"]
            actor_name = message.get("name")
            if actor_name:
                per_message_metadata["actor_id"] = actor_name
            text = message["content"]
            embedding = await self.embedding_model.embed(text, "add")
            memory_id = await self._create_memory(
                text,
                embedding,
                per_message_metadata,
            )
            returned_memories.append(
                {
                    "id": memory_id,
                    "memory": text,
                    "event": "ADD",
                    "actor_id": actor_name if actor_name else None,
                    "role": message["role"],
                }
            )
        return {"results": returned_memories}

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        rerank: bool = False,  # noqa: ARG002
        explain: bool = False,
        reference_date: Any = None,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        await self.initialize()
        if reference_date is not None:
            raise ValueError(
                "The reference_date parameter is not supported by the OSS Memory SDK."
            )
        invalid_entity_parameters = {"user_id", "agent_id", "run_id"} & kwargs.keys()
        if invalid_entity_parameters:
            raise ValueError(
                f"Top-level entity parameters {invalid_entity_parameters} are not "
                "supported in search(). Use filters={'user_id': '...'} instead."
            )
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValueError("threshold must be a valid number")
            if threshold < 0 or threshold > 1:
                raise ValueError(
                    f"Invalid threshold: {threshold}. Must be between 0 and 1 "
                    "(inclusive)."
                )
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )
        query = _validate_query(query)
        if threshold is None:
            threshold = 0.1

        effective_filters = dict(filters or {})
        for key in ("user_id", "agent_id", "run_id"):
            if key in effective_filters:
                effective_filters[key] = _validate_entity_id(
                    effective_filters[key], key
                )
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        embedding = await self.embedding_model.embed(query, "search")
        internal_limit = max(top_k * 4, 60)
        semantic_results = await self.vector_store.search(
            query=query,
            vectors=embedding,
            top_k=internal_limit,
            filters=effective_filters,
        )
        await self.vector_store.keyword_search(
            query=query,
            top_k=internal_limit,
            filters=effective_filters,
        )

        candidates = []
        for memory in semantic_results:
            payload = getattr(memory, "payload", None) or {}
            if not show_expired and _is_expired(payload):
                continue
            score = getattr(memory, "score", None) or 0.0
            if score < threshold:
                continue
            candidates.append(
                {
                    "id": str(memory.id),
                    "score": min(score, 1.0),
                    "payload": payload,
                }
            )
        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

        results = []
        for candidate in candidates[:top_k]:
            payload = candidate["payload"]
            if not payload.get("data"):
                continue
            item: dict[str, Any] = {
                "id": candidate["id"],
                "memory": payload.get("data", ""),
                "hash": payload.get("hash"),
                "metadata": None,
                "score": candidate["score"],
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }
            for key in _PROMOTED_PAYLOAD_KEYS:
                if key in payload:
                    item[key] = payload[key]
            additional_metadata = {
                key: value
                for key, value in payload.items()
                if key not in _CORE_PAYLOAD_KEYS
            }
            if additional_metadata:
                item["metadata"] = additional_metadata
            if explain:
                item["score_details"] = {
                    "semantic_score": candidate["score"],
                    "bm25_score": 0.0,
                    "entity_boost": 0.0,
                    "raw_score": candidate["score"],
                    "max_possible_score": 1.0,
                    "final_score": candidate["score"],
                    "threshold": threshold,
                }
            results.append(item)
        return {"results": results}

    async def update(
        self,
        memory_id: str,
        data: str | None = None,
        metadata: dict[str, Any] | None = None,
        expiration_date: Any = _UNSET,
    ) -> dict[str, str]:
        await self.initialize()
        if data is None and metadata is None and expiration_date is _UNSET:
            raise ValueError(
                "At least one of data, metadata, or expiration_date must be provided."
            )
        update_metadata = deepcopy(metadata) if metadata is not None else None
        if expiration_date is not _UNSET:
            update_metadata = update_metadata or {}
            update_metadata["expiration_date"] = _normalize_expiration_date(
                expiration_date
            )
        existing = await self.vector_store.get(memory_id)
        if existing is None:
            raise ValueError(
                f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'"
            )
        previous = existing.payload.get("data")
        if data is None:
            data = previous
        if not isinstance(data, str):
            raise ValueError(
                f"Memory with id {memory_id} does not have text content to update"
            )
        payload = deepcopy(existing.payload)
        if update_metadata is not None:
            payload.update(update_metadata)
        payload["data"] = data
        payload["hash"] = hashlib.md5(data.encode()).hexdigest()
        payload["text_lemmatized"] = data
        payload["created_at"] = existing.payload.get("created_at")
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "actor_id" in existing.payload:
            payload["actor_id"] = existing.payload["actor_id"]
        embedding = await self.embedding_model.embed(data, "update")
        await self.vector_store.update(
            memory_id,
            vector=embedding,
            payload=payload,
        )
        await self.db.add_history(
            memory_id,
            previous,
            data,
            "UPDATE",
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            actor_id=payload.get("actor_id"),
            role=payload.get("role"),
        )
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id: str) -> dict[str, str]:
        await self.initialize()
        existing = await self.vector_store.get(memory_id)
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        payload = existing.payload or {}
        await self.vector_store.delete(memory_id)
        await self.db.add_history(
            memory_id,
            payload.get("data", ""),
            None,
            "DELETE",
            created_at=_normalize_timestamp(payload.get("created_at")),
            updated_at=datetime.now(timezone.utc).isoformat(),
            actor_id=payload.get("actor_id"),
            role=payload.get("role"),
            is_deleted=1,
        )
        return {"message": "Memory deleted successfully!"}

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            resources = [
                resource
                for resource in (
                    self.embedding_model,
                    self.llm,
                    self.vector_store,
                    self.db,
                )
                if resource is not None
            ]
            self.embedding_model = None
            self.llm = None
            self.vector_store = None
            self.db = None
            close_tasks = [
                asyncio.create_task(resource.close()) for resource in resources
            ]
            if not close_tasks:
                return
            close_group = asyncio.gather(*close_tasks, return_exceptions=True)
            try:
                results = await asyncio.shield(close_group)
            except asyncio.CancelledError:
                results = await close_group
                for result in results:
                    if isinstance(result, Exception):
                        logger.error("Mem0 resource cleanup failed: %s", result)
                raise
            errors = [result for result in results if isinstance(result, Exception)]
            if errors:
                raise errors[0]
