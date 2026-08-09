"""Native-async Mem0 OSS memory orchestration."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
import uuid

import aiofiles.os

from ._native_entities import NativeEntities
from ._native_nlp import NativeNLP
from ._native_oss import (
    OllamaEmbedding,
    OllamaLLM,
    OpenAIEmbedding,
    OpenAILLM,
    SQLiteManager,
    _extract_json,
)
from ._native_prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    generate_additive_extraction_prompt,
)
from ._native_scoring import get_bm25_params, normalize_bm25, score_and_rank
from ._native_vector import PGVector, Qdrant

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


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy mutable config blocks while preserving runtime client objects."""
    copied: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"embedder", "llm", "vector_store"} and isinstance(value, dict):
            block = dict(value)
            provider_config = value.get("config")
            if isinstance(provider_config, dict):
                block["config"] = dict(provider_config)
            copied[key] = block
            continue
        try:
            copied[key] = deepcopy(value)
        except Exception:
            copied[key] = value
    return copied


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


def _parse_messages(messages: list[dict[str, Any]]) -> str:
    parsed = ""
    for message in messages:
        content = message.get("content")
        if content is None:
            continue
        role = message.get("role")
        if role in {"system", "user", "assistant"}:
            parsed += f"{role}: {content}\n"
    return parsed


def _remove_code_blocks(content: str) -> str:
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    inner = match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", inner, flags=re.DOTALL).strip()


def _build_session_scope(filters: dict[str, Any]) -> str:
    parts = [
        f"{key}={filters[key]}"
        for key in sorted(("user_id", "agent_id", "run_id"))
        if filters.get(key)
    ]
    return "&".join(parts)


async def _describe_image(
    image: str | dict[str, Any],
    llm: Any,
    vision_details: str,
) -> str:
    if isinstance(image, str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "A user is providing an image. Provide a high level "
                            "description of the image and do not include any "
                            "additional text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                            "detail": vision_details,
                        },
                    },
                ],
            }
        ]
    else:
        messages = [image]
    return await llm.generate_response(messages=messages)


async def _parse_vision_messages(
    messages: list[dict[str, Any]],
    llm: Any = None,
    vision_details: str = "auto",
) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            normalized.append(message)
            continue
        if content is None:
            continue
        if isinstance(content, list):
            if llm is None:
                text_parts = [
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if text_parts:
                    normalized.append(
                        {"role": role, "content": " ".join(text_parts)}
                    )
            else:
                normalized.append(
                    {
                        "role": role,
                        "content": await _describe_image(
                            message,
                            llm,
                            vision_details,
                        ),
                    }
                )
        elif isinstance(content, dict) and content.get("type") == "image_url":
            if llm is None:
                continue
            image_url = content.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not url:
                raise ValueError("image_url content part is missing image_url.url")
            try:
                description = await _describe_image(url, llm, vision_details)
            except Exception as exc:
                raise Exception(f"Error while downloading {url}.") from exc
            normalized.append({"role": role, "content": description})
        else:
            normalized.append(message)
    return normalized


class Memory:
    """Retained Mem0 OSS runtime with native coroutine boundaries."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = _copy_config(config)
        self.embedding_model: Any = None
        self.llm: Any = None
        self.vector_store: Any = None
        self.db: Any = None
        self.nlp: Any = None
        self.entities: Any = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        self.api_version = self.config.get("version", "v1.1")
        self.custom_instructions = self.config.get("custom_instructions")

    async def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot initialize a closed Mem0 Memory")
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._closed:
                raise RuntimeError("Cannot initialize a closed Mem0 Memory")
            if self._initialized:
                return

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
                vector_store = PGVector(vector_config)
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
            nlp = NativeNLP()
            entities = NativeEntities(
                vector_provider,
                vector_config,
                embedding_model,
                nlp,
                {"qdrant": Qdrant, "pgvector": PGVector},
            )
            try:
                await vector_store._initialize()
                await database._initialize()
            except BaseException:
                await asyncio.gather(
                    vector_store.close(),
                    database.close(),
                    entities.close(),
                    nlp.close(),
                    return_exceptions=True,
                )
                raise

            self.embedding_model = embedding_model
            self.llm = llm_client
            self.vector_store = vector_store
            self.db = database
            self.nlp = nlp
            self.entities = entities
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
        payload["text_lemmatized"] = await self.nlp.lemmatize(text)
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

    async def _infer_memories(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
        filters: dict[str, Any],
        prompt: str | None,
    ) -> list[dict[str, str]]:
        session_scope = _build_session_scope(filters)
        last_messages = await self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = _parse_messages(messages)

        search_filters = {
            key: value
            for key, value in filters.items()
            if key in {"user_id", "agent_id", "run_id"} and value
        }
        query_embedding = await self.embedding_model.embed(
            parsed_messages,
            "search",
        )
        existing_results = await self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )
        existing_memories = [
            {"id": str(index), "text": memory.payload.get("data", "")}
            for index, memory in enumerate(existing_results)
        ]

        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if filters.get("agent_id") and not filters.get("user_id"):
            system_prompt += AGENT_CONTEXT_SUFFIX
        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=prompt or self.custom_instructions,
        )
        try:
            response = await self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("LLM extraction failed: %s", exc)
            return []

        try:
            response = _remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get(
                        "memory",
                        [],
                    )
                except json.JSONDecodeError:
                    extracted_memories = json.loads(
                        _extract_json(response),
                        strict=False,
                    ).get("memory", [])
        except Exception as exc:
            logger.error("Error parsing extraction response: %s", exc)
            extracted_memories = []

        if not extracted_memories:
            await self.db.save_messages(messages, session_scope)
            return []

        memory_texts = [
            memory.get("text", "")
            for memory in extracted_memories
            if memory.get("text")
        ]
        try:
            embeddings = await self.embedding_model.embed_batch(
                memory_texts,
                "add",
            )
            embedding_map = dict(zip(memory_texts, embeddings, strict=False))
        except Exception:
            embedding_map = {}
            for text in memory_texts:
                try:
                    embedding_map[text] = await self.embedding_model.embed(
                        text,
                        "add",
                    )
                except Exception as exc:
                    logger.warning("Failed to embed memory text: %s", exc)

        existing_hashes = {
            memory.payload.get("hash")
            for memory in existing_results
            if getattr(memory, "payload", None) and memory.payload.get("hash")
        }
        records: list[tuple[str, str, list[float], dict[str, Any]]] = []
        seen_hashes: set[str] = set()
        for memory in extracted_memories:
            text = memory.get("text")
            if not text or text not in embedding_map:
                continue
            memory_hash = hashlib.md5(text.encode()).hexdigest()
            if memory_hash in existing_hashes or memory_hash in seen_hashes:
                logger.debug("Skipping duplicate memory (hash match): %s", text[:50])
                continue
            seen_hashes.add(memory_hash)

            memory_id = str(uuid.uuid4())
            memory_metadata = deepcopy(metadata)
            memory_metadata["data"] = text
            memory_metadata["text_lemmatized"] = await self.nlp.lemmatize(text)
            memory_metadata["hash"] = memory_hash
            if "created_at" not in memory_metadata:
                memory_metadata["created_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
            memory_metadata["updated_at"] = memory_metadata["created_at"]
            if memory.get("attributed_to"):
                memory_metadata["attributed_to"] = memory["attributed_to"]
            records.append(
                (memory_id, text, embedding_map[text], memory_metadata)
            )

        if not records:
            await self.db.save_messages(messages, session_scope)
            return []

        vectors = [record[2] for record in records]
        memory_ids = [record[0] for record in records]
        payloads = [record[3] for record in records]
        try:
            await self.vector_store.insert(
                vectors=vectors,
                ids=memory_ids,
                payloads=payloads,
            )
        except Exception:
            for memory_id, vector, payload in zip(
                memory_ids,
                vectors,
                payloads,
                strict=True,
            ):
                try:
                    await self.vector_store.insert(
                        vectors=[vector],
                        ids=[memory_id],
                        payloads=[payload],
                    )
                except Exception as exc:
                    logger.error("Failed to insert memory %s: %s", memory_id, exc)

        history_records = [
            {
                "memory_id": record[0],
                "old_memory": None,
                "new_memory": record[1],
                "event": "ADD",
                "created_at": record[3].get("created_at"),
                "is_deleted": 0,
            }
            for record in records
        ]
        try:
            await self.db.batch_add_history(history_records)
        except Exception:
            for history in history_records:
                try:
                    await self.db.add_history(
                        history["memory_id"],
                        None,
                        history["new_memory"],
                        "ADD",
                        created_at=history.get("created_at"),
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to add history for %s: %s",
                        history["memory_id"],
                        exc,
                    )

        await self.entities.link_batch(records, search_filters)
        await self.db.save_messages(messages, session_scope)
        return [
            {"id": record[0], "memory": record[1], "event": "ADD"}
            for record in records
        ]

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
        prompt: str | None = None,
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
        llm_config = (self.config.get("llm") or {}).get("config") or {}
        vision_llm = self.llm if llm_config.get("enable_vision") else None
        normalized_messages = await _parse_vision_messages(
            normalized_messages,
            vision_llm,
            llm_config.get("vision_details", "auto"),
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

        if infer:
            return {
                "results": await self._infer_memories(
                    normalized_messages,
                    base_metadata,
                    {
                        key: value
                        for key, value in (
                            ("user_id", user_id),
                            ("agent_id", agent_id),
                            ("run_id", run_id),
                        )
                        if value
                    },
                    prompt,
                )
            }

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

        query_lemmatized = await self.nlp.lemmatize(query)
        query_entities = await self.nlp.extract(query)
        embedding = await self.embedding_model.embed(query, "search")
        internal_limit = max(top_k * 4, 60)
        semantic_results = await self.vector_store.search(
            query=query,
            vectors=embedding,
            top_k=internal_limit,
            filters=effective_filters,
        )
        keyword_results = await self.vector_store.keyword_search(
            query=query_lemmatized,
            top_k=internal_limit,
            filters=effective_filters,
        )

        bm25_scores: dict[str, float] = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(
                query,
                lemmatized=query_lemmatized,
            )
            for memory in keyword_results:
                memory_id = str(getattr(memory, "id", ""))
                raw_score = getattr(memory, "score", 0.0)
                if raw_score and raw_score > 0:
                    bm25_scores[memory_id] = normalize_bm25(
                        raw_score,
                        midpoint,
                        steepness,
                    )
        entity_boosts = (
            await self.entities.boosts(query_entities, effective_filters)
            if query_entities
            else {}
        )
        candidates = []
        for memory in semantic_results:
            payload = getattr(memory, "payload", None) or {}
            if not show_expired and _is_expired(payload):
                continue
            candidates.append(
                {
                    "id": str(memory.id),
                    "score": getattr(memory, "score", None) or 0.0,
                    "payload": payload,
                }
            )
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=top_k,
            explain=explain,
        )

        results = []
        for candidate in scored_results:
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
            if explain and "score_details" in candidate:
                item["score_details"] = candidate["score_details"]
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
        text_changed = data != previous
        payload = deepcopy(existing.payload)
        if update_metadata is not None:
            payload.update(update_metadata)
        payload["data"] = data
        payload["hash"] = hashlib.md5(data.encode()).hexdigest()
        payload["text_lemmatized"] = await self.nlp.lemmatize(data)
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
        if text_changed:
            session_filters = {
                key: payload[key]
                for key in ("user_id", "agent_id", "run_id")
                if payload.get(key)
            }
            await self.entities.remove_memory(memory_id, session_filters)
            await self.entities.link_memory(memory_id, data, session_filters)
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
        session_filters = {
            key: payload[key]
            for key in ("user_id", "agent_id", "run_id")
            if payload.get(key)
        }
        await self.entities.remove_memory(memory_id, session_filters)
        return {"message": "Memory deleted successfully!"}

    async def _get_entity_store(self) -> Any:
        await self.initialize()
        return await self.entities.get_store()

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
                    self.entities,
                    self.nlp,
                )
                if resource is not None
            ]
            self.embedding_model = None
            self.llm = None
            self.vector_store = None
            self.db = None
            self.entities = None
            self.nlp = None
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
