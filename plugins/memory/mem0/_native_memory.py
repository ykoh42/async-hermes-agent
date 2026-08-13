"""Native-async Mem0 OSS memory orchestration."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, date, datetime
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
import uuid

import aiofiles.os

from agent.secret_scope import (
    UnscopedSecretError,
    current_secret_scope,
    get_secret,
    is_multiplex_active,
)
from hermes_constants import get_hermes_home

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
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
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
_expanduser = aiofiles.os.wrap(os.path.expanduser)
_realpath = aiofiles.os.wrap(os.path.realpath)


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one owned Mem0 cleanup through repeated cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
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


async def _profile_storage_path(*parts: str) -> str:
    """Resolve storage below the canonical active Hermes profile home."""
    expanded_home = await _expanduser(os.fspath(get_hermes_home()))
    canonical_home = await _realpath(expanded_home)
    return str(Path(canonical_home).joinpath(*parts))


async def _resolve_history_db_path(config: dict[str, Any]) -> str:
    """Preserve legacy history defaults outside multiplexed profile mode."""
    history_path = config.get("history_db_path")
    if history_path:
        return str(await _expanduser(str(history_path)))

    mem0_dir = get_secret("MEM0_DIR")
    if mem0_dir:
        expanded_dir = await _expanduser(str(mem0_dir))
        return os.path.join(expanded_dir, "history.db")
    if is_multiplex_active():
        return await _profile_storage_path("mem0", "history.db")

    legacy_dir = await _expanduser(os.path.join("~", ".mem0"))
    return os.path.join(legacy_dir, "history.db")


async def _resolve_vector_config(
    provider: str | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Isolate implicit local Qdrant storage for multiplexed profiles."""
    resolved = dict(config)
    if provider != "qdrant" or not is_multiplex_active():
        return resolved
    if any(key in resolved for key in ("client", "url", "host", "path")):
        return resolved
    if current_secret_scope() is None:
        raise UnscopedSecretError(
            "Implicit Mem0 Qdrant storage requires an active profile secret "
            "scope while multiplexing is enabled."
        )
    resolved["path"] = await _profile_storage_path("mem0_qdrant")
    return resolved


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
    return parsed.astimezone(UTC).isoformat()


def _is_expired(payload: dict[str, Any] | None) -> bool:
    if not payload or not payload.get("expiration_date"):
        return False
    try:
        return date.fromisoformat(str(payload["expiration_date"])) < datetime.now(
            UTC
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


def _vector_rows(listed: Any) -> list[Any]:
    if (
        isinstance(listed, (list, tuple))
        and listed
        and isinstance(listed[0], (list, tuple))
    ):
        return list(listed[0])
    if isinstance(listed, (list, tuple)):
        return list(listed)
    return []


def _validate_top_k(top_k: Any) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be a valid integer")
    if top_k < 0:
        raise ValueError(
            f"Invalid top_k: {top_k}. Must be a non-negative integer."
        )
    return top_k


def _reject_top_level_entity_parameters(
    kwargs: dict[str, Any],
    method_name: str,
) -> None:
    invalid = {"user_id", "agent_id", "run_id"} & kwargs.keys()
    if invalid:
        raise ValueError(
            f"Top-level entity parameters {invalid} are not supported in "
            f"{method_name}(). Use filters={{'user_id': '...'}} instead."
        )


def _memory_item(memory: Any, *, include_score: bool) -> dict[str, Any]:
    payload = memory.payload
    item: dict[str, Any] = {
        "id": str(memory.id),
        "memory": payload.get("data", ""),
        "hash": payload.get("hash"),
        "metadata": None,
    }
    if include_score:
        item["score"] = None
    item["created_at"] = payload.get("created_at")
    item["updated_at"] = payload.get("updated_at")
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
    return item


class _NativeOSSProject:
    async def update(
        self,
        custom_instructions: str | None = None,  # noqa: ARG002
        custom_categories: list[Any] | None = None,  # noqa: ARG002
        retrieval_criteria: list[Any] | None = None,  # noqa: ARG002
        multilingual: bool | None = None,  # noqa: ARG002
        decay: bool | None = None,
    ) -> None:
        if decay is True:
            raise ValueError(
                "The decay parameter is not supported by the OSS Memory SDK."
            )
        raise ValueError("Project updates are not supported by the OSS Memory SDK.")


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
        self._history_db_path: str | None = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        self.api_version = self.config.get("version", "v1.1")
        self.custom_instructions = self.config.get("custom_instructions")

    @property
    def project(self) -> _NativeOSSProject:
        return _NativeOSSProject()

    @classmethod
    async def from_config(cls, config_dict: dict[str, Any]) -> Memory:
        memory = cls(config_dict)
        try:
            await memory.initialize()
        except BaseException:
            await memory.close()
            raise
        return memory

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
                embedder_factory = OpenAIEmbedding
            elif embedder_provider == "ollama":
                embedder_factory = OllamaEmbedding
            else:
                raise ValueError(
                    f"Unsupported native Mem0 embedder provider: {embedder_provider}"
                )

            llm = self.config.get("llm") or {}
            llm_provider = llm.get("provider")
            llm_config = dict(llm.get("config") or {})
            if llm_provider == "openai":
                llm_factory = OpenAILLM
            elif llm_provider == "ollama":
                llm_factory = OllamaLLM
            else:
                raise ValueError(
                    f"Unsupported native Mem0 LLM provider: {llm_provider}"
                )

            vector = self.config.get("vector_store") or {}
            vector_provider = vector.get("provider")
            vector_config = dict(vector.get("config") or {})
            if vector_provider == "qdrant":
                vector_factory = Qdrant
            elif vector_provider == "pgvector":
                vector_factory = PGVector
            else:
                raise ValueError(
                    f"Unsupported native Mem0 vector provider: {vector_provider}"
                )

            history_path = await _resolve_history_db_path(self.config)
            vector_config = await _resolve_vector_config(
                vector_provider,
                vector_config,
            )
            if history_path != ":memory:":
                parent = str(Path(history_path).parent)
                await aiofiles.os.makedirs(parent, exist_ok=True)

            embedding_model = embedder_factory(embedder_config)
            llm_client = llm_factory(llm_config)
            vector_store = vector_factory(vector_config)
            database = SQLiteManager(history_path)
            nlp = NativeNLP()
            entities = NativeEntities(
                vector_provider,
                vector_config,
                embedding_model,
                nlp,
                {"qdrant": Qdrant, "pgvector": PGVector},
                main_store=vector_store,
            )
            try:
                await vector_store._initialize()
                await database._initialize()
            except BaseException:
                async def cleanup_failed_initialize() -> None:
                    await asyncio.gather(
                        embedding_model.close(),
                        llm_client.close(),
                        vector_store.close(),
                        database.close(),
                        entities.close(),
                        nlp.close(),
                        return_exceptions=True,
                    )

                await _finish_owned_task(
                    asyncio.create_task(
                        cleanup_failed_initialize(),
                        name="mem0-memory-initialize-cleanup",
                    )
                )
                raise

            self.embedding_model = embedding_model
            self.llm = llm_client
            self.vector_store = vector_store
            self.db = database
            self._history_db_path = history_path
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
            payload["created_at"] = datetime.now(UTC).isoformat()
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

    async def _create_procedural_memory(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        llm: Any = None,
        prompt: str | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        logger.info("Creating procedural memory")
        parsed_messages = [
            {
                "role": "system",
                "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT,
            },
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]
        try:
            if llm is None:
                procedural_memory = await self.llm.generate_response(
                    messages=parsed_messages
                )
            else:
                ainvoke = getattr(llm, "ainvoke", None)
                if not inspect.iscoroutinefunction(ainvoke):
                    raise TypeError(
                        "Procedural memory LLM override must provide native "
                        "async ainvoke()."
                    )
                response = await ainvoke(input=parsed_messages)
                procedural_memory = response.content
            procedural_memory = _remove_code_blocks(procedural_memory)
        except Exception as exc:
            logger.error("Error generating procedural memory summary: %s", exc)
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")
        procedural_metadata = {
            **metadata,
            "memory_type": "procedural_memory",
        }
        embedding = await self.embedding_model.embed(
            procedural_memory,
            memory_action="add",
        )
        memory_id = await self._create_memory(
            procedural_memory,
            embedding,
            procedural_metadata,
        )
        return {
            "results": [
                {
                    "id": memory_id,
                    "memory": procedural_memory,
                    "event": "ADD",
                }
            ]
        }

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
                    UTC
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
        llm: Any = None,
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

        if agent_id is not None and memory_type == "procedural_memory":
            return await self._create_procedural_memory(
                normalized_messages,
                metadata=base_metadata,
                llm=llm,
                prompt=prompt,
            )
        llm_config = (self.config.get("llm") or {}).get("config") or {}
        vision_llm = self.llm if llm_config.get("enable_vision") else None
        normalized_messages = await _parse_vision_messages(
            normalized_messages,
            vision_llm,
            llm_config.get("vision_details", "auto"),
        )

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

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        await self.initialize()
        memory = await self.vector_store.get(memory_id)
        if memory is None:
            return None
        return _memory_item(memory, include_score=True)

    async def get_all(
        self,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        await self.initialize()
        _reject_top_level_entity_parameters(kwargs, "get_all")
        limit = _validate_top_k(top_k)
        effective_filters = dict(filters) if filters else {}
        for key in ("user_id", "agent_id", "run_id"):
            if key in effective_filters:
                effective_filters[key] = _validate_entity_id(
                    effective_filters[key],
                    key,
                )
        if not any(
            key in effective_filters
            for key in ("user_id", "agent_id", "run_id")
        ):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, "
                "run_id. Example: filters={'user_id': 'u1'}"
            )

        fetch_limit = limit if show_expired else max(limit * 4, 60)
        listed = await self.vector_store.list(
            filters=effective_filters,
            top_k=fetch_limit,
        )
        results = []
        for memory in _vector_rows(listed):
            if not show_expired and _is_expired(memory.payload):
                continue
            results.append(_memory_item(memory, include_score=False))
            if len(results) >= limit:
                break
        return {"results": results}

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
        _reject_top_level_entity_parameters(kwargs, "search")
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValueError("threshold must be a valid number")
            if threshold < 0 or threshold > 1:
                raise ValueError(
                    f"Invalid threshold: {threshold}. Must be between 0 and 1 "
                    "(inclusive)."
                )
        top_k = _validate_top_k(top_k)
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
        payload["updated_at"] = datetime.now(UTC).isoformat()
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

    async def _delete_memory(
        self,
        memory_id: str,
        existing: Any = None,
        *,
        skip_entity_cleanup: bool = False,
    ) -> str:
        if existing is None:
            existing = await self.vector_store.get(memory_id)
        if existing is None:
            raise ValueError(
                f"Memory with id {memory_id} not found. Please provide a "
                "valid 'memory_id'"
            )
        payload = existing.payload or {}
        await self.vector_store.delete(memory_id)
        await self.db.add_history(
            memory_id,
            payload.get("data", ""),
            None,
            "DELETE",
            created_at=_normalize_timestamp(payload.get("created_at")),
            updated_at=datetime.now(UTC).isoformat(),
            actor_id=payload.get("actor_id"),
            role=payload.get("role"),
            is_deleted=1,
        )
        session_filters = {
            key: payload[key]
            for key in ("user_id", "agent_id", "run_id")
            if payload.get(key)
        }
        if not skip_entity_cleanup:
            await self.entities.remove_memory(memory_id, session_filters)
        return memory_id

    async def delete(self, memory_id: str) -> dict[str, str]:
        await self.initialize()
        await self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, str]:
        await self.initialize()
        user_id = _validate_entity_id(user_id, "user_id")
        agent_id = _validate_entity_id(agent_id, "agent_id")
        run_id = _validate_entity_id(run_id, "run_id")
        filters = {
            key: value
            for key, value in (
                ("user_id", user_id),
                ("agent_id", agent_id),
                ("run_id", run_id),
            )
            if value
        }
        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If "
                "you want to delete all memories, use the `reset()` method."
            )

        listed = await self.vector_store.list(filters=filters)
        rows = _vector_rows(listed)
        results = await asyncio.gather(
            *(
                self._delete_memory(
                    str(memory.id),
                    skip_entity_cleanup=True,
                )
                for memory in rows
            ),
            return_exceptions=True,
        )
        await self.entities.clear_scope(filters)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            logger.warning(
                "Failed to delete %d out of %d memories",
                len(errors),
                len(results),
            )
            for error in errors:
                logger.warning("Delete error: %s", error)
        logger.info("Deleted %d memories", len(results) - len(errors))
        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        await self.initialize()
        return await self.db.get_history(memory_id)

    async def reset(self) -> None:
        await self.initialize()
        logger.warning("Resetting all memories")
        await self.vector_store.reset()

        database = self.db
        await database.reset()
        await database.close()
        history_path = self._history_db_path
        assert history_path is not None
        replacement = SQLiteManager(history_path)
        try:
            await replacement._initialize()
        except BaseException:
            await _finish_owned_task(
                asyncio.create_task(
                    replacement.close(),
                    name="mem0-memory-reset-cleanup",
                )
            )
            raise
        self.db = replacement

        try:
            await self.entities.reset()
        except Exception as exc:
            logger.warning("Failed to reset entity store: %s", exc)

    async def chat(self, query: Any) -> None:  # noqa: ARG002
        raise NotImplementedError("Chat function not implemented yet.")

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
            results: list[Any] = []

            async def close_resources() -> None:
                results.extend(
                    await asyncio.gather(*close_tasks, return_exceptions=True)
                )

            cleanup = asyncio.create_task(
                close_resources(),
                name="mem0-memory-close",
            )
            try:
                await _finish_owned_task(cleanup)
            except asyncio.CancelledError:
                for result in results:
                    if isinstance(result, Exception):
                        logger.error("Mem0 resource cleanup failed: %s", result)
                raise
            errors = [result for result in results if isinstance(result, Exception)]
            if errors:
                raise errors[0]
