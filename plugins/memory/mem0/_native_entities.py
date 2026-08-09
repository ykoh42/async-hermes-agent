"""Native-async Mem0 2.0.10 entity-linking pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import uuid

from ._native_scoring import ENTITY_BOOST_WEIGHT
logger = logging.getLogger(__name__)

_SCOPE_KEYS = ("user_id", "agent_id", "run_id")


def _scope(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: filters[key] for key in _SCOPE_KEYS if filters.get(key)}


def _rows(listed: Any) -> list[Any]:
    if (
        isinstance(listed, (list, tuple))
        and listed
        and isinstance(listed[0], list)
    ):
        return listed[0]
    if isinstance(listed, (list, tuple)):
        return list(listed)
    return []


class NativeEntities:
    """Own the lazy entity vector store and its linking operations."""

    def __init__(
        self,
        provider: str,
        vector_config: dict[str, Any],
        embedding_model: Any,
        nlp: Any,
        store_classes: dict[str, Any],
    ) -> None:
        self._provider = provider
        self._vector_config = dict(vector_config)
        self._embedding_model = embedding_model
        self._nlp = nlp
        self._store_classes = store_classes
        self._store: Any = None
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())

    async def get_store(self) -> Any:
        if self._closed:
            raise RuntimeError("Cannot use a closed Mem0 entity store")
        if self._store is not None:
            return self._store
        async with self._initialize_lock:
            if self._closed:
                raise RuntimeError("Cannot use a closed Mem0 entity store")
            if self._store is not None:
                return self._store
            config = dict(self._vector_config)
            collection = config.get("collection_name", "mem0")
            separator = "-" if self._provider == "s3_vectors" else "_"
            config["collection_name"] = f"{collection}{separator}entities"
            store_class = self._store_classes.get(self._provider)
            if store_class is None:
                raise ValueError(
                    f"Unsupported native Mem0 vector provider: {self._provider}"
                )
            store = store_class(config)
            try:
                await store._initialize()
            except BaseException:
                await asyncio.gather(store.close(), return_exceptions=True)
                raise
            self._store = store
            return store

    async def _existing_by_text(
        self,
        store: Any,
        filters: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            listed = await store.list(filters=filters, top_k=10000)
        except Exception as exc:
            logger.debug(
                "Exact entity lookup failed, falling back to semantic dedup: %s",
                exc,
            )
            return {}
        by_text: dict[str, Any] = {}
        for row in _rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self.normalize(text)
            if normalized and normalized not in by_text:
                by_text[normalized] = row
        return by_text

    async def link_batch(
        self,
        records: list[tuple[str, str, list[float], dict[str, Any]]],
        filters: dict[str, Any],
    ) -> None:
        try:
            texts = [record[1] for record in records]
            extracted = await self._nlp.extract_batch(texts)
            entities: dict[str, list[Any]] = {}
            for index, record in enumerate(records):
                memory_id = record[0]
                for entity_type, entity_text in (
                    extracted[index] if index < len(extracted) else []
                ):
                    key = self.normalize(entity_text)
                    if key in entities:
                        entities[key][2].add(memory_id)
                    else:
                        entities[key] = [entity_type, entity_text, {memory_id}]
            if not entities:
                return

            ordered_keys = list(entities)
            entity_texts = [entities[key][1] for key in ordered_keys]
            try:
                embeddings = await self._embedding_model.embed_batch(
                    entity_texts,
                    "add",
                )
            except Exception:
                embeddings = []
                for text in entity_texts:
                    try:
                        embeddings.append(
                            await self._embedding_model.embed(text, "add")
                        )
                    except Exception:
                        embeddings.append(None)
            if len(embeddings) != len(ordered_keys):
                logger.warning(
                    "embed_batch returned %d vectors for %d entity texts — "
                    "padding/truncating to avoid dropping entity links",
                    len(embeddings),
                    len(ordered_keys),
                )
                embeddings = list(embeddings[: len(ordered_keys)])
                embeddings += [None] * (len(ordered_keys) - len(embeddings))

            valid = [
                (index, key)
                for index, key in enumerate(ordered_keys)
                if embeddings[index] is not None
            ]
            if not valid:
                return
            store = await self.get_store()
            valid_indices, valid_keys = zip(*valid, strict=True)
            vectors = [embeddings[index] for index in valid_indices]
            search_filters = _scope(filters)
            exact = await self._existing_by_text(store, search_filters)
            valid_texts = [entities[key][1] for key in valid_keys]
            matches = await store.search_batch(
                queries=valid_texts,
                vectors_list=vectors,
                top_k=1,
                filters=search_filters,
            )

            insert_vectors = []
            insert_ids = []
            insert_payloads = []
            for index, key in enumerate(valid_keys):
                entity_type, entity_text, memory_ids = entities[key]
                candidates = matches[index] if index < len(matches) else []
                semantic = (
                    candidates[0]
                    if candidates and candidates[0].score >= 0.95
                    else None
                )
                match = exact.get(key) or semantic
                if match:
                    payload = match.payload or {}
                    linked = set(payload.get("linked_memory_ids", []))
                    linked |= memory_ids
                    payload["linked_memory_ids"] = sorted(linked)
                    try:
                        await store.update(
                            match.id,
                            vector=None,
                            payload=payload,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Entity update failed for '%s': %s",
                            entity_text,
                            exc,
                        )
                    continue
                insert_vectors.append(vectors[index])
                insert_ids.append(str(uuid.uuid4()))
                insert_payloads.append(
                    {
                        "data": entity_text,
                        "entity_type": entity_type,
                        "linked_memory_ids": sorted(memory_ids),
                        **search_filters,
                    }
                )
            if insert_vectors:
                try:
                    await store.insert(
                        vectors=insert_vectors,
                        ids=insert_ids,
                        payloads=insert_payloads,
                    )
                except Exception as exc:
                    logger.warning("Batch entity insert failed: %s", exc)
        except Exception as exc:
            logger.warning("Batch entity linking failed: %s", exc)

    async def _upsert(
        self,
        entity_text: str,
        entity_type: str,
        memory_id: str,
        filters: dict[str, Any],
    ) -> None:
        try:
            embedding = await self._embedding_model.embed(entity_text, "add")
            search_filters = _scope(filters)
            store = await self.get_store()
            exact = (
                await self._existing_by_text(store, search_filters)
            ).get(self.normalize(entity_text))
            existing = []
            if exact is None:
                existing = await store.search(
                    query=entity_text,
                    vectors=embedding,
                    top_k=1,
                    filters=search_filters,
                )
            semantic = (
                existing[0]
                if existing and existing[0].score >= 0.95
                else None
            )
            match = exact or semantic
            if match:
                payload = match.payload or {}
                linked = payload.get("linked_memory_ids", [])
                if memory_id not in linked:
                    linked.append(memory_id)
                    payload["linked_memory_ids"] = linked
                    await store.update(match.id, vector=None, payload=payload)
                return
            await store.insert(
                vectors=[embedding],
                ids=[str(uuid.uuid4())],
                payloads=[
                    {
                        "data": entity_text,
                        "entity_type": entity_type,
                        "linked_memory_ids": [memory_id],
                        **search_filters,
                    }
                ],
            )
        except Exception as exc:
            logger.warning("Entity upsert failed for '%s': %s", entity_text, exc)

    async def link_memory(
        self,
        memory_id: str,
        text: str,
        filters: dict[str, Any],
    ) -> None:
        try:
            extracted = await self._nlp.extract(text)
            seen: set[str] = set()
            for entity_type, entity_text in extracted:
                key = self.normalize(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                await self._upsert(
                    entity_text,
                    entity_type,
                    memory_id,
                    filters,
                )
        except Exception as exc:
            logger.warning("Entity linking failed for memory_id=%s: %s", memory_id, exc)

    async def remove_memory(
        self,
        memory_id: str,
        filters: dict[str, Any],
    ) -> None:
        if self._store is None:
            return
        try:
            listed = await self._store.list(filters=_scope(filters), top_k=10000)
            for row in _rows(listed):
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [item for item in linked if item != memory_id]
                    if not remaining:
                        await self._store.delete(row.id)
                        continue
                    text = payload.get("data")
                    if not isinstance(text, str) or not text:
                        logger.debug(
                            "Entity id=%s missing 'data'; skipping cleanup",
                            row.id,
                        )
                        continue
                    vector = await self._embedding_model.embed(text, "update")
                    await self._store.update(
                        row.id,
                        vector=vector,
                        payload={**payload, "linked_memory_ids": remaining},
                    )
                except Exception as exc:
                    logger.debug("Entity cleanup error: %s", exc)
        except Exception as exc:
            logger.warning(
                "Entity store cleanup failed for memory_id=%s: %s",
                memory_id,
                exc,
            )

    async def clear_scope(self, filters: dict[str, Any]) -> None:
        """Delete entity records for one upstream identity scope."""
        if self._store is None:
            return
        try:
            listed = await self._store.list(
                filters=_scope(filters),
                top_k=10000,
            )
            for row in _rows(listed):
                try:
                    await self._store.delete(row.id)
                except Exception as exc:
                    logger.debug(
                        "Bulk entity delete failed for id=%s: %s",
                        row.id,
                        exc,
                    )
        except Exception as exc:
            logger.warning("Bulk entity store cleanup failed: %s", exc)

    async def boosts(
        self,
        query_entities: list[tuple[str, str]],
        filters: dict[str, Any],
    ) -> dict[str, float]:
        seen: set[str] = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self.normalize(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))
        if not deduped:
            return {}

        boosts: dict[str, float] = {}
        try:
            texts = [text for _, text in deduped]
            embeddings = await self._embedding_model.embed_batch(texts, "search")
            if len(embeddings) != len(texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping "
                    "entity boost",
                    len(embeddings),
                    len(texts),
                )
                return boosts
            store = await self.get_store()
            search_filters = _scope(filters)
            semaphore = asyncio.Semaphore(4)

            async def search(text: str, embedding: list[float]) -> Any:
                async with semaphore:
                    return await store.search(
                        query=text,
                        vectors=embedding,
                        top_k=500,
                        filters=search_filters,
                    )

            results = await asyncio.gather(
                *(search(text, embedding) for text, embedding in zip(texts, embeddings, strict=True)),
                return_exceptions=True,
            )
            for matches in results:
                if isinstance(matches, asyncio.CancelledError):
                    raise matches
                if isinstance(matches, Exception):
                    logger.warning(
                        "Entity boost search failed for one entity: %s",
                        matches,
                    )
                    continue
                if isinstance(matches, BaseException):
                    raise matches
                for match in matches:
                    similarity = getattr(match, "score", 0.0)
                    if similarity < 0.5:
                        continue
                    payload = getattr(match, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list):
                        continue
                    count = max(len(linked), 1)
                    weight = 1.0 / (1.0 + 0.001 * ((count - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * weight
                    for memory_id in linked:
                        if memory_id:
                            key = str(memory_id)
                            boosts[key] = max(boosts.get(key, 0.0), boost)
        except Exception as exc:
            logger.warning("Entity boost computation failed: %s", exc)
        return boosts

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._closed = True
            store = self._store
            self._store = None
            if store is not None:
                await store.close()

    async def reset(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed Mem0 entity store")
            store = self._store
            self._store = None
        if store is None:
            return
        try:
            await store.reset()
        finally:
            await store.close()
