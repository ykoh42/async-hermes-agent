"""Native-async boundary for Mem0's optional synchronous spaCy pipeline."""

from __future__ import annotations

from typing import Any

from ._native_worker import NativeWorker


def _entities(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise RuntimeError("Mem0 NLP worker returned invalid entity data")
    entities: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError("Mem0 NLP worker returned invalid entity data")
        entity_type, entity_text = item
        if not isinstance(entity_type, str) or not isinstance(entity_text, str):
            raise RuntimeError("Mem0 NLP worker returned invalid entity data")
        entities.append((entity_type, entity_text))
    return entities


class NativeNLP:
    """Own one stdio worker so spaCy I/O never blocks the caller's loop."""

    def __init__(self) -> None:
        self._worker = NativeWorker("spacy")

    async def lemmatize(self, text: str) -> str:
        result = await self._worker.request(
            "lemmatize",
            text=text,
            fallback=text,
        )
        if not isinstance(result, str):
            raise RuntimeError("Mem0 NLP worker returned invalid lemma data")
        return result

    async def extract(self, text: str) -> list[tuple[str, str]]:
        return _entities(
            await self._worker.request("extract", text=text, fallback=[])
        )

    async def extract_batch(
        self,
        texts: list[str],
    ) -> list[list[tuple[str, str]]]:
        result = await self._worker.request(
            "extract_batch",
            texts=texts,
            fallback=[[] for _ in texts],
        )
        if not isinstance(result, list):
            raise RuntimeError("Mem0 NLP worker returned invalid batch data")
        return [_entities(item) for item in result]

    async def close(self) -> None:
        await self._worker.close()
