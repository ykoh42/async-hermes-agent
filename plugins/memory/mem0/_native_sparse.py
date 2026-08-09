"""Native-async boundary for Mem0's optional fastembed BM25 encoder."""

from __future__ import annotations

from typing import Any

from ._native_worker import NativeWorker

SparseEncoding = tuple[list[int], list[float]]


def _encoding(value: Any) -> SparseEncoding:
    if not isinstance(value, dict):
        raise RuntimeError("Mem0 BM25 worker returned invalid sparse data")
    indices = value.get("indices")
    values = value.get("values")
    if not isinstance(indices, list) or not all(
        isinstance(index, int) for index in indices
    ):
        raise RuntimeError("Mem0 BM25 worker returned invalid sparse indices")
    if not isinstance(values, list) or not all(
        isinstance(score, (int, float)) for score in values
    ):
        raise RuntimeError("Mem0 BM25 worker returned invalid sparse values")
    if len(indices) != len(values):
        raise RuntimeError("Mem0 BM25 worker returned mismatched sparse data")
    return indices, [float(score) for score in values]


class NativeSparseEncoder:
    """Encode BM25 vectors outside the event loop in an owned subprocess."""

    def __init__(self) -> None:
        self._worker = NativeWorker("fastembed")

    async def encode_batch(
        self,
        texts: list[str],
    ) -> list[SparseEncoding] | None:
        result = await self._worker.request(
            "encode_bm25_batch",
            texts=texts,
            fallback=None,
        )
        if result is None:
            return None
        if not isinstance(result, list):
            raise RuntimeError("Mem0 BM25 worker returned invalid batch data")
        return [_encoding(item) for item in result]

    async def close(self) -> None:
        await self._worker.close()
