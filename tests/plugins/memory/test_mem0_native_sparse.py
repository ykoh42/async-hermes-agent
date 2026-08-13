"""Tests for the native-async Mem0 BM25 subprocess transform."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import ModuleType
import sys

import pytest

from plugins.memory.mem0 import _native_worker, _transform_worker
from plugins.memory.mem0._native_sparse import NativeSparseEncoder


class _Array:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _Sparse:
    def __init__(self, indices, values):
        self.indices = _Array(indices)
        self.values = _Array(values)


@pytest.fixture
def fake_fastembed_package(tmp_path, monkeypatch):
    package = tmp_path / "fastembed"
    package.mkdir()
    (package / "__init__.py").write_text(
        """
class _Array:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _Sparse:
    def __init__(self, index):
        self.indices = _Array([index])
        self.values = _Array([index / 10])


class SparseTextEmbedding:
    def __init__(self, *, model_name):
        assert model_name == "Qdrant/bm25"

    def embed(self, texts):
        return [_Sparse(index) for index, _ in enumerate(texts)]
""".lstrip()
    )
    monkeypatch.syspath_prepend(tmp_path)
    existing = os.environ.get("PYTHONPATH")
    python_path = str(tmp_path)
    if existing:
        python_path = f"{python_path}{os.pathsep}{existing}"
    monkeypatch.setenv("PYTHONPATH", python_path)


def test_transform_worker_loads_pinned_fastembed_model_and_preserves_order(
    monkeypatch,
):
    calls = []

    class SparseTextEmbedding:
        def __init__(self, *, model_name):
            calls.append(("initialize", model_name))

        def embed(self, texts):
            calls.append(("embed", list(texts)))
            return [_Sparse([index], [index / 10]) for index, _ in enumerate(texts)]

    fastembed = ModuleType("fastembed")
    fastembed.SparseTextEmbedding = SparseTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(_transform_worker, "_bm25_encoder", None)
    monkeypatch.setattr(_transform_worker, "_bm25_unavailable", False)

    first = _transform_worker._execute({
        "operation": "encode_bm25_batch",
        "texts": ["one", "two"],
    })
    second = _transform_worker._execute({
        "operation": "encode_bm25_batch",
        "texts": ["three"],
    })

    assert first == [
        {"indices": [0], "values": [0.0]},
        {"indices": [1], "values": [0.1]},
    ]
    assert second == [{"indices": [0], "values": [0.0]}]
    assert calls == [
        ("initialize", "Qdrant/bm25"),
        ("embed", ["one", "two"]),
        ("embed", ["three"]),
    ]


def test_transform_worker_caches_fastembed_initialization_failure(monkeypatch):
    initialization_count = 0

    class SparseTextEmbedding:
        def __init__(self, *, model_name):
            nonlocal initialization_count
            initialization_count += 1
            raise RuntimeError(model_name)

    fastembed = ModuleType("fastembed")
    fastembed.SparseTextEmbedding = SparseTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(_transform_worker, "_bm25_encoder", None)
    monkeypatch.setattr(_transform_worker, "_bm25_unavailable", False)

    request = {"operation": "encode_bm25_batch", "texts": ["one"]}
    assert _transform_worker._execute(request) is None
    assert _transform_worker._execute(request) is None
    assert initialization_count == 1


@pytest.mark.asyncio
async def test_sparse_encoder_absence_preserves_upstream_fallback_without_process(
    monkeypatch,
):
    async def locate(module_name):
        assert module_name == "fastembed"
        return None

    async def forbidden_spawn(*args, **kwargs):
        raise AssertionError("BM25 subprocess must not start without fastembed")

    monkeypatch.setattr(_native_worker, "_locate_source_module", locate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    encoder = NativeSparseEncoder()

    assert await encoder.encode_batch(["one", "two"]) is None
    await encoder.close()


@pytest.mark.asyncio
async def test_sparse_encoder_runs_end_to_end_through_owned_subprocess(
    fake_fastembed_package,
):
    encoder = NativeSparseEncoder()

    assert encoder._worker._process is None
    assert await encoder.encode_batch(["one", "two"]) == [
        ([0], [0.0]),
        ([1], [0.1]),
    ]
    process = encoder._worker._process
    assert process is not None

    await encoder.close()

    assert process.returncode == 0


@pytest.mark.asyncio
async def test_sparse_encoder_validates_worker_result(monkeypatch):
    class Worker:
        def __init__(self, dependency):
            assert dependency == "fastembed"

        async def request(self, operation, *, fallback, **payload):
            assert operation == "encode_bm25_batch"
            assert fallback is None
            assert payload == {"texts": ["one"]}
            return [{"indices": [1, 2], "values": [0.25, 0.75]}]

        async def close(self):
            return None

    from plugins.memory.mem0 import _native_sparse

    monkeypatch.setattr(_native_sparse, "NativeWorker", Worker)
    encoder = _native_sparse.NativeSparseEncoder()

    assert await encoder.encode_batch(["one"]) == [([1, 2], [0.25, 0.75])]
    await encoder.close()


def test_transform_worker_path_is_not_resolved_from_working_directory():
    worker_path = Path(_native_worker.__file__).with_name("_transform_worker.py")

    assert worker_path.is_absolute()
    assert worker_path.is_file()
