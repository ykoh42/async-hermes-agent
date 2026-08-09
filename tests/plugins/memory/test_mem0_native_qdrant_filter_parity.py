"""Behavior parity for the vendored Mem0 2.0.10 Qdrant filter builder."""

from __future__ import annotations

import pytest
from qdrant_client import models

from mem0.vector_stores.qdrant import Qdrant as UpstreamQdrant
from plugins.memory.mem0._native_vector import Qdrant


def _builders():
    upstream = UpstreamQdrant.__new__(UpstreamQdrant)
    native = Qdrant({"url": "https://qdrant.test"})
    native._models = models
    return upstream, native


@pytest.mark.parametrize(
    "filters",
    [
        {"user_id": "u1"},
        {"role": ["user", "assistant"]},
        {"status": {"eq": "open"}},
        {"status": {"ne": "closed"}},
        {"tag": {"in": ["a", "b"]}},
        {"tag": {"nin": ["a", "b"]}},
        {"priority": {"gte": 2, "lt": 10}},
        {"created_at": {"gte": "2026-08-01", "lt": "2026-09-01"}},
        {"title": {"contains": "Hermes"}},
        {"title": {"icontains": "hermes"}},
        {"source": "*"},
        {
            "$and": [{"user_id": "u1"}],
            "$or": [{"role": "user"}, {"role": "assistant"}],
            "$not": [{"status": "deleted"}],
        },
    ],
)
def test_native_qdrant_filter_matches_pinned_upstream(filters):
    upstream, native = _builders()

    expected = upstream._create_filter(filters)
    actual = native._create_filter(filters)

    if expected is None:
        assert actual is None
    else:
        assert actual.model_dump(mode="json") == expected.model_dump(mode="json")


@pytest.mark.parametrize(
    "filters",
    [
        {"OR": {"user_id": "u1"}},
        {"AND": ["not-a-filter"]},
        {"priority": {"gte": 2, "eq": 3}},
        {"priority": {"unknown": 3}},
    ],
)
def test_native_qdrant_filter_errors_match_pinned_upstream(filters):
    upstream, native = _builders()

    with pytest.raises(Exception) as expected:
        upstream._create_filter(filters)
    with pytest.raises(type(expected.value)) as actual:
        native._create_filter(filters)

    assert str(actual.value) == str(expected.value)
