"""Behavior parity for the vendored Mem0 2.0.10 Qdrant filter builder."""

from __future__ import annotations

import ast

import pytest
from qdrant_client import models

from mem0.vector_stores.qdrant import Qdrant as UpstreamQdrant
from plugins.memory.mem0._native_vector import Qdrant


_SUPPORTED_OPERATORS_SEPARATOR = ". Supported operators: "


def _error_signature(message: str) -> tuple[str, frozenset[str] | None]:
    """Compare the stable error text and supported operators separately."""

    prefix, separator, operator_repr = message.rpartition(
        _SUPPORTED_OPERATORS_SEPARATOR
    )
    if not separator:
        return message, None

    operators = ast.literal_eval(operator_repr)
    assert isinstance(operators, set)
    assert all(isinstance(operator, str) for operator in operators)
    return prefix, frozenset(operators)


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

    assert _error_signature(str(actual.value)) == _error_signature(
        str(expected.value)
    )


def test_qdrant_supported_operator_error_ignores_set_display_order():
    prefix = "Unsupported filter operator(s) for field 'priority': {'unknown'}"

    assert _error_signature(
        f"{prefix}{_SUPPORTED_OPERATORS_SEPARATOR}{{'eq', 'ne'}}"
    ) == _error_signature(
        f"{prefix}{_SUPPORTED_OPERATORS_SEPARATOR}{{'ne', 'eq'}}"
    )
