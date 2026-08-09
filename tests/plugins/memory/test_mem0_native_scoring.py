"""Parity tests for Mem0's pure hybrid-scoring transformations."""

from __future__ import annotations

import pytest

from plugins.memory.mem0._native_scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)


@pytest.mark.parametrize(
    ("query", "lemmatized"),
    [
        ("tea", "tea"),
        ("one two three four", "one two three four"),
        ("seven term query", "one two three four five six seven"),
        ("ten term query", "one two three four five six seven eight nine ten"),
        ("long query", " ".join(str(index) for index in range(16))),
        ("empty", ""),
    ],
)
def test_bm25_parameters_are_exactly_upstream(query, lemmatized):
    from mem0.utils.scoring import get_bm25_params as upstream_get_bm25_params

    assert get_bm25_params(query, lemmatized=lemmatized) == (
        upstream_get_bm25_params(query, lemmatized=lemmatized)
    )


@pytest.mark.parametrize("raw_score", [-4.0, 0.0, 5.0, 20.0])
def test_bm25_normalization_is_exactly_upstream(raw_score):
    from mem0.utils.scoring import normalize_bm25 as upstream_normalize_bm25

    assert normalize_bm25(raw_score, 5.0, 0.7) == upstream_normalize_bm25(
        raw_score,
        5.0,
        0.7,
    )


@pytest.mark.parametrize(
    ("bm25_scores", "entity_boosts"),
    [
        ({}, {}),
        ({"m1": 0.5}, {}),
        ({}, {"m1": 0.4}),
        ({"m1": 0.5}, {"m1": 0.4}),
    ],
)
def test_additive_scoring_is_exactly_upstream(bm25_scores, entity_boosts):
    from mem0.utils.scoring import score_and_rank as upstream_score_and_rank

    semantic = [
        {"id": "m1", "score": 0.4, "payload": {"data": "first"}},
        {"id": "m2", "score": 0.8, "payload": {"data": "second"}},
        {"id": "below", "score": 0.05, "payload": {}},
        {"score": 1.0, "payload": {}},
    ]
    kwargs = {
        "semantic_results": semantic,
        "bm25_scores": bm25_scores,
        "entity_boosts": entity_boosts,
        "threshold": 0.1,
        "top_k": 3,
        "explain": True,
    }

    assert score_and_rank(**kwargs) == upstream_score_and_rank(**kwargs)
    assert ENTITY_BOOST_WEIGHT == 0.5
