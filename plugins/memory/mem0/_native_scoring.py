"""Pure scoring helpers vendored from Mem0 2.0.10."""

from __future__ import annotations

import math
from typing import Any

ENTITY_BOOST_WEIGHT = 0.5


def get_bm25_params(
    query: str,  # noqa: ARG001 - retained upstream signature
    *,
    lemmatized: str,
) -> tuple[float, float]:
    num_terms = len(lemmatized.split()) if lemmatized else 1
    if num_terms <= 3:
        return 5.0, 0.7
    if num_terms <= 6:
        return 7.0, 0.6
    if num_terms <= 9:
        return 9.0, 0.5
    if num_terms <= 15:
        return 10.0, 0.5
    return 12.0, 0.5


def normalize_bm25(
    raw_score: float,
    midpoint: float,
    steepness: float,
) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def score_and_rank(
    semantic_results: list[dict[str, Any]],
    bm25_scores: dict[str, float],
    entity_boosts: dict[str, float],
    threshold: float,
    top_k: int,
    explain: bool = False,
) -> list[dict[str, Any]]:
    max_possible = 1.0
    if bm25_scores:
        max_possible += 1.0
    if entity_boosts:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: list[dict[str, Any]] = []
    for result in semantic_results:
        memory_id = result.get("id")
        if memory_id is None:
            continue
        semantic_score = result.get("score") or 0.0
        if semantic_score < threshold:
            continue
        memory_id = str(memory_id)
        bm25_score = bm25_scores.get(memory_id, 0.0)
        entity_boost = entity_boosts.get(memory_id, 0.0)
        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)
        scored_result: dict[str, Any] = {
            "id": memory_id,
            "score": combined,
            "payload": result.get("payload"),
        }
        if explain:
            scored_result["score_details"] = {
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "entity_boost": entity_boost,
                "raw_score": raw_combined,
                "max_possible_score": max_possible,
                "final_score": combined,
                "threshold": threshold,
            }
        scored.append(scored_result)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
