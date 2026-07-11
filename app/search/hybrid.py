"""Hybrid search combining keyword, fuzzy, and semantic methods."""

from __future__ import annotations

from typing import Dict, Any, Literal

from .engine import search
from .fuzzy import search_fuzzy
from .semantic import search_semantic


def search_hybrid(
    query: str,
    top_n: int = 5,
    method: Literal["hybrid", "keyword", "fuzzy", "semantic"] = "hybrid",
    keyword_weight: float = 0.3,
    fuzzy_weight: float = 0.35,
    semantic_weight: float = 0.35,
) -> Dict[str, Any]:
    """Hybrid search combining multiple ranking methods.

    Args:
        query: User question
        top_n: Number of results to return
        method: Search method to use
        keyword_weight: Weight for keyword method (0-1)
        fuzzy_weight: Weight for fuzzy method (0-1)
        semantic_weight: Weight for semantic method (0-1)

    Returns:
        Dict with search results
    """

    if method == "keyword":
        return search(query, top_n=top_n)
    elif method == "fuzzy":
        return search_fuzzy(query, top_n=top_n)
    elif method == "semantic":
        return search_semantic(query, top_n=top_n)
    elif method == "hybrid":
        return _hybrid_combine(
            query,
            top_n,
            keyword_weight,
            fuzzy_weight,
            semantic_weight,
        )
    else:
        return {"found": False, "message": f"Unknown search method: {method}"}


def _hybrid_combine(
    query: str,
    top_n: int,
    keyword_weight: float,
    fuzzy_weight: float,
    semantic_weight: float,
) -> Dict[str, Any]:
    """Combine results from multiple search methods."""

    # Normalize weights
    total_weight = keyword_weight + fuzzy_weight + semantic_weight
    keyword_weight /= total_weight
    fuzzy_weight /= total_weight
    semantic_weight /= total_weight

    # Run all search methods
    keyword_results = search(query, top_n=top_n * 2)
    fuzzy_results = search_fuzzy(query, top_n=top_n * 2)
    semantic_results = search_semantic(query, top_n=top_n * 2)

    # Build score map
    scores = {}

    if keyword_results.get("found"):
        for idx, result in enumerate(keyword_results["results"]):
            result_id = result["id"]
            # Normalize keyword score from 0-6 to 0-1
            normalized = min(result["confidence"], 1.0)
            scores[result_id] = scores.get(result_id, 0) + normalized * keyword_weight

    if fuzzy_results.get("found"):
        for idx, result in enumerate(fuzzy_results["results"]):
            result_id = result["id"]
            # Normalize fuzzy score from 0-1
            normalized = min(result["confidence"], 1.0)
            scores[result_id] = scores.get(result_id, 0) + normalized * fuzzy_weight

    if semantic_results.get("found"):
        for idx, result in enumerate(semantic_results["results"]):
            result_id = result["id"]
            # Use similarity directly (already 0-1)
            normalized = result.get("similarity", result.get("confidence", 0))
            scores[result_id] = scores.get(result_id, 0) + normalized * semantic_weight

    if not scores:
        return {
            "found": False,
            "message": "No matching answer found.",
            "method": "hybrid"
        }

    # Collate results from best scoring IDs
    combined_results = {}

    for search_results in [keyword_results, fuzzy_results, semantic_results]:
        if search_results.get("found"):
            for result in search_results["results"]:
                rid = result["id"]
                if rid not in combined_results:
                    combined_results[rid] = result.copy()
                    combined_results[rid]["hybrid_score"] = scores[rid]

    # Sort by hybrid score
    sorted_results = sorted(
        combined_results.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    for r in sorted_results:
        r["confidence"] = round(r["hybrid_score"], 3)

    return {
        "found": True,
        "query": query,
        "results": sorted_results[:top_n],
        "method": "hybrid",
        "weights": {
            "keyword": round(keyword_weight, 3),
            "fuzzy": round(fuzzy_weight, 3),
            "semantic": round(semantic_weight, 3),
        }
    }


__all__ = ["search_hybrid"]
