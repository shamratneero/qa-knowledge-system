"""Fuzzy search module using RapidFuzz for typo-tolerant matching."""

from __future__ import annotations

from typing import Any, Dict
import re
import time

import pandas as pd
from rapidfuzz import fuzz

from ..core.logging import logger
from ..core.loader import load_knowledge_base


def _normalize(text: Any) -> str:
    """Normalize text for comparison."""
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Load knowledge base once at import time
try:
    _KB_DF = load_knowledge_base()
    _KB_LOAD_ERROR = None
except Exception as e:  # pragma: no cover
    _KB_DF = None
    _KB_LOAD_ERROR = e


def _fuzzy_score_row(row: pd.Series, query_norm: str, threshold: int = 80) -> int:
    """Score a row using fuzzy matching."""
    score = 0
    if not query_norm:
        return 0

    q = _normalize(row.get("question", ""))
    q_ratio = fuzz.token_set_ratio(query_norm, q)
    if q_ratio >= threshold:
        score += int(q_ratio / 20)  # Max 5 points

    kw = _normalize(row.get("keywords", ""))
    kw_ratio = fuzz.token_set_ratio(query_norm, kw)
    if kw_ratio >= threshold:
        score += int(kw_ratio / 25)  # Max 4 points

    cat = _normalize(row.get("category", ""))
    cat_ratio = fuzz.token_set_ratio(query_norm, cat)
    if cat_ratio >= threshold:
        score += int(cat_ratio / 20)  # Max 5 points

    return score


def search_fuzzy(query: str, top_n: int = 5, threshold: int = 80) -> Dict[str, Any]:
    """Fuzzy search using RapidFuzz for typo-tolerant matching.

    Args:
        query: User question (typos allowed)
        top_n: Number of results to return
        threshold: Minimum similarity score (0-100)

    Returns:
        Dict with 'found' boolean and either 'results' or 'message'
    """

    start = time.perf_counter()

    if _KB_DF is None:
        logger.error("fuzzy_search kb_not_loaded query=%r", query)
        return {
            "found": False,
            "message": f"Knowledge base not loaded: {_KB_LOAD_ERROR}",
        }

    query_norm = _normalize(query)
    if not query_norm:
        logger.warning("fuzzy_search empty_query")
        return {"found": False, "message": "Empty query."}

    scored = []

    for _, row in _KB_DF.iterrows():
        sc = _fuzzy_score_row(row, query_norm, threshold)
        if sc > 0:
            scored.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "category": row.get("category"),
                    "keywords": row.get("keywords"),
                    "score": int(sc),
                }
            )

    if not scored:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "fuzzy_search no_match query=%r top_n=%d threshold=%d elapsed_ms=%.1f",
            query,
            top_n,
            threshold,
            elapsed_ms,
        )
        return {"found": False, "message": "No matching answer found."}

    # compute confidence (max possible score is 14)
    for r in scored:
        r["confidence"] = round(min(r["score"] / 14.0, 1.0), 3)

    scored.sort(key=lambda x: x["score"], reverse=True)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "fuzzy_search match query=%r top_n=%d threshold=%d candidates=%d elapsed_ms=%.1f",
        query,
        top_n,
        threshold,
        len(scored),
        elapsed_ms,
    )

    return {"found": True, "query": query, "results": scored[:top_n], "method": "fuzzy"}


__all__ = ["search_fuzzy"]
