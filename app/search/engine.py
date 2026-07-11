from __future__ import annotations

from typing import Any, Dict
import re
from pathlib import Path

import pandas as pd

from ..core.loader import load_knowledge_base


def _normalize(text: Any) -> str:
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Load knowledge base once at import time
try:
    _KB_DF = load_knowledge_base()
    _KB_LOAD_ERROR = None
except Exception as e:  # pragma: no cover - surface load errors to callers
    _KB_DF = None
    _KB_LOAD_ERROR = e


def _score_row(row: pd.Series, query_norm: str) -> int:
    score = 0
    if not query_norm:
        return 0

    q = _normalize(row.get("question", ""))
    if query_norm in q:
        score += 3

    kw = _normalize(row.get("keywords", ""))
    if query_norm in kw:
        score += 2

    cat = _normalize(row.get("category", ""))
    if query_norm in cat:
        score += 1

    return score


def search(query: str, top_n: int = 5) -> Dict[str, Any]:
    """Search the knowledge base using a simple keyword-based ranker.

    Returns a dict with `found` boolean and either `results` (list) or `message`.

    The function is intentionally a thin, stable API so the internal ranking
    logic can be swapped for semantic embeddings later without changing callers.
    """

    if _KB_DF is None:
        return {"found": False, "message": f"Knowledge base not loaded: {_KB_LOAD_ERROR}"}

    query_norm = _normalize(query)
    if not query_norm:
        return {"found": False, "message": "Empty query."}

    scored = []

    for _, row in _KB_DF.iterrows():
        sc = _score_row(row, query_norm)
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
        return {"found": False, "message": "No matching answer found."}

    # compute confidence (max possible score is 6)
    for r in scored:
        r["confidence"] = round(r["score"] / 6.0, 3)

    scored.sort(key=lambda x: x["score"], reverse=True)

    return {"found": True, "query": query, "results": scored[:top_n], "method": "keyword"}


__all__ = ["search"]
