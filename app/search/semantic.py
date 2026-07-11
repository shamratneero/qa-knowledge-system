"""Semantic search using sentence-transformers embeddings."""

from __future__ import annotations

from typing import Any, Dict, Optional
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..core.loader import load_knowledge_base


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers required: pip install sentence-transformers")


# Model cache
_model = None
_model_name = "sentence-transformers/all-MiniLM-L6-v2"
_kb_embeddings = None
_KB_DF = None
_EMBEDDING_CACHE_FILE = Path(__file__).resolve().parent.parent.parent / ".cache" / "embeddings.pkl"


def _get_model():
    """Lazily load the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {_model_name}")
        _model = SentenceTransformer(_model_name)
    return _model


def _load_or_create_embeddings():
    """Load embeddings from cache or create new ones."""
    global _kb_embeddings, _KB_DF

    if _KB_DF is None:
        _KB_DF = load_knowledge_base()

    # Try to load from cache
    if _EMBEDDING_CACHE_FILE.exists():
        try:
            with open(_EMBEDDING_CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
                if cached.get("kb_hash") == hash(_KB_DF.to_json()):
                    _kb_embeddings = cached["embeddings"]
                    print("Loaded embeddings from cache")
                    return _kb_embeddings
        except Exception as e:
            print(f"Cache load failed: {e}, regenerating...")

    # Create embeddings
    print("Creating embeddings...")
    model = _get_model()

    texts = [str(_KB_DF.iloc[i]["question"]) for i in range(len(_KB_DF))]
    embeddings = model.encode(texts, show_progress_bar=False)

    # Cache embeddings
    _EMBEDDING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "kb_hash": hash(_KB_DF.to_json()),
        }, f)

    _kb_embeddings = embeddings
    return _kb_embeddings


def search_semantic(query: str, top_n: int = 5, threshold: float = 0.3) -> Dict[str, Any]:
    """Semantic search using embeddings.

    Does NOT match exact keywords, but understands meaning.
    Example: "Forgot password" → "How do I reset my password?"

    Args:
        query: User question
        top_n: Number of results to return
        threshold: Minimum similarity score (0-1)

    Returns:
        Dict with 'found' boolean and either 'results' or 'message'
    """

    try:
        embeddings = _load_or_create_embeddings()
    except Exception as e:
        return {"found": False, "message": f"Embedding load failed: {e}"}

    if _KB_DF is None or len(_KB_DF) == 0:
        return {"found": False, "message": "Knowledge base is empty"}

    if not query.strip():
        return {"found": False, "message": "Empty query"}

    # Encode query
    model = _get_model()
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    # Compute similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    scored = []
    for idx, sim_score in enumerate(similarities):
        if sim_score >= threshold:
            row = _KB_DF.iloc[idx]
            scored.append({
                "id": row.get("id"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "category": row.get("category"),
                "keywords": row.get("keywords"),
                "score": float(sim_score),
                "similarity": float(sim_score),
            })

    if not scored:
        return {"found": False, "message": "No matching answer found."}

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Use similarity as confidence
    for r in scored:
        r["confidence"] = round(r["similarity"], 3)

    return {"found": True, "query": query, "results": scored[:top_n], "method": "semantic"}


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _kb_embeddings
    if _EMBEDDING_CACHE_FILE.exists():
        _EMBEDDING_CACHE_FILE.unlink()
    _kb_embeddings = None
    print("Embedding cache cleared")


__all__ = ["search_semantic", "clear_embedding_cache"]
