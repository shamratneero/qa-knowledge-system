"""Semantic classification for reconstructed conversations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.core.logging import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers required: pip install sentence-transformers"
    )


_model = None
_model_name = settings.embedding_model_name


def _get_model() -> SentenceTransformer:
    """Lazily load the sentence embedding model."""
    global _model
    if _model is None:
        logger.info("Loading conversation classification model: %s", _model_name)
        _model = SentenceTransformer(_model_name)
    return _model


def classify_conversations(
    conversations_df: pd.DataFrame,
    duplicate_threshold: float = 0.95,
    similar_threshold: float = 0.80,
) -> pd.DataFrame:
    """Classify conversations as duplicate, similar, or unique using semantic similarity."""
    if conversations_df.empty:
        return conversations_df.copy()

    df = conversations_df.copy()
    texts = [str(x) for x in df["conversation_text"].tolist()]
    ticket_ids = [str(x) for x in df["ticket_id"].tolist()]

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    sim = cosine_similarity(embeddings)

    status: list[str] = []
    nearest_ticket_ids: list[str | None] = []
    nearest_scores: list[float] = []

    for idx in range(len(texts)):
        row_sim = sim[idx].copy()
        row_sim[idx] = -1.0
        nearest_idx = int(np.argmax(row_sim))
        nearest_score = float(row_sim[nearest_idx])

        if nearest_score >= duplicate_threshold:
            s = "duplicate"
        elif nearest_score >= similar_threshold:
            s = "similar"
        else:
            s = "unique"

        status.append(s)
        nearest_ticket_ids.append(
            ticket_ids[nearest_idx] if nearest_score >= 0 else None
        )
        nearest_scores.append(round(max(nearest_score, 0.0), 3))

    df["status"] = status
    df["similarity_score"] = nearest_scores
    df["nearest_ticket_id"] = nearest_ticket_ids

    logger.info(
        "conversation_classification completed rows=%d duplicate=%d similar=%d unique=%d",
        len(df),
        int((df["status"] == "duplicate").sum()),
        int((df["status"] == "similar").sum()),
        int((df["status"] == "unique").sum()),
    )
    return df


def classification_counts(classified_df: pd.DataFrame) -> dict[str, int]:
    """Return aggregate counts for classification statuses."""
    if classified_df.empty:
        return {"duplicate": 0, "similar": 0, "unique": 0}

    return {
        "duplicate": int((classified_df["status"] == "duplicate").sum()),
        "similar": int((classified_df["status"] == "similar").sum()),
        "unique": int((classified_df["status"] == "unique").sum()),
    }


__all__ = ["classify_conversations", "classification_counts"]
