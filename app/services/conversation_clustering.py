"""Semantic clustering for reconstructed conversations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

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
        logger.info("Loading conversation clustering model: %s", _model_name)
        _model = SentenceTransformer(_model_name)
    return _model


def cluster_conversations(
    conversations_df: pd.DataFrame,
    min_cluster_similarity: float = 0.78,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Assign semantic cluster IDs and labels to conversations."""
    if conversations_df.empty:
        return conversations_df.copy()

    df = conversations_df.copy()
    text_col = "embedding_text" if "embedding_text" in df.columns else "conversation_text"
    texts = [str(x) for x in df[text_col].tolist()]

    if len(texts) == 1:
        df["cluster_id"] = [0]
        df["cluster_label"] = ["Cluster 0"]
        return df

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    eps = max(0.01, 1.0 - float(min_cluster_similarity))
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(np.asarray(embeddings))

    next_cluster_id = (labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0
    normalized = []
    for lbl in labels:
        if int(lbl) == -1:
            normalized.append(int(next_cluster_id))
            next_cluster_id += 1
        else:
            normalized.append(int(lbl))

    df["cluster_id"] = normalized
    df["cluster_label"] = [f"Cluster {cid}" for cid in normalized]

    logger.info(
        "conversation_clustering completed rows=%d clusters=%d",
        len(df),
        len(set(normalized)),
    )
    return df


def cluster_count(clustered_df: pd.DataFrame) -> int:
    """Return the number of distinct semantic clusters."""
    if clustered_df.empty or "cluster_id" not in clustered_df.columns:
        return 0
    return int(clustered_df["cluster_id"].nunique())


__all__ = ["cluster_conversations", "cluster_count"]
