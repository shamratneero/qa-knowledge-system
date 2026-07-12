"""Deterministic cluster labeling for semantic conversation clusters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class ClusterLabelGenerator(Protocol):
    """Interface for pluggable cluster label generators (e.g., future LLM-backed generators)."""

    def generate_label(self, cluster_id: int, cluster_df: pd.DataFrame) -> str:
        """Generate a concise human-readable label for one cluster."""


@dataclass(slots=True)
class DeterministicClusterLabelGenerator:
    """Generate cluster labels using deterministic keyword extraction."""

    max_terms: int = 2
    max_label_chars: int = 100

    def generate_label(self, cluster_id: int, cluster_df: pd.DataFrame) -> str:
        if cluster_df.empty:
            return f"Cluster {cluster_id}"

        texts = self._build_documents(cluster_df)
        if not texts:
            return f"Cluster {cluster_id}"

        terms = self._extract_top_terms(texts)
        if not terms:
            return f"Cluster {cluster_id}"

        label = " / ".join(
            self._title_case_term(term) for term in terms[: self.max_terms]
        ).strip()
        if not label:
            return f"Cluster {cluster_id}"

        return label[: self.max_label_chars]

    def _build_documents(self, cluster_df: pd.DataFrame) -> list[str]:
        documents: list[str] = []
        for _, row in cluster_df.iterrows():
            parts = [
                str(row.get("subject", "") or "").strip(),
                str(row.get("conversation_text", "") or "").strip(),
            ]
            text = "\n".join(p for p in parts if p)
            if text:
                documents.append(text)
        return documents

    def _extract_top_terms(self, texts: list[str]) -> list[str]:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",
            max_features=1500,
        )
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] == 0:
            return []

        mean_scores = matrix.mean(axis=0).A1
        features = vectorizer.get_feature_names_out()

        ranked = sorted(
            ((features[idx], float(mean_scores[idx])) for idx in range(len(features))),
            key=lambda item: (-item[1], item[0]),
        )

        unique_terms: list[str] = []
        seen_tokens: set[str] = set()
        for term, _ in ranked:
            normalized = term.strip().lower()
            if not normalized:
                continue
            token_key = tuple(sorted(normalized.split()))
            token_hash = " ".join(token_key)
            if token_hash in seen_tokens:
                continue
            seen_tokens.add(token_hash)
            unique_terms.append(normalized)
            if len(unique_terms) >= max(self.max_terms * 4, 6):
                break

        return unique_terms

    def _title_case_term(self, term: str) -> str:
        cleaned = re.sub(r"\s+", " ", term).strip()
        return " ".join(part.capitalize() for part in cleaned.split(" "))


def assign_cluster_labels(
    clustered_df: pd.DataFrame,
    generator: ClusterLabelGenerator | None = None,
) -> pd.DataFrame:
    """Assign deterministic cluster labels while preserving numeric cluster IDs."""
    if clustered_df.empty:
        return clustered_df.copy()

    if "cluster_id" not in clustered_df.columns:
        return clustered_df.copy()

    df = clustered_df.copy()
    label_generator = generator or DeterministicClusterLabelGenerator()

    labels_by_id: dict[int, str] = {}
    for cluster_id, group in df.groupby("cluster_id", sort=True):
        cid = int(cluster_id)
        generated = label_generator.generate_label(cid, group)
        labels_by_id[cid] = generated or f"Cluster {cid}"

    df["cluster_label"] = [
        labels_by_id.get(int(cid), f"Cluster {int(cid)}")
        for cid in df["cluster_id"].tolist()
    ]
    return df


__all__ = [
    "ClusterLabelGenerator",
    "DeterministicClusterLabelGenerator",
    "assign_cluster_labels",
]
