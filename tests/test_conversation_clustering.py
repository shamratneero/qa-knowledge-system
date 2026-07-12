"""Tests for semantic conversation clustering."""

from __future__ import annotations

import pandas as pd

from app.services.conversation_clustering import cluster_conversations, cluster_count


def test_cluster_count_empty():
    df = pd.DataFrame(columns=["ticket_id", "conversation_text"])
    assert cluster_count(df) == 0


def test_cluster_single_row():
    df = pd.DataFrame({"ticket_id": ["1"], "conversation_text": ["reset password"]})
    out = cluster_conversations(df)
    assert out.iloc[0]["cluster_id"] == 0
    assert out.iloc[0]["cluster_label"] == "Cluster 0"


def test_cluster_assignments(monkeypatch):
    df = pd.DataFrame(
        {
            "ticket_id": ["1", "2", "3"],
            "conversation_text": ["reset password", "forgot password", "booking issue"],
        }
    )

    class FakeModel:
        def encode(self, texts, show_progress_bar=False):
            return [
                [1.0, 0.0, 0.0],
                [0.98, 0.02, 0.0],
                [0.0, 1.0, 0.0],
            ]

    monkeypatch.setattr(
        "app.services.conversation_clustering._get_model", lambda: FakeModel()
    )

    out = cluster_conversations(df, min_cluster_similarity=0.9, min_samples=2)
    assert "cluster_id" in out.columns
    assert "cluster_label" in out.columns
    assert cluster_count(out) >= 2
