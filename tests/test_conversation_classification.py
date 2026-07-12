"""Tests for semantic conversation classification."""

from __future__ import annotations

import pandas as pd

from app.services.conversation_classification import (
    classification_counts,
    classify_conversations,
)


def test_classification_counts_empty():
    df = pd.DataFrame(columns=["ticket_id", "conversation_text"])
    out = classification_counts(df)
    assert out == {"duplicate": 0, "similar": 0, "unique": 0}


def test_classification_columns_present(monkeypatch):
    df = pd.DataFrame(
        {
            "ticket_id": ["1", "2", "3"],
            "conversation_text": [
                "reset password",
                "forgot password",
                "request refund",
            ],
        }
    )

    class FakeModel:
        def encode(self, texts, show_progress_bar=False):
            return [
                [1.0, 0.0, 0.0],
                [0.98, 0.02, 0.0],
                [0.0, 0.0, 1.0],
            ]

    monkeypatch.setattr(
        "app.services.conversation_classification._get_model", lambda: FakeModel()
    )

    out = classify_conversations(df, duplicate_threshold=0.95, similar_threshold=0.80)
    assert "status" in out.columns
    assert "similarity_score" in out.columns
    assert "nearest_ticket_id" in out.columns

    counts = classification_counts(out)
    assert counts["duplicate"] >= 1
    assert counts["unique"] >= 1


def test_classification_prefers_embedding_text_when_present(monkeypatch):
    df = pd.DataFrame(
        {
            "ticket_id": ["1", "2", "3"],
            "conversation_text": ["Hi", "Hello", "Need a refund"],
            "embedding_text": [
                "Customer greeted the agent. Intent: General Inquiry.",
                "Customer greeted the agent. Intent: General Inquiry.",
                "Customer requested a refund. Intent: Refund Request.",
            ],
        }
    )

    class FakeModel:
        def encode(self, texts, show_progress_bar=False):
            vectors = []
            for text in texts:
                if "General Inquiry" in text:
                    vectors.append([1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 1.0])
            return vectors

    monkeypatch.setattr(
        "app.services.conversation_classification._get_model", lambda: FakeModel()
    )

    out = classify_conversations(df, duplicate_threshold=0.95, similar_threshold=0.80)
    # Rows 1 and 2 embed identically (via embedding_text) despite different raw text.
    assert out.loc[0, "status"] == "duplicate"
    assert out.loc[1, "status"] == "duplicate"
    assert out.loc[2, "status"] == "unique"
