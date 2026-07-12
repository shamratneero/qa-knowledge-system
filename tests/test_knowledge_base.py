"""Unit tests for persistent knowledge base services."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base
from app.services import knowledge_base


class FakeModel:
    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            value = 1.0 if "password" in str(text).lower() else 0.2
            vectors.append([value, 0.0, 0.0])
        return vectors


def test_knowledge_ingest_and_search(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(knowledge_base, "SessionLocal", Session)
    monkeypatch.setattr(knowledge_base, "_get_model", lambda: FakeModel())

    first = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2"],
            "subject": ["Password reset", "Refund request"],
            "conversation_text": ["password reset issue", "need refund"],
            "cluster_id": [1, 2],
            "cluster_label": ["Password", "Refund"],
        }
    )
    second = pd.DataFrame(
        {
            "ticket_id": ["T3"],
            "subject": ["Password help"],
            "conversation_text": ["password reset issue"],
            "cluster_id": [1],
            "cluster_label": ["Password"],
        }
    )

    s1 = knowledge_base.ingest_conversations_to_knowledge_base(
        first, upload_batch="batch_a"
    )
    assert s1["saved"] == 2
    assert s1["new_intent"] == 2

    s2 = knowledge_base.ingest_conversations_to_knowledge_base(
        second, upload_batch="batch_b"
    )
    assert s2["saved"] == 1
    assert s2["existing_intent"] >= 1

    results = knowledge_base.search_knowledge_base(query="password", limit=10)
    assert results["total"] >= 2
    assert any(item["semantic_similarity"] is not None for item in results["items"])


def test_knowledge_similar_and_dashboard(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(knowledge_base, "SessionLocal", Session)
    monkeypatch.setattr(knowledge_base, "_get_model", lambda: FakeModel())

    df = pd.DataFrame(
        {
            "ticket_id": ["K1", "K2", "K3"],
            "subject": ["Password", "Password", "Refund"],
            "conversation_text": ["password issue", "password issue", "refund issue"],
            "cluster_id": [1, 1, 2],
            "cluster_label": ["Password", "Password", "Refund"],
        }
    )

    knowledge_base.ingest_conversations_to_knowledge_base(df, upload_batch="batch_1")

    search = knowledge_base.search_knowledge_base(limit=10)
    conv_id = search["items"][0]["conversation_id"]

    similar = knowledge_base.get_similar_historical_conversations(conv_id, top_n=10)
    assert len(similar) >= 1
    assert "similarity" in similar[0]

    dashboard = knowledge_base.get_knowledge_dashboard()
    assert dashboard["total_uploads"] == 1
    assert dashboard["knowledge_base_size"] == 3
    assert isinstance(dashboard["historical_growth"], list)


def test_ingest_stores_summary_fields_and_embeds_off_embedding_text(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    class DirectionalFakeModel:
        """Encodes along different axes depending on content, so cosine
        similarity actually reveals which text column was embedded."""

        def encode(self, texts, show_progress_bar=False):
            vectors = []
            for text in texts:
                if "password" in str(text).lower():
                    vectors.append([1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 1.0, 0.0])
            return vectors

    monkeypatch.setattr(knowledge_base, "SessionLocal", Session)
    monkeypatch.setattr(knowledge_base, "_get_model", lambda: DirectionalFakeModel())

    df = pd.DataFrame(
        {
            "ticket_id": ["T1"],
            "subject": ["Password reset"],
            "conversation_text": ["Hi"],
            "embedding_text": ["Customer forgot their password. Intent: Password Reset."],
            "cluster_id": [1],
            "cluster_label": ["Password"],
            "summary": ["Customer forgot their password."],
            "intent": ["Password Reset"],
            "keywords": ["password, login"],
            "category": ["Authentication"],
            "sentiment": ["negative"],
            "priority": ["high"],
        }
    )

    knowledge_base.ingest_conversations_to_knowledge_base(df, upload_batch="batch_x")

    results = knowledge_base.search_knowledge_base(keywords="password", limit=10)
    assert results["total"] == 1
    item = results["items"][0]
    assert item["summary"] == "Customer forgot their password."
    assert item["intent"] == "Password Reset"
    assert item["category"] == "Authentication"
    assert item["sentiment"] == "negative"
    assert item["priority"] == "high"

    # "Hi" (raw conversation_text) does not contain "password" and would embed
    # along a different axis -- similarity to a "password" query only stays
    # 1.0 if embedding_text (which contains "password") is what was embedded.
    semantic = knowledge_base.search_knowledge_base(query="password", limit=10)
    assert semantic["items"][0]["semantic_similarity"] == 1.0
