"""Unit tests for retrieval-first AI analyst assistant service."""

from __future__ import annotations

from app.services import ai_assistant


def test_run_ai_query_deterministic(monkeypatch):
    def fake_search_knowledge_base(**kwargs):
        return {
            "total": 3,
            "items": [
                {
                    "conversation_id": "batch_1:1:T1",
                    "ticket_id": "T1",
                    "subject": "Refund issue",
                    "reconstructed_conversation": "Need refund",
                    "cluster_id": 3,
                    "cluster_label": "Refund",
                    "similarity": 0.8,
                    "classification": "existing_intent",
                    "upload_batch": "batch_1",
                    "upload_timestamp": "2026-07-12T00:00:00",
                    "semantic_similarity": 0.91,
                },
                {
                    "conversation_id": "batch_2:1:T2",
                    "ticket_id": "T2",
                    "subject": "Refund delayed",
                    "reconstructed_conversation": "Refund is delayed",
                    "cluster_id": 3,
                    "cluster_label": "Refund",
                    "similarity": 0.7,
                    "classification": "similar_historical",
                    "upload_batch": "batch_2",
                    "upload_timestamp": "2026-07-12T01:00:00",
                    "semantic_similarity": 0.84,
                },
            ],
        }

    monkeypatch.setattr(
        ai_assistant, "search_knowledge_base", fake_search_knowledge_base
    )
    monkeypatch.setattr(
        ai_assistant,
        "get_analytics_overview",
        lambda: {
            "total_tickets": 20,
            "total_conversations": 100,
            "duplicate_conversations": 30,
            "similar_conversations": 25,
            "unique_conversations": 45,
            "total_clusters": 8,
        },
    )
    monkeypatch.setattr(
        ai_assistant,
        "generate_business_insights",
        lambda: {"summary": {"redundancy_percentage": 55.0}},
    )

    result = ai_assistant.run_ai_query("Show refund conversations", limit=10)

    assert result["answer"]
    assert "Sources:" in result["answer"]
    assert result["generation_mode"] == "deterministic"
    assert result["confidence_score"] > 0
    assert len(result["sources"]) >= 1
    assert len(result["matching_clusters"]) >= 1


def test_run_ai_query_empty_question_raises():
    try:
        ai_assistant.run_ai_query("   ", limit=5)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_run_ai_query_uses_intent_and_summary_fields(monkeypatch):
    def fake_search_knowledge_base(**kwargs):
        return {
            "total": 1,
            "items": [
                {
                    "conversation_id": "batch_1:1:T1",
                    "ticket_id": "T1",
                    "subject": "Login issue",
                    "reconstructed_conversation": "Cannot log in",
                    "cluster_id": 5,
                    "cluster_label": "Password Reset",
                    "similarity": 0.9,
                    "classification": "existing_intent",
                    "upload_batch": "batch_1",
                    "upload_timestamp": "2026-07-12T00:00:00",
                    "semantic_similarity": 0.88,
                    "summary": "Customer cannot log in due to a forgotten password.",
                    "intent": "Password Reset",
                    "keywords": "password, login",
                    "category": "Authentication",
                    "sentiment": "negative",
                    "priority": "high",
                }
            ],
        }

    monkeypatch.setattr(
        ai_assistant, "search_knowledge_base", fake_search_knowledge_base
    )
    monkeypatch.setattr(
        ai_assistant,
        "get_analytics_overview",
        lambda: {
            "total_tickets": 5,
            "total_conversations": 10,
            "duplicate_conversations": 2,
            "similar_conversations": 3,
            "unique_conversations": 5,
            "total_clusters": 2,
        },
    )
    monkeypatch.setattr(
        ai_assistant,
        "generate_business_insights",
        lambda: {"summary": {"redundancy_percentage": 50.0}},
    )

    result = ai_assistant.run_ai_query(
        "What password issues have we seen before?", limit=5
    )

    assert result["answer"]
    assert result["generation_mode"] == "deterministic"
    assert result["matching_conversations"][0]["intent"] == "Password Reset"
    assert result["matching_conversations"][0]["summary"]
