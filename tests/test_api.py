"""Tests for FastAPI routes."""

from __future__ import annotations

import time

import pytest


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "app" in data
        assert "version" in data

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data

    def test_ui_route(self, client):
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Upload Conversation Excel" in response.text

    def test_request_id_header_added(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers


class TestAskEndpoint:
    def test_keyword_search_success(self, client, sample_questions):
        response = client.post(
            "/ask",
            json={
                "question": sample_questions["ai"],
                "method": "keyword",
                "top_n": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["method"] == "keyword"
        assert data["answer"]
        assert 0 < data["score"] <= 1
        assert len(data["results"]) >= 1

    def test_not_found_returns_404(self, client, sample_questions):
        response = client.post(
            "/ask",
            json={"question": sample_questions["missing"], "method": "keyword"},
        )
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_empty_question_returns_422(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_invalid_method_returns_422(self, client, sample_questions):
        response = client.post(
            "/ask",
            json={"question": sample_questions["ai"], "method": "invalid"},
        )
        assert response.status_code == 422

    def test_top_n_bounds(self, client, sample_questions):
        response = client.post(
            "/ask",
            json={"question": sample_questions["ai"], "top_n": 0},
        )
        assert response.status_code == 422

    def test_ask_returns_request_id(self, client, sample_questions):
        response = client.post(
            "/ask",
            json={"question": sample_questions["ai"], "method": "keyword"},
        )
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_keyword_performance_smoke(self, client, sample_questions):
        start = time.perf_counter()
        response = client.post(
            "/ask",
            json={"question": sample_questions["ai"], "method": "keyword"},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        # generous bound to avoid flaky tests while still catching regressions
        assert elapsed_ms < 1500

    @pytest.mark.slow
    def test_hybrid_search_success(self, client, sample_questions):
        response = client.post(
            "/ask",
            json={
                "question": sample_questions["ml"],
                "method": "hybrid",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "hybrid"
        assert data["score"] > 0


class TestAnalyticsEndpoints:
    def test_analytics_overview(self, client):
        response = client.get("/analytics/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_tickets" in data
        assert "total_conversations" in data
        assert "duplicate_conversations" in data
        assert "similar_conversations" in data
        assert "unique_conversations" in data
        assert "total_clusters" in data

    def test_analytics_conversations(self, client):
        response = client.get("/analytics/conversations?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_analytics_charts(self, client):
        response = client.get("/analytics/charts")
        assert response.status_code == 200
        data = response.json()
        assert "status_distribution" in data
        assert "cluster_distribution" in data
        assert "daily_volume" in data
        assert "top_recurring_issues" in data

    def test_analytics_insights(self, client):
        response = client.get("/analytics/insights")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "recurring_issues" in data
        assert "automation_opportunities" in data
        assert "emerging_issues" in data
        assert "recommendations" in data

    def test_analytics_export_csv(self, client):
        response = client.get("/analytics/export/csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_analytics_export_excel(self, client):
        response = client.get("/analytics/export/excel")
        assert response.status_code == 200
        assert (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            in response.headers.get("content-type", "")
        )

    def test_analytics_conversation_detail_not_found(self, client):
        response = client.get("/analytics/conversations/999999")
        assert response.status_code == 404

    def test_analytics_intents(self, client):
        response = client.get("/analytics/intents")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_analytics_conversations_filters_by_intent(self, client):
        response = client.get("/analytics/conversations?intent=Nonexistent Intent XYZ")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []


class TestKnowledgeEndpoints:
    def test_knowledge_dashboard(self, client):
        response = client.get("/knowledge/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "total_uploads" in data
        assert "knowledge_base_size" in data
        assert "new_conversations" in data
        assert "known_conversations" in data
        assert "new_intents_discovered" in data
        assert "historical_growth" in data

    def test_knowledge_search(self, client):
        response = client.get("/knowledge/search?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data

    def test_knowledge_similar_not_found(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.main.get_similar_historical_conversations",
            lambda conversation_id, top_n=10: [],
        )
        monkeypatch.setattr(
            "app.main.search_knowledge_base",
            lambda **kwargs: {"total": 1, "items": [{"conversation_id": "x"}]},
        )

        response = client.get("/knowledge/similar/missing-conversation")
        assert response.status_code == 404


class TestAIEndpoints:
    def test_ai_query_success(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.main.run_ai_query",
            lambda question, limit=10: {
                "answer": "Top issue appears to be refunds. Sources: batch_1:1:T1.",
                "sources": [
                    {
                        "conversation_id": "batch_1:1:T1",
                        "ticket_id": "T1",
                        "cluster_id": 3,
                        "cluster_label": "Refund",
                        "upload_batch": "batch_1",
                        "semantic_similarity": 0.91,
                    }
                ],
                "matching_conversations": [
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
                    }
                ],
                "matching_clusters": [
                    {
                        "cluster_id": 3,
                        "cluster_label": "Refund",
                        "conversation_count": 1,
                    }
                ],
                "confidence_score": 0.91,
                "generation_mode": "deterministic",
            },
        )

        response = client.post(
            "/ai/query", json={"question": "Show refund conversations", "top_k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "matching_conversations" in data
        assert "matching_clusters" in data
        assert "confidence_score" in data

    def test_ai_query_validation(self, client):
        response = client.post("/ai/query", json={"question": "", "top_k": 5})
        assert response.status_code == 422
