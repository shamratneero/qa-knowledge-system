"""Tests for FastAPI routes."""

from __future__ import annotations

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
        assert "message" in response.json()


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
