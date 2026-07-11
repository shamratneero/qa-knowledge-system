"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_questions():
    return {
        "ai": "What is AI?",
        "ml": "What is Machine Learning?",
        "typo": "What is Machne Learning?",
        "paraphrase": "Tell me about artificial intelligence",
        "missing": "nonexistenttermxyz123",
        "empty": "",
    }
