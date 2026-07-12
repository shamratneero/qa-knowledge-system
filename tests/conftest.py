"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.main import app

_TEST_USERNAME = "test-admin"
_TEST_PASSWORD = "Testpass123!"


def _authenticated_test_client() -> TestClient:
    """A TestClient pre-authenticated as the single admin test account.

    All routes require a bearer token now, so every existing test that uses
    the `client` fixture needs to keep working unmodified. Registration is
    idempotent against the shared test DB: the first test run registers the
    account, later runs just log in.
    """
    c = TestClient(app)
    response = c.post(
        "/auth/login", json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD}
    )
    if response.status_code != 200:
        c.post(
            "/auth/register",
            json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD},
        )
        response = c.post(
            "/auth/login", json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD}
        )
    token = response.json()["access_token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


@pytest.fixture
def client():
    return _authenticated_test_client()


@pytest.fixture
def unauthenticated_client():
    """A plain, unauthenticated TestClient for testing the auth boundary itself."""
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
