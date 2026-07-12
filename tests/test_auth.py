"""Tests for the single-admin-account authentication layer."""

from __future__ import annotations


def test_public_routes_reachable_without_a_token(unauthenticated_client):
    assert unauthenticated_client.get("/health").status_code == 200
    assert unauthenticated_client.get("/ui").status_code == 200
    assert unauthenticated_client.get("/auth/status").status_code == 200


def test_protected_route_rejects_missing_token(unauthenticated_client):
    response = unauthenticated_client.get("/analytics/overview")
    assert response.status_code == 401


def test_protected_route_rejects_garbage_token(unauthenticated_client):
    response = unauthenticated_client.get(
        "/analytics/overview", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_protected_route_accepts_valid_token(client):
    response = client.get("/analytics/overview")
    assert response.status_code == 200


def test_login_and_me_happy_path(client, unauthenticated_client):
    """Exercises login + /auth/me directly against the shared test-admin
    account. Depends on the `client` fixture (even though unused directly)
    to guarantee the account has been registered first, since only one
    account can ever exist and test order isn't guaranteed."""
    from tests.conftest import _TEST_PASSWORD, _TEST_USERNAME

    login_resp = unauthenticated_client.post(
        "/auth/login", json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD}
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    assert token

    me_resp = unauthenticated_client.get(
        "/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == _TEST_USERNAME


def test_second_registration_is_rejected(client):
    """An account already exists (via the `client` fixture's setup) -- a
    second registration attempt must be blocked (single-admin-account model)."""
    response = client.post(
        "/auth/register", json={"username": "someone-else", "password": "Passw0rd123!"}
    )
    assert response.status_code == 403


def test_login_with_wrong_password_is_rejected(client, unauthenticated_client):
    from tests.conftest import _TEST_USERNAME

    response = unauthenticated_client.post(
        "/auth/login", json={"username": _TEST_USERNAME, "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_auth_status_reflects_registration(client):
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json()["registered"] is True
