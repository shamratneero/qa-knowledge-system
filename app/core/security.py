"""Password hashing and JWT access tokens for the single-admin-account auth layer."""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import User
from app.services.database import get_db

_PBKDF2_ALGORITHM = "sha256"
_PBKDF2_ITERATIONS = 260_000
_SALT_BYTES = 16

_bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random per-user salt."""
    salt = os.urandom(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGORITHM, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return f"{salt.hex()}${derived.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a hash produced by hash_password()."""
    try:
        salt_hex, hash_hex = password_hash.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(hash_hex)
    derived = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGORITHM, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(derived, expected)


def create_access_token(
    data: dict[str, Any], expires_minutes: int | None = None
) -> str:
    """Create a signed JWT access token."""
    minutes = (
        expires_minutes
        if expires_minutes is not None
        else settings.access_token_expire_minutes
    )
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload = {**data, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and verify a JWT access token; returns None if invalid/expired."""
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except jwt.PyJWTError:
        return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: resolve the authenticated User from a bearer token."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise unauthorized

    username = payload.get("sub")
    if not username:
        raise unauthorized

    user = (
        db.query(User)
        .filter(User.username == username, User.is_active.is_(True))
        .first()
    )
    if user is None:
        raise unauthorized

    return user


__all__ = [
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "hash_password",
    "verify_password",
]
