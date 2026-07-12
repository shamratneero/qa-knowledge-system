"""Registration and authentication business logic (single-admin-account model)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import logger
from app.core.security import hash_password, verify_password
from app.models.database import User


def user_count(db: Session) -> int:
    """Return the number of accounts that exist."""
    return int(db.query(User).count())


def register_user(username: str, password: str, db: Session) -> User:
    """Create the single admin account. Raises ValueError if one already exists
    or the username is already taken."""
    if user_count(db) > 0:
        raise ValueError("Registration is closed: an account already exists.")

    existing = db.query(User).filter(User.username == username).first()
    if existing is not None:
        raise ValueError("Username already taken.")

    user = User(
        username=username, password_hash=hash_password(password), is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("user_registered username=%s", username)
    return user


def authenticate_user(username: str, password: str, db: Session) -> User | None:
    """Return the User if credentials are valid, else None."""
    user = (
        db.query(User)
        .filter(User.username == username, User.is_active.is_(True))
        .first()
    )
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


__all__ = ["authenticate_user", "register_user", "user_count"]
