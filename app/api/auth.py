"""Authentication routes: registration, login, session status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.core.security import create_access_token, get_current_user
from app.models.database import User
from app.models.schemas import (
    AuthStatusResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth import authenticate_user, register_user, user_count
from app.services.database import get_db, init_db

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get(
    "/status",
    response_model=AuthStatusResponse,
    summary="Whether an account exists yet",
)
def auth_status(db: Session = Depends(get_db)):
    init_db()
    return AuthStatusResponse(registered=user_count(db) > 0)


@router.post(
    "/register", response_model=TokenResponse, summary="Create the single admin account"
)
def auth_register(request: RegisterRequest, db: Session = Depends(get_db)):
    init_db()
    try:
        user = register_user(request.username, request.password, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e)) from None

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse, summary="Sign in")
def auth_login(request: LoginRequest, db: Session = Depends(get_db)):
    init_db()
    user = authenticate_user(request.username, request.password, db)
    if user is None:
        logger.warning("login_failed username=%s", request.username)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse, summary="Current authenticated user")
def auth_me(current_user: User = Depends(get_current_user)):
    return UserResponse(username=current_user.username)


__all__ = ["router"]
