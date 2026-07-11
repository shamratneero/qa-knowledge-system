"""FastAPI entrypoint for QA Knowledge System."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    AskResponse,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    SearchResult,
)
from app.search.hybrid import search_hybrid


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    responses={
        404: {"model": ErrorResponse, "description": "No matching answer found."},
        422: {"model": ErrorResponse, "description": "Validation error."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


def _build_response(result: dict[str, Any]) -> AskResponse:
    """Convert a search result dict into an API response."""
    top = result["results"][0]
    return AskResponse(
        found=True,
        query=result["query"],
        method=result.get("method", "keyword"),
        answer=str(top["answer"]),
        score=float(top.get("confidence", top.get("score", 0))),
        results=[
            SearchResult(
                id=r.get("id"),
                question=str(r["question"]),
                answer=str(r["answer"]),
                category=r.get("category"),
                keywords=r.get("keywords"),
                score=float(r.get("score", 0)),
                confidence=float(r.get("confidence", r.get("score", 0))),
            )
            for r in result["results"]
        ],
    )


@app.get("/", tags=["Root"])
def home():
    return {
        "message": "Q&A Knowledge System API is running!",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
def health():
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Knowledge Base"],
    summary="Search the knowledge base",
    responses={
        200: {"description": "Matching answer found."},
        404: {"description": "No matching answer found."},
    },
)
def ask_question(request: QuestionRequest) -> AskResponse:
    """Search the knowledge base and return the best answer with confidence score."""
    start = time.perf_counter()

    logger.info(
        "query=%r method=%s top_n=%d",
        request.question,
        request.method,
        request.top_n,
    )

    try:
        result = search_hybrid(
            request.question,
            top_n=request.top_n,
            method=request.method,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("search failed after %.1fms", elapsed_ms)
        raise HTTPException(status_code=500, detail="Search failed.") from None

    elapsed_ms = (time.perf_counter() - start) * 1000

    if not result.get("found"):
        logger.warning(
            "no match query=%r method=%s elapsed_ms=%.1f message=%s",
            request.question,
            request.method,
            elapsed_ms,
            result.get("message"),
        )
        raise HTTPException(status_code=404, detail=result.get("message", "Not found."))

    response = _build_response(result)
    logger.info(
        "match query=%r method=%s score=%.3f elapsed_ms=%.1f results=%d",
        request.question,
        response.method,
        response.score,
        elapsed_ms,
        len(response.results),
    )
    return response


__all__ = ["app", "ask_question", "health", "home"]
