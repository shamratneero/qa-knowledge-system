"""Pydantic request and response schemas for the API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


SearchMethod = Literal["hybrid", "keyword", "fuzzy", "semantic"]


class QuestionRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["What is machine learning?"],
        description="The user's question to search the knowledge base.",
    )
    method: SearchMethod = Field(
        default="hybrid",
        description="Search strategy: hybrid (default), keyword, fuzzy, or semantic.",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of ranked results to return.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is machine learning?",
                    "method": "hybrid",
                    "top_n": 3,
                }
            ]
        }
    }


class SearchResult(BaseModel):
    """A single knowledge-base match."""

    id: int | str | None = None
    question: str
    answer: str
    category: Optional[str] = None
    keywords: Optional[str] = None
    score: float
    confidence: float


class AskResponse(BaseModel):
    """Successful search response."""

    found: bool = True
    query: str
    method: str
    answer: str = Field(description="Best-matching answer from the knowledge base.")
    score: float = Field(description="Confidence score of the top result (0–1).")
    results: list[SearchResult]


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    app: str
    version: str


__all__ = [
    "AskResponse",
    "ErrorResponse",
    "HealthResponse",
    "QuestionRequest",
    "SearchMethod",
    "SearchResult",
]
