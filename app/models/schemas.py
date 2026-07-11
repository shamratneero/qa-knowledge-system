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

    id: int | str | None = Field(default=None, description="Knowledge entry identifier.")
    question: str = Field(description="Matched question text from the knowledge base.")
    answer: str = Field(description="Matched answer text from the knowledge base.")
    category: Optional[str] = None
    keywords: Optional[str] = None
    score: float = Field(description="Raw score produced by the selected search method.")
    confidence: float = Field(description="Normalized confidence score in the range 0 to 1.")


class AskResponse(BaseModel):
    """Successful search response."""

    found: bool = Field(default=True, description="Whether at least one relevant result was found.")
    query: str = Field(description="The original user query.")
    method: str = Field(description="Search strategy used to produce ranking.")
    answer: str = Field(description="Best-matching answer from the knowledge base.")
    score: float = Field(description="Confidence score of the top result (0–1).")
    results: list[SearchResult]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "found": True,
                    "query": "What is machine learning?",
                    "method": "hybrid",
                    "answer": "Machine Learning is a subset of AI that learns patterns from data.",
                    "score": 0.87,
                    "results": [
                        {
                            "id": 2,
                            "question": "What is Machine Learning?",
                            "answer": "Machine Learning is a subset of AI that learns patterns from data.",
                            "category": "ML",
                            "keywords": "Machine Learning,ML",
                            "score": 0.87,
                            "confidence": 0.87,
                        }
                    ],
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str = Field(description="Human-readable error description.")


class RootResponse(BaseModel):
    """Root endpoint payload."""

    message: str
    docs: str
    health: str


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
    "RootResponse",
    "SearchMethod",
    "SearchResult",
]
