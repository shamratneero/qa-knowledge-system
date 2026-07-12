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

    id: int | str | None = Field(
        default=None, description="Knowledge entry identifier."
    )
    question: str = Field(description="Matched question text from the knowledge base.")
    answer: str = Field(description="Matched answer text from the knowledge base.")
    category: Optional[str] = None
    keywords: Optional[str] = None
    score: float = Field(
        description="Raw score produced by the selected search method."
    )
    confidence: float = Field(
        description="Normalized confidence score in the range 0 to 1."
    )


class AskResponse(BaseModel):
    """Successful search response."""

    found: bool = Field(
        default=True, description="Whether at least one relevant result was found."
    )
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


class ConversationPreviewRow(BaseModel):
    """A reconstructed conversation returned in an upload preview."""

    ticket_id: str
    subject: str
    message_count: int
    status: str = "unique"
    similarity_score: float = 0.0
    nearest_ticket_id: Optional[str] = None
    cluster_id: int = 0
    cluster_label: str = "Cluster 0"
    first_sent_at: Optional[str] = None
    last_sent_at: Optional[str] = None
    conversation_text: str
    summary: str = ""
    intent: str = ""
    keywords: str = ""
    category: str = ""
    sentiment: str = "neutral"
    priority: str = "low"


class ConversationImportReport(BaseModel):
    """Summary of a validated conversation Excel upload."""

    file_name: str
    total_source_rows: int
    valid_conversations: int
    stored_conversations: int
    unique_ticket_ids: int
    duplicate_conversations: int
    similar_conversations: int
    unique_conversations: int
    total_clusters: int
    invalid_rows: int
    required_sheets: list[str]
    required_columns: list[str]


class ConversationPreviewResponse(BaseModel):
    """Preview response for an uploaded conversation Excel file."""

    report: ConversationImportReport
    preview_rows: list[ConversationPreviewRow]
    preview_count: int


class AnalyticsOverviewResponse(BaseModel):
    """KPI overview for conversation analytics dashboard."""

    total_tickets: int
    total_conversations: int
    duplicate_conversations: int
    similar_conversations: int
    unique_conversations: int
    total_clusters: int


class ConversationTableRow(BaseModel):
    """Row for analytics conversation table."""

    id: int
    ticket_id: str
    subject: str
    status: str
    similarity_score: float
    nearest_ticket_id: Optional[str] = None
    cluster_id: int
    cluster_label: str
    message_count: int
    first_sent_at: Optional[str] = None
    last_sent_at: Optional[str] = None
    representative_conversation: str
    conversation_text: str
    summary: str = ""
    intent: str = ""
    keywords: str = ""
    category: str = ""
    sentiment: str = "neutral"
    priority: str = "low"


class ConversationTableResponse(BaseModel):
    """Filtered conversation table payload for dashboard."""

    total: int
    items: list[ConversationTableRow]


class ClusterSummary(BaseModel):
    """Summary details for one cluster."""

    cluster_id: int
    cluster_label: str
    conversation_count: int


class InsightsSummary(BaseModel):
    """High-level deterministic insight metrics."""

    total_tickets: int
    total_conversations: int
    duplicate_conversations: int
    similar_conversations: int
    unique_conversations: int
    total_clusters: int
    redundancy_percentage: float
    largest_cluster: ClusterSummary | None = None
    smallest_cluster: ClusterSummary | None = None
    average_similarity_score: float
    median_similarity_score: float
    average_messages_per_ticket: float


class RecurringIssue(BaseModel):
    """Recurring issue entry derived from clusters, grouped by customer intent."""

    cluster_id: int
    cluster_label: str
    conversation_count: int
    percentage_of_total: float
    representative_ticket: str
    intent: Optional[str] = None


class AutomationOpportunity(BaseModel):
    """Cluster that meets deterministic automation opportunity criteria."""

    cluster_id: int
    cluster_label: str
    conversation_count: int
    duplicate_rate: float
    trigger_reasons: list[str]
    estimated_automatable_conversations: int
    estimated_automation_opportunity: float


class EmergingIssue(BaseModel):
    """Potential emerging issue based on uniqueness or growth signals."""

    cluster_id: int
    cluster_label: str
    reason: str
    conversation_count: int
    unique_rate: float | None = None
    recent_count: int | None = None
    previous_count: int | None = None
    growth_ratio: float | None = None


class AnalyticsInsightsResponse(BaseModel):
    """Business insights response for analytics dashboard."""

    summary: InsightsSummary
    recurring_issues: list[RecurringIssue]
    automation_opportunities: list[AutomationOpportunity]
    emerging_issues: list[EmergingIssue]
    recommendations: list[str]


class KnowledgeSearchRow(BaseModel):
    """Knowledge base search result row."""

    conversation_id: str
    ticket_id: str
    subject: str
    reconstructed_conversation: str
    cluster_id: int
    cluster_label: str
    similarity: float
    classification: str
    upload_batch: str
    upload_timestamp: Optional[str] = None
    semantic_similarity: Optional[float] = None
    summary: str = ""
    intent: str = ""
    keywords: str = ""
    category: str = ""
    sentiment: str = "neutral"
    priority: str = "low"


class KnowledgeSearchResponse(BaseModel):
    """Search results for historical knowledge base conversations."""

    total: int
    items: list[KnowledgeSearchRow]


class SimilarConversationRow(BaseModel):
    """Nearest historical conversation row."""

    conversation_id: str
    ticket_id: str
    cluster_id: int
    cluster_label: str
    upload_batch: str
    classification: str
    similarity: float
    subject: str


class SimilarConversationsResponse(BaseModel):
    """Similar historical conversations payload."""

    conversation_id: str
    top_n: int
    items: list[SimilarConversationRow]


class KnowledgeGrowthPoint(BaseModel):
    """Knowledge growth data point."""

    date: str
    new_records: int
    cumulative_records: int


class KnowledgeDashboardResponse(BaseModel):
    """Knowledge base dashboard metrics."""

    total_uploads: int
    knowledge_base_size: int
    new_conversations: int
    known_conversations: int
    new_intents_discovered: int
    historical_growth: list[KnowledgeGrowthPoint]


class AIQueryRequest(BaseModel):
    """Request body for AI analyst query endpoint."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are customers complaining about most?",
                    "top_k": 10,
                }
            ]
        }
    }


class AISource(BaseModel):
    """Cited source conversation used in an AI answer."""

    conversation_id: str
    ticket_id: str
    cluster_id: int
    cluster_label: str
    upload_batch: str
    semantic_similarity: Optional[float] = None


class AIClusterMatch(BaseModel):
    """Cluster-level aggregation for AI retrieval context."""

    cluster_id: int
    cluster_label: str
    conversation_count: int


class AIQueryResponse(BaseModel):
    """Response payload for AI analyst queries."""

    answer: str
    sources: list[AISource]
    matching_conversations: list[KnowledgeSearchRow]
    matching_clusters: list[AIClusterMatch]
    confidence_score: float
    generation_mode: str


__all__ = [
    "AskResponse",
    "ErrorResponse",
    "HealthResponse",
    "ConversationImportReport",
    "AnalyticsOverviewResponse",
    "ConversationTableResponse",
    "ConversationTableRow",
    "KnowledgeDashboardResponse",
    "KnowledgeGrowthPoint",
    "KnowledgeSearchResponse",
    "KnowledgeSearchRow",
    "AIClusterMatch",
    "AIQueryRequest",
    "AIQueryResponse",
    "AISource",
    "SimilarConversationRow",
    "SimilarConversationsResponse",
    "AnalyticsInsightsResponse",
    "AutomationOpportunity",
    "ClusterSummary",
    "ConversationPreviewResponse",
    "ConversationPreviewRow",
    "EmergingIssue",
    "InsightsSummary",
    "QuestionRequest",
    "RecurringIssue",
    "RootResponse",
    "SearchMethod",
    "SearchResult",
]
