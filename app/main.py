"""FastAPI entrypoint for QA Knowledge System."""

from __future__ import annotations

import time
import uuid
import re
from io import BytesIO, StringIO
from typing import Any

from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.requests import Request
import pandas as pd

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    AIQueryRequest,
    AIQueryResponse,
    AskResponse,
    AnalyticsInsightsResponse,
    AnalyticsOverviewResponse,
    ConversationTableResponse,
    ConversationImportReport,
    ConversationPreviewResponse,
    ConversationPreviewRow,
    KnowledgeDashboardResponse,
    KnowledgeSearchResponse,
    SimilarConversationsResponse,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    RootResponse,
    SearchResult,
)
from app.search.hybrid import search_hybrid
from app.services.conversation_ingestion import (
    REQUIRED_CONVERSATION_COLUMNS,
    REQUIRED_SHEETS,
    load_conversations_sheet_from_bytes,
    reconstruct_conversations,
    summarize_conversation_upload,
)
from app.services.conversation_summary import summarize_conversations
from app.services.conversation_classification import (
    classify_conversations,
    classification_counts,
)
from app.services.conversation_clustering import cluster_conversations, cluster_count
from app.services.cluster_labeling import assign_cluster_labels
from app.services.conversation_analytics import (
    export_conversations,
    get_dashboard_charts,
    get_analytics_overview,
    get_conversation_detail,
    invalidate_analytics_cache,
    list_conversations,
)
from app.services.conversation_insights import (
    InsightsConfig,
    generate_business_insights,
)
from app.services.knowledge_base import (
    get_knowledge_dashboard,
    get_similar_historical_conversations,
    ingest_conversations_to_knowledge_base,
    search_knowledge_base,
)
from app.services.database import save_conversations_to_db
from app.services.ai_assistant import run_ai_query
from pathlib import Path

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
    },
    responses={
        404: {"model": ErrorResponse, "description": "No matching answer found."},
        422: {"model": ErrorResponse, "description": "Validation error."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed id=%s method=%s path=%s elapsed_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_done id=%s method=%s path=%s status=%d elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


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


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Root"],
    summary="API info",
)
def home():
    return {
        "message": "Q&A Knowledge System API is running!",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/ui",
    tags=["Root"],
    summary="Minimal frontend UI",
)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


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


@app.get(
    "/analytics/charts",
    tags=["Analytics"],
    summary="Chart datasets for dashboard visualizations",
)
def analytics_charts():
    """Return chart-ready datasets for status, clusters, volume, and recurring issues."""
    data = get_dashboard_charts()
    logger.info(
        "analytics_charts_generated status=%d clusters=%d",
        len(data.get("status_distribution", [])),
        len(data.get("cluster_distribution", [])),
    )
    return data


@app.get(
    "/knowledge/dashboard",
    response_model=KnowledgeDashboardResponse,
    tags=["Knowledge"],
    summary="Knowledge base dashboard metrics",
)
def knowledge_dashboard():
    """Return accumulated knowledge base metrics and historical growth."""
    data = get_knowledge_dashboard()
    logger.info(
        "knowledge_dashboard_generated total_uploads=%s kb_size=%s",
        data.get("total_uploads"),
        data.get("knowledge_base_size"),
    )
    return KnowledgeDashboardResponse(**data)


@app.get(
    "/knowledge/search",
    response_model=KnowledgeSearchResponse,
    tags=["Knowledge"],
    summary="Search accumulated historical conversations",
)
def knowledge_search(
    query: str | None = Query(default=None),
    ticket: str | None = Query(default=None),
    cluster_id: int | None = Query(default=None),
    cluster_label: str | None = Query(default=None),
    keywords: str | None = Query(default=None),
    upload_batch: str | None = Query(default=None),
    min_similarity: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Search the historical knowledge base by metadata and semantic similarity."""
    data = search_knowledge_base(
        query=query,
        ticket=ticket,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        keywords=keywords,
        upload_batch=upload_batch,
        min_similarity=min_similarity,
        limit=limit,
    )
    logger.info(
        "knowledge_search_done total=%d query=%r ticket=%r cluster_id=%r",
        int(data.get("total", 0)),
        query,
        ticket,
        cluster_id,
    )
    return KnowledgeSearchResponse(**data)


@app.get(
    "/knowledge/similar/{conversation_id}",
    response_model=SimilarConversationsResponse,
    tags=["Knowledge"],
    summary="Find nearest historical conversations",
)
def knowledge_similar(
    conversation_id: str, top_n: int = Query(default=10, ge=1, le=50)
):
    """Return nearest historical conversations for a conversation ID."""
    items = get_similar_historical_conversations(
        conversation_id=conversation_id, top_n=top_n
    )
    if not items:
        # Keep backward compatibility with empty data behavior; 404 only if specific ID has no neighbors/record.
        exists = search_knowledge_base(query=None, limit=1)
        if exists.get("total", 0) == 0:
            return SimilarConversationsResponse(
                conversation_id=conversation_id, top_n=top_n, items=[]
            )
        raise HTTPException(
            status_code=404, detail="Conversation not found in knowledge base."
        )
    logger.info(
        "knowledge_similar_done conversation_id=%s total=%d",
        conversation_id,
        len(items),
    )
    return SimilarConversationsResponse(
        conversation_id=conversation_id, top_n=top_n, items=items
    )


@app.post(
    "/ai/query",
    response_model=AIQueryResponse,
    tags=["AI"],
    summary="Ask natural-language questions over historical conversation intelligence",
)
def ai_query(request: AIQueryRequest):
    """Run retrieval-first AI analyst query with optional LLM generation."""
    try:
        data = run_ai_query(question=request.question, limit=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception:
        logger.exception("ai_query failed question=%r", request.question)
        raise HTTPException(status_code=500, detail="AI query failed.") from None

    logger.info(
        "ai_query_done mode=%s confidence=%s sources=%d",
        data.get("generation_mode"),
        data.get("confidence_score"),
        len(data.get("sources", [])),
    )
    return AIQueryResponse(**data)


@app.get(
    "/analytics/insights",
    response_model=AnalyticsInsightsResponse,
    tags=["Analytics"],
    summary="Deterministic business insights",
)
def analytics_insights(
    duplicate_rate_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
    large_cluster_min_size: int = Query(default=8, ge=1, le=10000),
    mostly_unique_threshold: float = Query(default=0.70, ge=0.0, le=1.0),
    rapid_growth_multiplier: float = Query(default=1.8, ge=1.0, le=20.0),
    rapid_growth_window_days: int = Query(default=7, ge=1, le=90),
    rapid_growth_min_recent: int = Query(default=3, ge=1, le=10000),
):
    """Return deterministic business insights built from persisted conversation analytics."""
    config = InsightsConfig(
        duplicate_rate_threshold=duplicate_rate_threshold,
        large_cluster_min_size=large_cluster_min_size,
        mostly_unique_threshold=mostly_unique_threshold,
        rapid_growth_multiplier=rapid_growth_multiplier,
        rapid_growth_window_days=rapid_growth_window_days,
        rapid_growth_min_recent=rapid_growth_min_recent,
    )
    payload = generate_business_insights(config=config)
    logger.info(
        "analytics_insights_generated recurring=%d opportunities=%d",
        len(payload.get("recurring_issues", [])),
        len(payload.get("automation_opportunities", [])),
    )
    return AnalyticsInsightsResponse(**payload)


@app.get(
    "/analytics/overview",
    response_model=AnalyticsOverviewResponse,
    tags=["Analytics"],
    summary="Dashboard KPI overview",
)
def analytics_overview():
    """Return core dashboard KPIs from persisted conversation data."""
    data = get_analytics_overview()
    logger.info(
        "analytics_overview_generated total_conversations=%s",
        data.get("total_conversations"),
    )
    return AnalyticsOverviewResponse(**data)


@app.get(
    "/analytics/conversations",
    response_model=ConversationTableResponse,
    tags=["Analytics"],
    summary="Filtered conversation analytics table",
)
def analytics_conversations(
    status: str | None = Query(default=None),
    cluster_id: int | None = Query(default=None),
    search: str | None = Query(default=None),
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
    max_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(
        default=int(settings.default_page_size), ge=1, le=int(settings.max_page_size)
    ),
    offset: int = Query(default=0, ge=0),
):
    """Return dashboard table rows with status/cluster/search/similarity filtering."""
    data = list_conversations(
        status=status,
        cluster_id=cluster_id,
        search=search,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        limit=limit,
        offset=offset,
    )
    return ConversationTableResponse(**data)


@app.get(
    "/analytics/conversations/{conversation_id}",
    tags=["Analytics"],
    summary="Conversation detail with nearest match",
)
def analytics_conversation_detail(conversation_id: int):
    """Return one conversation detail payload, including nearest conversation context."""
    data = get_conversation_detail(conversation_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return data


@app.get(
    "/analytics/export/csv",
    tags=["Analytics"],
    summary="Export filtered conversations as CSV",
)
def export_analytics_csv(
    status: str | None = Query(default=None),
    cluster_id: int | None = Query(default=None),
    search: str | None = Query(default=None),
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
    max_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
):
    rows = export_conversations(
        status=status,
        cluster_id=cluster_id,
        search=search,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
    )
    df = pd.DataFrame(rows)
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=conversation_analytics.csv"
        },
    )


@app.get(
    "/analytics/export/excel",
    tags=["Analytics"],
    summary="Export filtered conversations as Excel",
)
def export_analytics_excel(
    status: str | None = Query(default=None),
    cluster_id: int | None = Query(default=None),
    search: str | None = Query(default=None),
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
    max_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
):
    rows = export_conversations(
        status=status,
        cluster_id=cluster_id,
        search=search,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
    )
    df = pd.DataFrame(rows)
    stream = BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analytics")
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=conversation_analytics.xlsx"
        },
    )


@app.post(
    "/admin/conversations/upload",
    response_model=ConversationPreviewResponse,
    tags=["Admin"],
    summary="Upload a conversation Excel file and preview reconstructed conversations",
)
async def upload_conversations_excel(
    file: UploadFile = File(...),
    preview_count: int = Query(default=5, ge=1, le=50),
):
    """Validate an uploaded conversation Excel file and return a preview of reconstructed conversations."""
    filename = file.filename or "uploaded.xlsx"
    sanitized_name = _sanitize_filename(filename)
    ext = "." + sanitized_name.split(".")[-1].lower() if "." in sanitized_name else ""

    if ext not in settings.allowed_upload_exts:
        raise HTTPException(
            status_code=400, detail="Please upload an Excel file (.xlsx or .xls)."
        )

    try:
        content = await file.read()
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max upload size is {settings.max_upload_size_mb} MB.",
            )

        upload_batch = f"batch_{uuid.uuid4().hex[:12]}"
        source_df = load_conversations_sheet_from_bytes(
            content, file_name=sanitized_name
        )
        reconstructed = reconstruct_conversations(source_df)
        summarized = summarize_conversations(reconstructed)
        classified = classify_conversations(summarized)
        clustered = cluster_conversations(classified)
        labeled = assign_cluster_labels(clustered)
        counts = classification_counts(labeled)
        total_clusters = cluster_count(labeled)
        summary = summarize_conversation_upload(source_df)
        stored_count = save_conversations_to_db(labeled, source_file=sanitized_name)
        ingest_conversations_to_knowledge_base(labeled, upload_batch=upload_batch)
        invalidate_analytics_cache()
        logger.info(
            "upload_completed file=%s rows=%d upload_batch=%s",
            sanitized_name,
            len(labeled),
            upload_batch,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("conversation upload failed filename=%s", sanitized_name)
        raise HTTPException(
            status_code=500, detail="Conversation upload failed."
        ) from None

    preview_df = labeled.head(preview_count)
    preview_rows = [
        ConversationPreviewRow(
            ticket_id=str(row["ticket_id"]),
            subject=str(row["subject"]),
            message_count=int(row["message_count"]),
            status=str(row.get("status", "unique")),
            similarity_score=float(row.get("similarity_score", 0.0)),
            nearest_ticket_id=(
                None
                if pd.isna(row.get("nearest_ticket_id"))
                else str(row.get("nearest_ticket_id"))
            ),
            cluster_id=int(row.get("cluster_id", 0)),
            cluster_label=str(row.get("cluster_label", "Cluster 0")),
            first_sent_at=(
                None
                if pd.isna(row["first_sent_at"])
                else row["first_sent_at"].isoformat()
            ),
            last_sent_at=(
                None
                if pd.isna(row["last_sent_at"])
                else row["last_sent_at"].isoformat()
            ),
            conversation_text=str(row["conversation_text"]),
            summary=str(row.get("summary", "") or ""),
            intent=str(row.get("intent", "") or ""),
            keywords=str(row.get("keywords", "") or ""),
            category=str(row.get("category", "") or ""),
            sentiment=str(row.get("sentiment", "neutral") or "neutral"),
            priority=str(row.get("priority", "low") or "low"),
        )
        for _, row in preview_df.iterrows()
    ]

    return ConversationPreviewResponse(
        report=ConversationImportReport(
            file_name=sanitized_name,
            total_source_rows=summary["total_source_rows"],
            valid_conversations=int(len(classified)),
            stored_conversations=int(stored_count),
            unique_ticket_ids=summary["unique_ticket_ids"],
            duplicate_conversations=counts["duplicate"],
            similar_conversations=counts["similar"],
            unique_conversations=counts["unique"],
            total_clusters=int(total_clusters),
            invalid_rows=summary["invalid_rows"],
            required_sheets=REQUIRED_SHEETS,
            required_columns=REQUIRED_CONVERSATION_COLUMNS,
        ),
        preview_rows=preview_rows,
        preview_count=int(len(preview_rows)),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Knowledge Base"],
    summary="Search the knowledge base",
    description="Accepts a user question and returns ranked matches with confidence scores.",
    responses={
        200: {"description": "Matching answer found.", "model": AskResponse},
        404: {"description": "No matching answer found.", "model": ErrorResponse},
        422: {"description": "Invalid request payload.", "model": ErrorResponse},
        500: {"description": "Internal search error.", "model": ErrorResponse},
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


def _sanitize_filename(name: str) -> str:
    """Sanitize upload filenames to safe ASCII-ish names."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name or "uploaded.xlsx")
    cleaned = cleaned.strip("._")
    if not cleaned:
        return "uploaded.xlsx"
    return cleaned[:180]


__all__ = ["app", "ask_question", "health", "home"]
