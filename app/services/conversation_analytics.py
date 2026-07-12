"""Analytics queries over persisted conversation threads."""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import func, or_

from app.core.config import settings
from app.models.database import ConversationThread
from app.services.database import SessionLocal

_CACHE_TTL_SECONDS = 20
_CACHE: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str):
    payload = _CACHE.get(key)
    if not payload:
        return None
    ts, value = payload
    if (time.time() - ts) > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any):
    _CACHE[key] = (time.time(), value)


def invalidate_analytics_cache() -> None:
    """Invalidate analytics in-memory cache after new uploads."""
    _CACHE.clear()


def get_analytics_overview() -> dict[str, Any]:
    """Return high-level KPI metrics for persisted conversations."""
    cached = _cache_get("overview")
    if cached is not None:
        return cached

    db = SessionLocal()
    try:
        total_conversations = int(
            db.query(func.count(ConversationThread.id)).scalar() or 0
        )
        total_tickets = int(
            db.query(func.count(func.distinct(ConversationThread.ticket_id))).scalar()
            or 0
        )

        duplicate = int(
            db.query(func.count(ConversationThread.id))
            .filter(ConversationThread.status == "duplicate")
            .scalar()
            or 0
        )
        similar = int(
            db.query(func.count(ConversationThread.id))
            .filter(ConversationThread.status == "similar")
            .scalar()
            or 0
        )
        unique = int(
            db.query(func.count(ConversationThread.id))
            .filter(ConversationThread.status == "unique")
            .scalar()
            or 0
        )

        total_clusters = int(
            db.query(func.count(func.distinct(ConversationThread.cluster_id))).scalar()
            or 0
        )

        payload = {
            "total_tickets": total_tickets,
            "total_conversations": total_conversations,
            "duplicate_conversations": duplicate,
            "similar_conversations": similar,
            "unique_conversations": unique,
            "total_clusters": total_clusters,
        }
        _cache_set("overview", payload)
        return payload
    finally:
        db.close()


def list_conversations(
    status: str | None = None,
    cluster_id: int | None = None,
    search: str | None = None,
    min_similarity: float | None = None,
    max_similarity: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Return filtered conversation rows for dashboard tables."""
    db = SessionLocal()
    try:
        query = db.query(ConversationThread)

        if status in {"duplicate", "similar", "unique"}:
            query = query.filter(ConversationThread.status == status)

        if cluster_id is not None:
            query = query.filter(ConversationThread.cluster_id == cluster_id)

        if search:
            search_like = f"%{search}%"
            query = query.filter(
                or_(
                    ConversationThread.ticket_id.ilike(search_like),
                    ConversationThread.subject.ilike(search_like),
                    ConversationThread.conversation_text.ilike(search_like),
                    ConversationThread.cluster_label.ilike(search_like),
                )
            )

        if min_similarity is not None:
            query = query.filter(
                ConversationThread.similarity_score >= int(min_similarity * 1000)
            )

        if max_similarity is not None:
            query = query.filter(
                ConversationThread.similarity_score <= int(max_similarity * 1000)
            )

        total = int(query.count())

        page_limit = max(1, min(limit, int(settings.max_page_size)))
        rows = (
            query.order_by(ConversationThread.id.desc())
            .offset(max(offset, 0))
            .limit(page_limit)
            .all()
        )

        items = [
            {
                "id": int(r.id),
                "ticket_id": str(r.ticket_id),
                "subject": str(r.subject or ""),
                "status": str(r.status),
                "similarity_score": round(float(r.similarity_score or 0) / 1000.0, 3),
                "nearest_ticket_id": (
                    None if not r.nearest_ticket_id else str(r.nearest_ticket_id)
                ),
                "cluster_id": int(r.cluster_id or 0),
                "cluster_label": str(r.cluster_label or "Cluster 0"),
                "message_count": int(r.message_count or 0),
                "first_sent_at": (
                    None if r.first_sent_at is None else r.first_sent_at.isoformat()
                ),
                "last_sent_at": (
                    None if r.last_sent_at is None else r.last_sent_at.isoformat()
                ),
                "representative_conversation": str(r.conversation_text or "")[:320],
                "conversation_text": str(r.conversation_text or ""),
            }
            for r in rows
        ]

        return {"total": total, "items": items}
    finally:
        db.close()


__all__ = ["get_analytics_overview", "list_conversations"]


def export_conversations(
    status: str | None = None,
    cluster_id: int | None = None,
    search: str | None = None,
    min_similarity: float | None = None,
    max_similarity: float | None = None,
) -> list[dict[str, Any]]:
    """Return full filtered conversation rows for export."""
    data = list_conversations(
        status=status,
        cluster_id=cluster_id,
        search=search,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        limit=int(settings.export_page_size),
        offset=0,
    )
    return data["items"]


def get_dashboard_charts() -> dict[str, Any]:
    """Return aggregate chart-ready analytics datasets."""
    cached = _cache_get("charts")
    if cached is not None:
        return cached

    db = SessionLocal()
    try:
        status_rows = (
            db.query(ConversationThread.status, func.count(ConversationThread.id))
            .group_by(ConversationThread.status)
            .all()
        )
        status_distribution = [
            {"status": str(status or "unknown"), "count": int(count or 0)}
            for status, count in status_rows
        ]

        cluster_rows = (
            db.query(
                ConversationThread.cluster_id,
                ConversationThread.cluster_label,
                func.count(ConversationThread.id),
            )
            .group_by(ConversationThread.cluster_id, ConversationThread.cluster_label)
            .order_by(
                func.count(ConversationThread.id).desc(),
                ConversationThread.cluster_id.asc(),
            )
            .limit(int(settings.chart_top_n))
            .all()
        )
        cluster_distribution = [
            {
                "cluster_id": int(cluster_id or 0),
                "cluster_label": str(cluster_label or "Cluster 0"),
                "count": int(count or 0),
            }
            for cluster_id, cluster_label, count in cluster_rows
        ]

        daily_rows = (
            db.query(
                func.date(ConversationThread.last_sent_at).label("day"),
                func.count(ConversationThread.id),
            )
            .filter(ConversationThread.last_sent_at.isnot(None))
            .group_by(func.date(ConversationThread.last_sent_at))
            .order_by(func.date(ConversationThread.last_sent_at).asc())
            .all()
        )
        daily_volume = [
            {"date": str(day), "count": int(count or 0)}
            for day, count in daily_rows
            if day is not None
        ]

        recurring_rows = (
            db.query(
                ConversationThread.cluster_id,
                ConversationThread.cluster_label,
                func.count(ConversationThread.id),
            )
            .group_by(ConversationThread.cluster_id, ConversationThread.cluster_label)
            .order_by(
                func.count(ConversationThread.id).desc(),
                ConversationThread.cluster_id.asc(),
            )
            .limit(int(settings.chart_top_n))
            .all()
        )
        top_recurring_issues = [
            {
                "cluster_id": int(cluster_id or 0),
                "cluster_label": str(cluster_label or "Cluster 0"),
                "count": int(count or 0),
            }
            for cluster_id, cluster_label, count in recurring_rows
        ]

        payload = {
            "status_distribution": status_distribution,
            "cluster_distribution": cluster_distribution,
            "daily_volume": daily_volume,
            "top_recurring_issues": top_recurring_issues,
        }
        _cache_set("charts", payload)
        return payload
    finally:
        db.close()


def get_conversation_detail(conversation_id: int) -> dict[str, Any] | None:
    """Return one conversation row with nearest conversation details if available."""
    db = SessionLocal()
    try:
        row = (
            db.query(ConversationThread)
            .filter(ConversationThread.id == conversation_id)
            .first()
        )
        if row is None:
            return None

        nearest = None
        if row.nearest_ticket_id:
            nearest_row = (
                db.query(ConversationThread)
                .filter(ConversationThread.ticket_id == row.nearest_ticket_id)
                .order_by(ConversationThread.id.desc())
                .first()
            )
            if nearest_row is not None:
                nearest = {
                    "id": int(nearest_row.id),
                    "ticket_id": str(nearest_row.ticket_id),
                    "subject": str(nearest_row.subject or ""),
                    "status": str(nearest_row.status),
                    "similarity_score": round(
                        float(nearest_row.similarity_score or 0) / 1000.0, 3
                    ),
                    "cluster_id": int(nearest_row.cluster_id or 0),
                    "cluster_label": str(nearest_row.cluster_label or "Cluster 0"),
                    "conversation_text": str(nearest_row.conversation_text or ""),
                }

        return {
            "id": int(row.id),
            "ticket_id": str(row.ticket_id),
            "subject": str(row.subject or ""),
            "status": str(row.status),
            "similarity_score": round(float(row.similarity_score or 0) / 1000.0, 3),
            "nearest_ticket_id": (
                None if not row.nearest_ticket_id else str(row.nearest_ticket_id)
            ),
            "cluster_id": int(row.cluster_id or 0),
            "cluster_label": str(row.cluster_label or "Cluster 0"),
            "message_count": int(row.message_count or 0),
            "first_sent_at": (
                None if row.first_sent_at is None else row.first_sent_at.isoformat()
            ),
            "last_sent_at": (
                None if row.last_sent_at is None else row.last_sent_at.isoformat()
            ),
            "conversation_text": str(row.conversation_text or ""),
            "nearest_conversation": nearest,
        }
    finally:
        db.close()


__all__ = [
    "export_conversations",
    "get_analytics_overview",
    "get_dashboard_charts",
    "get_conversation_detail",
    "invalidate_analytics_cache",
    "list_conversations",
]
