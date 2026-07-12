"""Persistent knowledge base services for historical conversation learning."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, or_

from app.core.config import settings
from app.core.logging import logger
from app.models.database import KnowledgeConversation
from app.services.conversation_classification import _get_model
from app.services.database import (
    SessionLocal,
    _ensure_knowledge_conversation_schema,
    init_db,
)

EMBED_SCALE = 1_000
KNOWN_THRESHOLD = float(settings.known_threshold)
VARIATION_THRESHOLD = float(settings.variation_threshold)


def ingest_conversations_to_knowledge_base(
    conversations_df: pd.DataFrame,
    upload_batch: str,
    upload_timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Persist processed conversations and classify them against historical records."""
    if conversations_df.empty:
        return {
            "upload_batch": upload_batch,
            "saved": 0,
            "existing_intent": 0,
            "similar_historical": 0,
            "new_intent": 0,
        }

    timestamp = upload_timestamp or datetime.utcnow()
    work = conversations_df.copy()
    init_db()
    _ensure_knowledge_conversation_schema()

    text_col = "embedding_text" if "embedding_text" in work.columns else "conversation_text"
    texts = [str(x or "") for x in work[text_col].tolist()]
    embeddings = _encode_texts(texts)

    db = SessionLocal()
    try:
        history = _load_history_embeddings(db)

        existing_intent = 0
        similar_historical = 0
        new_intent = 0

        rows: list[KnowledgeConversation] = []
        for idx, row in work.reset_index(drop=True).iterrows():
            vector = embeddings[idx]
            classification, similarity, matched = _classify_against_history(
                vector, history
            )

            if classification == "existing_intent":
                existing_intent += 1
            elif classification == "similar_historical":
                similar_historical += 1
            else:
                new_intent += 1

            conversation_id = (
                f"{upload_batch}:{idx + 1}:{str(row.get('ticket_id', 'unknown'))}"
            )

            entity = KnowledgeConversation(
                conversation_id=conversation_id,
                ticket_id=str(row.get("ticket_id", "")),
                subject=str(row.get("subject", "")),
                reconstructed_conversation=str(row.get("conversation_text", "")),
                embedding=json.dumps(vector.tolist()),
                cluster_id=int(row.get("cluster_id", 0)),
                cluster_label=str(row.get("cluster_label", "Cluster 0")),
                similarity=int(round(float(similarity) * EMBED_SCALE)),
                classification=classification,
                summary=str(row.get("summary", "") or ""),
                intent=str(row.get("intent", "") or ""),
                keywords=str(row.get("keywords", "") or ""),
                category=str(row.get("category", "") or ""),
                sentiment=str(row.get("sentiment", "neutral") or "neutral"),
                priority=str(row.get("priority", "low") or "low"),
                upload_batch=upload_batch,
                upload_timestamp=timestamp,
            )
            rows.append(entity)

        db.add_all(rows)
        db.commit()

        logger.info(
            "knowledge_base_ingest upload_batch=%s saved=%d existing=%d variation=%d new=%d",
            upload_batch,
            len(rows),
            existing_intent,
            similar_historical,
            new_intent,
        )
        return {
            "upload_batch": upload_batch,
            "saved": len(rows),
            "existing_intent": existing_intent,
            "similar_historical": similar_historical,
            "new_intent": new_intent,
        }
    except Exception:
        db.rollback()
        logger.exception("knowledge base ingest failed upload_batch=%s", upload_batch)
        raise
    finally:
        db.close()


def search_knowledge_base(
    query: str | None = None,
    ticket: str | None = None,
    cluster_id: int | None = None,
    cluster_label: str | None = None,
    keywords: str | None = None,
    upload_batch: str | None = None,
    limit: int = 20,
    min_similarity: float = 0.0,
) -> dict[str, Any]:
    """Search persistent knowledge records by metadata and optional semantic similarity."""
    init_db()
    db = SessionLocal()
    try:
        q = db.query(KnowledgeConversation)

        if ticket:
            q = q.filter(KnowledgeConversation.ticket_id.ilike(f"%{ticket}%"))
        if cluster_id is not None:
            q = q.filter(KnowledgeConversation.cluster_id == cluster_id)
        if cluster_label:
            q = q.filter(
                KnowledgeConversation.cluster_label.ilike(f"%{cluster_label}%")
            )
        if upload_batch:
            q = q.filter(KnowledgeConversation.upload_batch == upload_batch)

        if keywords:
            terms = [x.strip() for x in keywords.split(",") if x.strip()]
            if terms:
                keyword_filters = []
                for term in terms:
                    like = f"%{term}%"
                    keyword_filters.append(
                        or_(
                            KnowledgeConversation.subject.ilike(like),
                            KnowledgeConversation.reconstructed_conversation.ilike(
                                like
                            ),
                            KnowledgeConversation.cluster_label.ilike(like),
                            KnowledgeConversation.summary.ilike(like),
                            KnowledgeConversation.intent.ilike(like),
                            KnowledgeConversation.keywords.ilike(like),
                            KnowledgeConversation.category.ilike(like),
                        )
                    )
                q = q.filter(and_(*keyword_filters))

        candidates = (
            q.order_by(KnowledgeConversation.upload_timestamp.desc())
            .limit(int(settings.knowledge_search_scan_limit))
            .all()
        )
        items = [_to_search_item(x) for x in candidates]

        if query:
            query_vec = _encode_texts([query])[0]
            for item, entity in zip(items, candidates):
                emb = _parse_embedding(entity.embedding)
                score = _cosine_similarity(query_vec, emb)
                item["semantic_similarity"] = round(score, 4)
            items = [
                x
                for x in items
                if x.get("semantic_similarity", 0.0) >= max(min_similarity, 0.0)
            ]
            items.sort(
                key=lambda x: (-x.get("semantic_similarity", 0.0), x["conversation_id"])
            )
        else:
            for item in items:
                item["semantic_similarity"] = None

        items = items[: max(1, min(limit, 100))]
        return {
            "total": len(items),
            "items": items,
        }
    finally:
        db.close()


def get_similar_historical_conversations(
    conversation_id: str, top_n: int = 10
) -> list[dict[str, Any]]:
    """Return nearest historical conversations for one conversation ID."""
    init_db()
    db = SessionLocal()
    try:
        target = (
            db.query(KnowledgeConversation)
            .filter(KnowledgeConversation.conversation_id == conversation_id)
            .first()
        )
        if target is None:
            return []

        target_emb = _parse_embedding(target.embedding)

        rows = (
            db.query(KnowledgeConversation)
            .filter(KnowledgeConversation.conversation_id != conversation_id)
            .order_by(KnowledgeConversation.upload_timestamp.desc())
            .limit(int(settings.knowledge_similar_scan_limit))
            .all()
        )

        scored: list[dict[str, Any]] = []
        for row in rows:
            score = _cosine_similarity(target_emb, _parse_embedding(row.embedding))
            scored.append(
                {
                    "conversation_id": str(row.conversation_id),
                    "ticket_id": str(row.ticket_id),
                    "cluster_id": int(row.cluster_id),
                    "cluster_label": str(row.cluster_label),
                    "upload_batch": str(row.upload_batch),
                    "classification": str(row.classification),
                    "similarity": round(score, 4),
                    "subject": str(row.subject or ""),
                }
            )

        scored.sort(key=lambda x: (-x["similarity"], x["conversation_id"]))
        return scored[: max(1, min(top_n, 50))]
    finally:
        db.close()


def get_knowledge_dashboard() -> dict[str, Any]:
    """Return aggregate knowledge growth and learning metrics."""
    init_db()
    db = SessionLocal()
    try:
        total_uploads = int(
            db.query(
                func.count(func.distinct(KnowledgeConversation.upload_batch))
            ).scalar()
            or 0
        )
        knowledge_base_size = int(
            db.query(func.count(KnowledgeConversation.id)).scalar() or 0
        )

        known_conversations = int(
            db.query(func.count(KnowledgeConversation.id))
            .filter(
                KnowledgeConversation.classification.in_(
                    ["existing_intent", "similar_historical"]
                )
            )
            .scalar()
            or 0
        )
        new_conversations = int(
            db.query(func.count(KnowledgeConversation.id))
            .filter(KnowledgeConversation.classification == "new_intent")
            .scalar()
            or 0
        )
        new_intents_discovered = int(
            db.query(func.count(func.distinct(KnowledgeConversation.cluster_label)))
            .filter(KnowledgeConversation.classification == "new_intent")
            .scalar()
            or 0
        )

        rows = (
            db.query(KnowledgeConversation.upload_timestamp)
            .order_by(KnowledgeConversation.upload_timestamp.asc())
            .all()
        )

        by_day: dict[str, int] = defaultdict(int)
        for row in rows:
            if row.upload_timestamp is None:
                continue
            key = row.upload_timestamp.date().isoformat()
            by_day[key] += 1

        cumulative = 0
        growth: list[dict[str, Any]] = []
        for day in sorted(by_day.keys()):
            cumulative += int(by_day[day])
            growth.append(
                {
                    "date": day,
                    "new_records": int(by_day[day]),
                    "cumulative_records": cumulative,
                }
            )

        return {
            "total_uploads": total_uploads,
            "knowledge_base_size": knowledge_base_size,
            "new_conversations": new_conversations,
            "known_conversations": known_conversations,
            "new_intents_discovered": new_intents_discovered,
            "historical_growth": growth,
        }
    finally:
        db.close()


def _to_search_item(row: KnowledgeConversation) -> dict[str, Any]:
    return {
        "conversation_id": str(row.conversation_id),
        "ticket_id": str(row.ticket_id),
        "subject": str(row.subject or ""),
        "reconstructed_conversation": str(row.reconstructed_conversation or ""),
        "cluster_id": int(row.cluster_id or 0),
        "cluster_label": str(row.cluster_label or "Cluster 0"),
        "similarity": round(float(row.similarity or 0) / EMBED_SCALE, 4),
        "classification": str(row.classification or "new_intent"),
        "summary": str(row.summary or ""),
        "intent": str(row.intent or ""),
        "keywords": str(row.keywords or ""),
        "category": str(row.category or ""),
        "sentiment": str(row.sentiment or "neutral"),
        "priority": str(row.priority or "low"),
        "upload_batch": str(row.upload_batch),
        "upload_timestamp": (
            None if row.upload_timestamp is None else row.upload_timestamp.isoformat()
        ),
    }


def _encode_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def _load_history_embeddings(db) -> list[dict[str, Any]]:
    rows = db.query(KnowledgeConversation).all()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "conversation_id": str(row.conversation_id),
                "ticket_id": str(row.ticket_id),
                "cluster_id": int(row.cluster_id or 0),
                "cluster_label": str(row.cluster_label or "Cluster 0"),
                "upload_batch": str(row.upload_batch),
                "embedding": _parse_embedding(row.embedding),
            }
        )
    return out


def _classify_against_history(
    vector: np.ndarray, history: list[dict[str, Any]]
) -> tuple[str, float, dict[str, Any] | None]:
    if not history:
        return "new_intent", 0.0, None

    scores = [_cosine_similarity(vector, item["embedding"]) for item in history]

    best_idx = int(np.argmax(np.asarray(scores)))
    best = float(scores[best_idx])
    matched = history[best_idx]

    if best >= KNOWN_THRESHOLD:
        return "existing_intent", best, matched
    if best >= VARIATION_THRESHOLD:
        return "similar_historical", best, matched
    return "new_intent", best, matched


def _parse_embedding(raw: str) -> np.ndarray:
    try:
        arr = json.loads(raw)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return np.asarray([], dtype=np.float32)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.size == 0 or v2.size == 0:
        return 0.0
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


__all__ = [
    "get_knowledge_dashboard",
    "get_similar_historical_conversations",
    "ingest_conversations_to_knowledge_base",
    "search_knowledge_base",
]
