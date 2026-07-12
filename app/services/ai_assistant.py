"""Retrieval-first AI analyst assistant over the historical knowledge base."""

from __future__ import annotations

import json
import os
import importlib
from collections import Counter
from typing import Any

from app.core.logging import logger
from app.services.conversation_analytics import get_analytics_overview
from app.services.conversation_insights import generate_business_insights
from app.services.knowledge_base import search_knowledge_base

DEFAULT_MIN_SEMANTIC = 0.12


def retrieve_ai_context(question: str, limit: int = 10) -> dict[str, Any]:
    """Retrieve relevant knowledge conversations and supporting analytics context."""
    q = (question or "").strip()
    if not q:
        raise ValueError("Question cannot be empty.")

    query_limit = max(5, min(limit * 3, 50))
    retrieved = search_knowledge_base(
        query=q,
        limit=query_limit,
        min_similarity=DEFAULT_MIN_SEMANTIC,
    )

    items = list(retrieved.get("items", []))
    if not items:
        keyword_query = _extract_keyword_hint(q)
        if keyword_query:
            retrieved = search_knowledge_base(
                query=None,
                keywords=keyword_query,
                limit=query_limit,
            )
            items = list(retrieved.get("items", []))

    filtered = _apply_question_filters(items, q)
    selected = filtered[: max(1, min(limit, 20))]

    matching_clusters = _build_cluster_matches(selected)
    sources = _build_sources(selected)
    confidence = _compute_confidence(selected)

    analytics_overview = get_analytics_overview()
    insights = generate_business_insights()

    return {
        "question": q,
        "matching_conversations": selected,
        "matching_clusters": matching_clusters,
        "sources": sources,
        "confidence_score": confidence,
        "analytics_overview": analytics_overview,
        "insights": insights,
    }


def generate_ai_answer(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """Generate answer from retrieved context using optional LLM or deterministic summary."""
    llm_answer = _generate_with_optional_llm(question, context)
    if llm_answer:
        answer = _attach_citations(llm_answer, context.get("sources", []))
        return {
            "answer": answer,
            "generation_mode": "llm",
        }

    deterministic = _generate_deterministic_answer(question, context)
    answer = _attach_citations(deterministic, context.get("sources", []))
    return {
        "answer": answer,
        "generation_mode": "deterministic",
    }


def run_ai_query(question: str, limit: int = 10) -> dict[str, Any]:
    """Run full AI analyst query pipeline."""
    context = retrieve_ai_context(question=question, limit=limit)
    generated = generate_ai_answer(question=question, context=context)

    return {
        "answer": generated["answer"],
        "sources": context["sources"],
        "matching_conversations": context["matching_conversations"],
        "matching_clusters": context["matching_clusters"],
        "confidence_score": context["confidence_score"],
        "generation_mode": generated["generation_mode"],
    }


def _generate_deterministic_answer(question: str, context: dict[str, Any]) -> str:
    items = context.get("matching_conversations", [])
    if not items:
        return "I could not find relevant historical conversations for this question."

    q = question.lower()
    top_clusters = context.get("matching_clusters", [])
    top_cluster_text = (
        f"Top cluster: {top_clusters[0]['cluster_label']} (ID {top_clusters[0]['cluster_id']}, {top_clusters[0]['conversation_count']} matches)."
        if top_clusters
        else ""
    )

    if "duplicate" in q:
        duplicate_count = sum(
            1
            for x in items
            if str(x.get("classification", ""))
            in {"existing_intent", "similar_historical"}
        )
        pct = round((duplicate_count / max(len(items), 1)) * 100.0, 1)
        return f"Among retrieved conversations, {duplicate_count} out of {len(items)} ({pct}%) appear to be known or repetitive issues. {top_cluster_text}".strip()

    if "seen" in q or "before" in q:
        known = [
            x
            for x in items
            if str(x.get("classification", ""))
            in {"existing_intent", "similar_historical"}
        ]
        if known:
            return f"Yes. This issue pattern appears in {len(known)} retrieved historical conversations. {top_cluster_text}".strip()
        return "No strong historical match was found in the current knowledge base for this issue."

    if "refund" in q:
        refund_hits = [
            x
            for x in items
            if "refund" in str(x.get("cluster_label", "")).lower()
            or "refund" in str(x.get("subject", "")).lower()
        ]
        if refund_hits:
            return f"I found {len(refund_hits)} refund-related conversations in the retrieved context. {top_cluster_text}".strip()
        return (
            "I did not find refund-related matches in the retrieved conversation set."
        )

    if "cluster" in q and "belong" in q:
        if top_clusters:
            return f"Based on retrieval, this question most likely belongs to cluster '{top_clusters[0]['cluster_label']}' (ID {top_clusters[0]['cluster_id']})."
        return "No cluster match could be inferred from available conversations."

    top_subjects = Counter(
        str(x.get("subject", "")).strip()
        for x in items
        if str(x.get("subject", "")).strip()
    )
    top_subject = (
        top_subjects.most_common(1)[0][0] if top_subjects else "general support issues"
    )

    return (
        f"The strongest signal is around '{top_subject}'. "
        f"I retrieved {len(items)} relevant conversations and {len(top_clusters)} matching clusters. "
        f"{top_cluster_text}"
    ).strip()


def _generate_with_optional_llm(question: str, context: dict[str, Any]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("AI_ASSISTANT_LLM_MODEL")
    if not api_key or not model:
        return None

    try:
        openai_module = importlib.import_module("openai")
        OpenAI = getattr(openai_module, "OpenAI")
    except Exception:
        logger.warning(
            "LLM requested but openai package is unavailable; using deterministic mode."
        )
        return None

    try:
        client = OpenAI(api_key=api_key)
        compact_context = {
            "matching_conversations": [
                {
                    "conversation_id": x.get("conversation_id"),
                    "ticket_id": x.get("ticket_id"),
                    "subject": x.get("subject"),
                    "cluster_id": x.get("cluster_id"),
                    "cluster_label": x.get("cluster_label"),
                    "classification": x.get("classification"),
                    "semantic_similarity": x.get("semantic_similarity"),
                    "summary": x.get("summary"),
                    "intent": x.get("intent"),
                    "category": x.get("category"),
                }
                for x in context.get("matching_conversations", [])[:10]
            ],
            "matching_clusters": context.get("matching_clusters", []),
            "analytics_overview": context.get("analytics_overview", {}),
            "insights_summary": context.get("insights", {}).get("summary", {}),
        }

        prompt = (
            "You are an analytics assistant. Answer ONLY using the provided JSON context. "
            "If evidence is weak, say so. Do not invent facts. Keep answer concise.\n"
            f"Question: {question}\n"
            f"Context JSON: {json.dumps(compact_context)}"
        )

        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "Use only supplied context. No hallucinations.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        text = (completion.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        logger.exception(
            "Optional LLM generation failed; falling back to deterministic answer."
        )
        return None


def _attach_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return f"{answer} Sources: none."
    refs = ", ".join(str(s.get("conversation_id")) for s in sources[:5])
    return f"{answer} Sources: {refs}."


def _build_cluster_matches(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cluster: dict[tuple[int, str], int] = {}
    for item in items:
        cid = int(item.get("cluster_id", 0))
        label = str(item.get("cluster_label", "Cluster 0"))
        key = (cid, label)
        by_cluster[key] = by_cluster.get(key, 0) + 1

    rows = [
        {
            "cluster_id": cid,
            "cluster_label": label,
            "conversation_count": count,
        }
        for (cid, label), count in by_cluster.items()
    ]
    rows.sort(key=lambda x: (-x["conversation_count"], x["cluster_id"]))
    return rows[:10]


def _build_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for item in items:
        cid = str(item.get("conversation_id", ""))
        if not cid or cid in seen:
            continue
        seen.add(cid)
        sources.append(
            {
                "conversation_id": cid,
                "ticket_id": str(item.get("ticket_id", "")),
                "cluster_id": int(item.get("cluster_id", 0)),
                "cluster_label": str(item.get("cluster_label", "Cluster 0")),
                "upload_batch": str(item.get("upload_batch", "")),
                "semantic_similarity": (
                    None
                    if item.get("semantic_similarity") is None
                    else float(item.get("semantic_similarity"))
                ),
            }
        )
    return sources[:10]


def _compute_confidence(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    sims = [float(x.get("semantic_similarity", 0.0) or 0.0) for x in items]
    sims.sort(reverse=True)
    top = sims[:3]
    if not top:
        return 0.0
    return round(float(sum(top) / len(top)), 3)


def _extract_keyword_hint(question: str) -> str | None:
    q = question.lower()
    hints = []
    for token in [
        "refund",
        "booking",
        "password",
        "parking",
        "duplicate",
        "billing",
        "login",
    ]:
        if token in q:
            hints.append(token)
    if not hints:
        return None
    return ",".join(hints)


def _apply_question_filters(
    items: list[dict[str, Any]], question: str
) -> list[dict[str, Any]]:
    q = question.lower()
    filtered = list(items)

    if "duplicate" in q:
        subset = [
            x
            for x in filtered
            if str(x.get("classification", ""))
            in {"existing_intent", "similar_historical"}
        ]
        if subset:
            filtered = subset

    if "refund" in q:
        subset = [
            x
            for x in filtered
            if "refund" in str(x.get("cluster_label", "")).lower()
            or "refund" in str(x.get("subject", "")).lower()
        ]
        if subset:
            filtered = subset

    filtered.sort(
        key=lambda x: (
            -(float(x.get("semantic_similarity", 0.0) or 0.0)),
            x.get("conversation_id", ""),
        )
    )
    return filtered


__all__ = ["generate_ai_answer", "retrieve_ai_context", "run_ai_query"]
