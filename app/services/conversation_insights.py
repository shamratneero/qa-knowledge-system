"""Deterministic business insights over persisted conversation analytics data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from app.services.conversation_analytics import (
    get_analytics_overview,
    list_conversations,
)


@dataclass(slots=True)
class InsightsConfig:
    """Configurable thresholds for deterministic insight rules."""

    duplicate_rate_threshold: float = 0.35
    large_cluster_min_size: int = 8
    mostly_unique_threshold: float = 0.70
    rapid_growth_multiplier: float = 1.8
    rapid_growth_window_days: int = 7
    rapid_growth_min_recent: int = 3


def generate_business_insights(config: InsightsConfig | None = None) -> dict[str, Any]:
    """Generate deterministic business insights from all stored conversations."""
    cfg = config or InsightsConfig()

    overview = get_analytics_overview()
    rows = _fetch_all_conversations()

    if not rows:
        return {
            "summary": {
                **overview,
                "redundancy_percentage": 0.0,
                "largest_cluster": None,
                "smallest_cluster": None,
                "average_similarity_score": 0.0,
                "median_similarity_score": 0.0,
                "average_messages_per_ticket": 0.0,
            },
            "recurring_issues": [],
            "automation_opportunities": [],
            "emerging_issues": [],
            "recommendations": [
                "No conversation data is available yet. Upload an Excel file to generate insights."
            ],
        }

    df = pd.DataFrame(rows)
    df["status"] = df["status"].fillna("unique").astype(str)
    df["cluster_id"] = (
        pd.to_numeric(df["cluster_id"], errors="coerce").fillna(0).astype(int)
    )
    df["cluster_label"] = df["cluster_label"].fillna("Cluster 0").astype(str)
    df["message_count"] = (
        pd.to_numeric(df["message_count"], errors="coerce").fillna(0).astype(int)
    )
    df["similarity_score"] = (
        pd.to_numeric(df["similarity_score"], errors="coerce").fillna(0.0).astype(float)
    )

    total = int(len(df))
    duplicate_count = int((df["status"] == "duplicate").sum())
    similar_count = int((df["status"] == "similar").sum())
    redundancy_percentage = (
        round(((duplicate_count + similar_count) / total) * 100.0, 2)
        if total > 0
        else 0.0
    )

    cluster_stats = _cluster_statistics(df)
    recurring_issues = _build_recurring_issues(df)
    automation_opportunities = _build_automation_opportunities(df, cfg)
    emerging_issues = _build_emerging_issues(df, cfg)

    summary = {
        **overview,
        "redundancy_percentage": redundancy_percentage,
        "largest_cluster": cluster_stats["largest_cluster"],
        "smallest_cluster": cluster_stats["smallest_cluster"],
        "average_similarity_score": round(float(df["similarity_score"].mean()), 3),
        "median_similarity_score": round(float(df["similarity_score"].median()), 3),
        "average_messages_per_ticket": round(float(df["message_count"].mean()), 2),
    }

    recommendations = _build_recommendations(
        total=total,
        summary=summary,
        recurring_issues=recurring_issues,
        automation_opportunities=automation_opportunities,
    )

    return {
        "summary": summary,
        "recurring_issues": recurring_issues,
        "automation_opportunities": automation_opportunities,
        "emerging_issues": emerging_issues,
        "recommendations": recommendations,
    }


def _fetch_all_conversations(page_size: int = 500) -> list[dict[str, Any]]:
    """Fetch all stored conversations via existing analytics pagination."""
    offset = 0
    items: list[dict[str, Any]] = []

    while True:
        page = list_conversations(limit=page_size, offset=offset)
        batch = page.get("items", [])
        total = int(page.get("total", 0))

        if not batch:
            break

        items.extend(batch)
        offset += len(batch)
        if offset >= total:
            break

    return items


def _cluster_statistics(df: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    grouped = (
        df.groupby(["cluster_id", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "conversation_count"})
        .sort_values(by=["conversation_count", "cluster_id"], ascending=[False, True])
    )

    if grouped.empty:
        return {"largest_cluster": None, "smallest_cluster": None}

    largest = grouped.iloc[0]
    smallest = grouped.sort_values(
        by=["conversation_count", "cluster_id"], ascending=[True, True]
    ).iloc[0]

    return {
        "largest_cluster": {
            "cluster_id": int(largest["cluster_id"]),
            "cluster_label": str(largest["cluster_label"]),
            "conversation_count": int(largest["conversation_count"]),
        },
        "smallest_cluster": {
            "cluster_id": int(smallest["cluster_id"]),
            "cluster_label": str(smallest["cluster_label"]),
            "conversation_count": int(smallest["conversation_count"]),
        },
    }


def _build_recurring_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    total = max(int(len(df)), 1)
    grouped = (
        df.groupby(["cluster_id", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "conversation_count"})
        .sort_values(by=["conversation_count", "cluster_id"], ascending=[False, True])
        .head(10)
    )

    issues: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        cluster_id = int(row["cluster_id"])
        cluster_df = df[df["cluster_id"] == cluster_id].copy()
        cluster_df = cluster_df.sort_values(
            by=["similarity_score", "message_count", "ticket_id"],
            ascending=[False, False, True],
        )
        representative_ticket = (
            str(cluster_df.iloc[0]["ticket_id"]) if not cluster_df.empty else ""
        )

        count = int(row["conversation_count"])
        issues.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": str(row["cluster_label"]),
                "conversation_count": count,
                "percentage_of_total": round((count / total) * 100.0, 2),
                "representative_ticket": representative_ticket,
            }
        )

    return issues


def _build_automation_opportunities(
    df: pd.DataFrame, cfg: InsightsConfig
) -> list[dict[str, Any]]:
    total = max(int(len(df)), 1)
    opportunities: list[dict[str, Any]] = []

    grouped = df.groupby(["cluster_id", "cluster_label"], as_index=False)
    for (cluster_id, cluster_label), cluster_df in grouped:
        cluster_total = int(len(cluster_df))
        duplicate_rate = float((cluster_df["status"] == "duplicate").sum()) / max(
            cluster_total, 1
        )

        reasons: list[str] = []
        if duplicate_rate >= cfg.duplicate_rate_threshold:
            reasons.append("high_duplicate_rate")
        if cluster_total >= cfg.large_cluster_min_size:
            reasons.append("large_cluster_volume")

        if not reasons:
            continue

        # Deterministic estimate: base on duplicate rate and cluster share of total conversations.
        estimated_rate = max(duplicate_rate, min(cluster_total / total, 0.7))
        estimated_automatable = int(round(cluster_total * estimated_rate))

        opportunities.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": str(cluster_label),
                "conversation_count": cluster_total,
                "duplicate_rate": round(duplicate_rate * 100.0, 2),
                "trigger_reasons": reasons,
                "estimated_automatable_conversations": estimated_automatable,
                "estimated_automation_opportunity": round(estimated_rate * 100.0, 2),
            }
        )

    opportunities.sort(
        key=lambda item: (
            -item["estimated_automatable_conversations"],
            item["cluster_id"],
        )
    )
    return opportunities


def _build_emerging_issues(
    df: pd.DataFrame, cfg: InsightsConfig
) -> list[dict[str, Any]]:
    emerging: list[dict[str, Any]] = []

    grouped = df.groupby(["cluster_id", "cluster_label"], as_index=False)
    for (cluster_id, cluster_label), cluster_df in grouped:
        cluster_total = int(len(cluster_df))
        if cluster_total == 0:
            continue

        unique_rate = float((cluster_df["status"] == "unique").sum()) / cluster_total
        if unique_rate >= cfg.mostly_unique_threshold:
            emerging.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_label": str(cluster_label),
                    "reason": "mostly_unique_conversations",
                    "conversation_count": cluster_total,
                    "unique_rate": round(unique_rate * 100.0, 2),
                }
            )

        growth_stats = _cluster_growth_stats(cluster_df, cfg.rapid_growth_window_days)
        if growth_stats is None:
            continue

        recent = growth_stats["recent_count"]
        previous = growth_stats["previous_count"]
        growth_ratio = growth_stats["growth_ratio"]

        if (
            recent >= cfg.rapid_growth_min_recent
            and growth_ratio >= cfg.rapid_growth_multiplier
            and recent > previous
        ):
            emerging.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_label": str(cluster_label),
                    "reason": "rapid_cluster_growth",
                    "conversation_count": cluster_total,
                    "recent_count": int(recent),
                    "previous_count": int(previous),
                    "growth_ratio": round(growth_ratio, 2),
                }
            )

    emerging.sort(
        key=lambda item: (
            -item.get("conversation_count", 0),
            item["cluster_id"],
            item["reason"],
        )
    )
    return emerging


def _cluster_growth_stats(
    cluster_df: pd.DataFrame, window_days: int
) -> dict[str, float] | None:
    dates = pd.to_datetime(cluster_df.get("last_sent_at"), errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return None

    end = valid_dates.max()
    recent_start = end - timedelta(days=window_days)
    previous_start = recent_start - timedelta(days=window_days)

    recent_count = int(((valid_dates > recent_start) & (valid_dates <= end)).sum())
    previous_count = int(
        ((valid_dates > previous_start) & (valid_dates <= recent_start)).sum()
    )

    growth_ratio = float(recent_count / max(previous_count, 1))
    return {
        "recent_count": recent_count,
        "previous_count": previous_count,
        "growth_ratio": growth_ratio,
    }


def _build_recommendations(
    total: int,
    summary: dict[str, Any],
    recurring_issues: list[dict[str, Any]],
    automation_opportunities: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []

    if recurring_issues:
        top = recurring_issues[0]
        recommendations.append(
            f"{top['percentage_of_total']:.0f}% of conversations are {top['cluster_label']}."
        )

    redundancy = float(summary.get("redundancy_percentage", 0.0))
    recommendations.append(
        f"{redundancy:.0f}% of conversations are duplicates or similar and may be reduced through better flows."
    )

    if automation_opportunities:
        estimated_total = int(
            sum(
                x.get("estimated_automatable_conversations", 0)
                for x in automation_opportunities
            )
        )
        estimated_rate = (estimated_total / max(total, 1)) * 100.0
        recommendations.append(
            f"{estimated_rate:.0f}% of tickets are strong automation candidates based on recurring patterns."
        )

    if recurring_issues:
        smallest = recurring_issues[-1]
        recommendations.append(
            f"Only {smallest['conversation_count']} conversations relate to {smallest['cluster_label']}."
        )

    return recommendations[:6]


__all__ = ["InsightsConfig", "generate_business_insights"]
