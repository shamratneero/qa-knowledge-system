"""Deterministic business insights over persisted conversation analytics data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from app.services.conversation_analytics import (
    get_analytics_overview,
    list_conversations,
)


_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_RANK_PRIORITY = {v: k for k, v in _PRIORITY_RANK.items()}


@dataclass(slots=True)
class InsightsConfig:
    """Configurable thresholds for deterministic insight rules."""

    duplicate_rate_threshold: float = 0.35
    large_cluster_min_size: int = 8
    # No longer used by _build_emerging_issues (that rule produced noise --
    # nearly every singleton unique conversation qualified). Kept only so
    # existing /analytics/insights callers passing this query param don't break.
    mostly_unique_threshold: float = 0.70
    rapid_growth_multiplier: float = 1.8
    rapid_growth_window_days: int = 7
    rapid_growth_min_recent: int = 3
    emerging_min_cluster_size: int = 3
    emerging_priority_threshold: str = "high"


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
    for optional_col, default in (("intent", ""), ("category", ""), ("priority", "low")):
        if optional_col in df.columns:
            df[optional_col] = df[optional_col].fillna(default).astype(str)
            df.loc[df[optional_col].str.strip() == "", optional_col] = default
        else:
            df[optional_col] = default

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


def _with_group_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a group_key column: customer intent, falling back to cluster label
    for legacy rows with no intent recorded. Recurring issues, automation
    opportunities, and emerging issues all group by this key instead of raw
    cluster wording or keyword frequency."""
    work = df.copy()
    work["group_key"] = work["intent"].str.strip()
    work.loc[work["group_key"] == "", "group_key"] = work["cluster_label"]
    return work


def _build_recurring_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group recurring issues by customer intent (falling back to cluster label for
    legacy rows with no intent recorded), instead of raw cluster wording."""
    total = max(int(len(df)), 1)
    work = _with_group_key(df)

    grouped = (
        work.groupby("group_key", as_index=False)
        .size()
        .rename(columns={"size": "conversation_count"})
        .sort_values(by=["conversation_count", "group_key"], ascending=[False, True])
        .head(10)
    )

    issues: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        group_key = row["group_key"]
        group_df = work[work["group_key"] == group_key].copy()

        representative_cluster = (
            group_df["cluster_id"].value_counts().idxmax()
            if not group_df.empty
            else 0
        )
        cluster_df = group_df[group_df["cluster_id"] == representative_cluster]
        representative_label = (
            str(cluster_df.iloc[0]["cluster_label"]) if not cluster_df.empty else ""
        )

        ranked = group_df.sort_values(
            by=["similarity_score", "message_count", "ticket_id"],
            ascending=[False, False, True],
        )
        representative_ticket = (
            str(ranked.iloc[0]["ticket_id"]) if not ranked.empty else ""
        )

        count = int(row["conversation_count"])
        intent_value = group_df.iloc[0]["intent"].strip() if not group_df.empty else ""
        issues.append(
            {
                "cluster_id": int(representative_cluster),
                "cluster_label": representative_label,
                "conversation_count": count,
                "percentage_of_total": round((count / total) * 100.0, 2),
                "representative_ticket": representative_ticket,
                "intent": intent_value or None,
            }
        )

    return issues


def _representative_cluster(group_df: pd.DataFrame) -> tuple[int, str]:
    """Pick the most common (cluster_id, cluster_label) within a group_key group."""
    if group_df.empty:
        return 0, ""
    representative_cluster = int(group_df["cluster_id"].value_counts().idxmax())
    cluster_df = group_df[group_df["cluster_id"] == representative_cluster]
    representative_label = (
        str(cluster_df.iloc[0]["cluster_label"]) if not cluster_df.empty else ""
    )
    return representative_cluster, representative_label


def _representative_ticket(group_df: pd.DataFrame) -> str:
    if group_df.empty:
        return ""
    ranked = group_df.sort_values(
        by=["similarity_score", "message_count", "ticket_id"],
        ascending=[False, False, True],
    )
    return str(ranked.iloc[0]["ticket_id"]) if not ranked.empty else ""


def _build_automation_opportunities(
    df: pd.DataFrame, cfg: InsightsConfig
) -> list[dict[str, Any]]:
    total = max(int(len(df)), 1)
    work = _with_group_key(df)
    opportunities: list[dict[str, Any]] = []

    for group_key, group_df in work.groupby("group_key"):
        group_total = int(len(group_df))
        duplicate_rate = float((group_df["status"] == "duplicate").sum()) / max(
            group_total, 1
        )

        reasons: list[str] = []
        if duplicate_rate >= cfg.duplicate_rate_threshold:
            reasons.append("high_duplicate_rate")
        if group_total >= cfg.large_cluster_min_size:
            reasons.append("large_cluster_volume")

        if not reasons:
            continue

        # Deterministic estimate: base on duplicate rate and group share of total conversations.
        estimated_rate = max(duplicate_rate, min(group_total / total, 0.7))
        estimated_automatable = int(round(group_total * estimated_rate))

        representative_cluster, representative_label = _representative_cluster(group_df)
        intent_value = group_df.iloc[0]["intent"].strip() if not group_df.empty else ""

        opportunities.append(
            {
                "cluster_id": representative_cluster,
                "cluster_label": representative_label,
                "conversation_count": group_total,
                "duplicate_rate": round(duplicate_rate * 100.0, 2),
                "trigger_reasons": reasons,
                "estimated_automatable_conversations": estimated_automatable,
                "estimated_automation_opportunity": round(estimated_rate * 100.0, 2),
                "intent": intent_value or None,
            }
        )

    opportunities.sort(
        key=lambda item: (
            -item["estimated_automatable_conversations"],
            item["cluster_id"],
        )
    )
    return opportunities


def _average_priority(group_df: pd.DataFrame) -> tuple[str, float]:
    ranks = [
        _PRIORITY_RANK.get(str(p).strip().lower(), 0) for p in group_df["priority"].tolist()
    ]
    avg_rank = float(sum(ranks)) / max(len(ranks), 1)
    label = _RANK_PRIORITY.get(round(avg_rank), _RANK_PRIORITY[min(3, max(0, round(avg_rank)))])
    return label, avg_rank


def _build_emerging_issues(
    df: pd.DataFrame, cfg: InsightsConfig
) -> list[dict[str, Any]]:
    """Group by customer intent and flag a group as an emerging issue only when
    it meets one of three explicit rules -- minimum size, rapid growth, or high
    average priority -- instead of flagging nearly every singleton unique
    conversation (the old "mostly unique" rule, which was pure noise)."""
    work = _with_group_key(df)
    priority_threshold_rank = _PRIORITY_RANK.get(
        cfg.emerging_priority_threshold.strip().lower(), _PRIORITY_RANK["high"]
    )
    emerging: list[dict[str, Any]] = []

    for group_key, group_df in work.groupby("group_key"):
        group_total = int(len(group_df))
        if group_total == 0:
            continue

        growth_stats = _cluster_growth_stats(group_df, cfg.rapid_growth_window_days)
        has_rapid_growth = bool(
            growth_stats
            and growth_stats["recent_count"] >= cfg.rapid_growth_min_recent
            and growth_stats["growth_ratio"] >= cfg.rapid_growth_multiplier
            and growth_stats["recent_count"] > growth_stats["previous_count"]
        )

        average_priority_label, avg_priority_rank = _average_priority(group_df)
        has_high_priority = avg_priority_rank >= priority_threshold_rank
        has_min_size = group_total >= cfg.emerging_min_cluster_size

        trigger_reasons: list[str] = []
        if has_min_size:
            trigger_reasons.append("minimum_cluster_size")
        if has_rapid_growth:
            trigger_reasons.append("rapid_growth")
        if has_high_priority:
            trigger_reasons.append("high_average_priority")

        if not trigger_reasons:
            continue

        representative_cluster, representative_label = _representative_cluster(group_df)
        representative_ticket = _representative_ticket(group_df)
        intent_value = group_df.iloc[0]["intent"].strip() or None
        category_counts = Counter(
            str(c).strip() for c in group_df["category"].tolist() if str(c).strip()
        )
        category_value = category_counts.most_common(1)[0][0] if category_counts else None

        representative_row = group_df[group_df["ticket_id"] == representative_ticket]
        summary_value = (
            str(representative_row.iloc[0].get("summary", "") or "").strip()
            if not representative_row.empty and "summary" in group_df.columns
            else ""
        )

        growth_percentage = (
            round((growth_stats["growth_ratio"] - 1.0) * 100.0, 2) if growth_stats else None
        )

        emerging.append(
            {
                "cluster_id": representative_cluster,
                "cluster_label": representative_label,
                "intent": intent_value,
                "category": category_value,
                "conversation_count": group_total,
                "growth_percentage": growth_percentage,
                "average_priority": average_priority_label,
                "summary": summary_value,
                "representative_ticket": representative_ticket,
                "trigger_reasons": trigger_reasons,
            }
        )

    emerging.sort(
        key=lambda item: (
            -item.get("conversation_count", 0),
            item["cluster_id"],
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
