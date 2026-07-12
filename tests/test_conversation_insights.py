"""Tests for deterministic conversation business insights."""

from __future__ import annotations

from app.services.conversation_insights import (
    InsightsConfig,
    generate_business_insights,
)

SAMPLE_ROWS = [
    {
        "id": 1,
        "ticket_id": "T-100",
        "subject": "Password reset",
        "status": "duplicate",
        "similarity_score": 0.98,
        "nearest_ticket_id": "T-101",
        "cluster_id": 10,
        "cluster_label": "Password Reset",
        "message_count": 3,
        "first_sent_at": "2026-07-01T10:00:00",
        "last_sent_at": "2026-07-12T10:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
    },
    {
        "id": 2,
        "ticket_id": "T-101",
        "subject": "Password reset",
        "status": "duplicate",
        "similarity_score": 0.97,
        "nearest_ticket_id": "T-100",
        "cluster_id": 10,
        "cluster_label": "Password Reset",
        "message_count": 2,
        "first_sent_at": "2026-07-02T10:00:00",
        "last_sent_at": "2026-07-12T11:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
    },
    {
        "id": 3,
        "ticket_id": "T-102",
        "subject": "Password reset",
        "status": "similar",
        "similarity_score": 0.88,
        "nearest_ticket_id": "T-100",
        "cluster_id": 10,
        "cluster_label": "Password Reset",
        "message_count": 4,
        "first_sent_at": "2026-07-03T10:00:00",
        "last_sent_at": "2026-07-04T10:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
    },
    {
        "id": 4,
        "ticket_id": "T-200",
        "subject": "Parking help",
        "status": "unique",
        "similarity_score": 0.21,
        "nearest_ticket_id": None,
        "cluster_id": 20,
        "cluster_label": "Parking Questions",
        "message_count": 1,
        "first_sent_at": "2026-07-05T08:00:00",
        "last_sent_at": "2026-07-05T08:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
    },
    {
        "id": 5,
        "ticket_id": "T-201",
        "subject": "Parking help",
        "status": "unique",
        "similarity_score": 0.19,
        "nearest_ticket_id": None,
        "cluster_id": 20,
        "cluster_label": "Parking Questions",
        "message_count": 2,
        "first_sent_at": "2026-07-11T08:00:00",
        "last_sent_at": "2026-07-12T08:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
    },
    {
        "id": 6,
        "ticket_id": "T-300",
        "subject": "Refund",
        "status": "unique",
        "similarity_score": 0.10,
        "nearest_ticket_id": None,
        "cluster_id": 30,
        "cluster_label": "Refund",
        "message_count": 2,
        "first_sent_at": "2026-07-10T14:00:00",
        "last_sent_at": "2026-07-10T14:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
    },
]


def test_generate_business_insights_metrics_and_sections(monkeypatch):
    def fake_overview():
        return {
            "total_tickets": 6,
            "total_conversations": 6,
            "duplicate_conversations": 2,
            "similar_conversations": 1,
            "unique_conversations": 3,
            "total_clusters": 3,
        }

    def fake_list(limit=100, offset=0, **kwargs):
        items = SAMPLE_ROWS[offset : offset + limit]
        return {"total": len(SAMPLE_ROWS), "items": items}

    monkeypatch.setattr(
        "app.services.conversation_insights.get_analytics_overview", fake_overview
    )
    monkeypatch.setattr(
        "app.services.conversation_insights.list_conversations", fake_list
    )

    data = generate_business_insights(
        InsightsConfig(
            duplicate_rate_threshold=0.5,
            large_cluster_min_size=4,
            mostly_unique_threshold=0.8,
            rapid_growth_multiplier=1.5,
            rapid_growth_window_days=7,
            rapid_growth_min_recent=2,
        )
    )

    assert data["summary"]["total_conversations"] == 6
    assert data["summary"]["redundancy_percentage"] == 50.0
    assert data["summary"]["largest_cluster"]["cluster_label"] == "Password Reset"
    assert data["summary"]["smallest_cluster"]["cluster_label"] == "Refund"

    assert len(data["recurring_issues"]) == 3
    assert data["recurring_issues"][0]["cluster_label"] == "Password Reset"

    assert data["automation_opportunities"]
    assert data["automation_opportunities"][0]["cluster_label"] == "Password Reset"

    # Password Reset group has 3 conversations (>= default emerging_min_cluster_size),
    # so it qualifies; the 1- and 2-conversation groups (Refund, Parking Questions)
    # do not -- singleton/small unique conversations no longer flood emerging issues.
    assert len(data["emerging_issues"]) == 1
    emerging = data["emerging_issues"][0]
    # SAMPLE_ROWS has no "intent" column, so the group_key falls back to
    # cluster_label -- intent stays None, cluster_label carries the group name.
    assert emerging["cluster_label"] == "Password Reset"
    assert "minimum_cluster_size" in emerging["trigger_reasons"]
    assert emerging["conversation_count"] == 3

    assert data["recommendations"]


def test_generate_business_insights_empty(monkeypatch):
    monkeypatch.setattr(
        "app.services.conversation_insights.get_analytics_overview",
        lambda: {
            "total_tickets": 0,
            "total_conversations": 0,
            "duplicate_conversations": 0,
            "similar_conversations": 0,
            "unique_conversations": 0,
            "total_clusters": 0,
        },
    )
    monkeypatch.setattr(
        "app.services.conversation_insights.list_conversations",
        lambda limit=100, offset=0, **kwargs: {"total": 0, "items": []},
    )

    data = generate_business_insights()
    assert data["summary"]["total_conversations"] == 0
    assert data["recurring_issues"] == []
    assert data["automation_opportunities"] == []
    assert data["emerging_issues"] == []
    assert data["recommendations"]


INTENT_ROWS = [
    {**row, "intent": "Password Reset" if row["cluster_id"] == 10 else ""}
    for row in SAMPLE_ROWS
    if row["cluster_id"] in (10, 20)
] + [
    {
        "id": 7,
        "ticket_id": "T-400",
        "subject": "Login help",
        "status": "similar",
        "similarity_score": 0.85,
        "nearest_ticket_id": "T-100",
        "cluster_id": 40,
        "cluster_label": "Login Help",
        "message_count": 2,
        "first_sent_at": "2026-07-06T10:00:00",
        "last_sent_at": "2026-07-06T10:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
        "intent": "Password Reset",
    },
]


def test_recurring_issues_group_by_intent_over_cluster_label(monkeypatch):
    def fake_overview():
        return {
            "total_tickets": len(INTENT_ROWS),
            "total_conversations": len(INTENT_ROWS),
            "duplicate_conversations": 0,
            "similar_conversations": 0,
            "unique_conversations": 0,
            "total_clusters": 3,
        }

    def fake_list(limit=100, offset=0, **kwargs):
        items = INTENT_ROWS[offset : offset + limit]
        return {"total": len(INTENT_ROWS), "items": items}

    monkeypatch.setattr(
        "app.services.conversation_insights.get_analytics_overview", fake_overview
    )
    monkeypatch.setattr(
        "app.services.conversation_insights.list_conversations", fake_list
    )

    data = generate_business_insights()

    # Cluster 10 ("Password Reset" label, 3 rows) and cluster 40 ("Login Help"
    # label, 1 row) both carry intent "Password Reset" -- they must merge into
    # a single recurring-issue group of 4, instead of two separate groups of
    # 3 and 1 keyed off raw cluster label.
    password_group = next(
        item for item in data["recurring_issues"] if item["intent"] == "Password Reset"
    )
    assert password_group["conversation_count"] == 4


def _row(**overrides):
    base = {
        "id": 1,
        "ticket_id": "T-1",
        "subject": "Issue",
        "status": "unique",
        "similarity_score": 0.2,
        "nearest_ticket_id": None,
        "cluster_id": 1,
        "cluster_label": "Issue",
        "message_count": 1,
        "first_sent_at": "2026-07-01T10:00:00",
        "last_sent_at": "2026-07-01T10:00:00",
        "representative_conversation": "...",
        "conversation_text": "...",
        "intent": "General Inquiry",
        "category": "General",
        "priority": "low",
        "summary": "Customer reported a general inquiry.",
    }
    base.update(overrides)
    return base


def _patched_insights(monkeypatch, rows):
    def fake_overview():
        return {
            "total_tickets": len(rows),
            "total_conversations": len(rows),
            "duplicate_conversations": 0,
            "similar_conversations": 0,
            "unique_conversations": len(rows),
            "total_clusters": len({r["cluster_id"] for r in rows}),
        }

    def fake_list(limit=100, offset=0, **kwargs):
        return {"total": len(rows), "items": rows[offset : offset + limit]}

    monkeypatch.setattr(
        "app.services.conversation_insights.get_analytics_overview", fake_overview
    )
    monkeypatch.setattr(
        "app.services.conversation_insights.list_conversations", fake_list
    )


def test_emerging_issue_min_size_trigger(monkeypatch):
    rows = [
        _row(id=i, ticket_id=f"T-{i}", cluster_id=1, intent="Booking Inquiry")
        for i in range(1, 4)
    ]
    _patched_insights(monkeypatch, rows)

    data = generate_business_insights()
    emerging = {item["intent"]: item for item in data["emerging_issues"]}
    assert "Booking Inquiry" in emerging
    assert "minimum_cluster_size" in emerging["Booking Inquiry"]["trigger_reasons"]


def test_emerging_issue_growth_trigger(monkeypatch):
    rows = [
        _row(
            id=1,
            ticket_id="T-1",
            cluster_id=2,
            intent="Refund Request",
            last_sent_at="2026-06-01T10:00:00",
        ),
        _row(
            id=2,
            ticket_id="T-2",
            cluster_id=2,
            intent="Refund Request",
            last_sent_at="2026-07-10T10:00:00",
        ),
        _row(
            id=3,
            ticket_id="T-3",
            cluster_id=2,
            intent="Refund Request",
            last_sent_at="2026-07-11T10:00:00",
        ),
    ]
    _patched_insights(monkeypatch, rows)

    data = generate_business_insights(
        InsightsConfig(
            rapid_growth_multiplier=1.5,
            rapid_growth_window_days=7,
            rapid_growth_min_recent=2,
        )
    )
    emerging = {item["intent"]: item for item in data["emerging_issues"]}
    assert "Refund Request" in emerging
    assert "rapid_growth" in emerging["Refund Request"]["trigger_reasons"]
    assert emerging["Refund Request"]["growth_percentage"] is not None


def test_emerging_issue_priority_trigger(monkeypatch):
    rows = [
        _row(
            id=1,
            ticket_id="T-1",
            cluster_id=3,
            intent="Technical Issue",
            priority="high",
        ),
        _row(
            id=2,
            ticket_id="T-2",
            cluster_id=3,
            intent="Technical Issue",
            priority="high",
        ),
    ]
    _patched_insights(monkeypatch, rows)

    data = generate_business_insights()
    emerging = {item["intent"]: item for item in data["emerging_issues"]}
    assert "Technical Issue" in emerging
    assert "high_average_priority" in emerging["Technical Issue"]["trigger_reasons"]
    assert emerging["Technical Issue"]["average_priority"] == "high"


def test_emerging_issue_card_fields_present(monkeypatch):
    rows = [
        _row(
            id=i,
            ticket_id=f"T-{i}",
            cluster_id=4,
            intent="Cancellation Request",
            category="Billing",
            priority="high",
        )
        for i in range(1, 4)
    ]
    _patched_insights(monkeypatch, rows)

    data = generate_business_insights()
    item = next(
        x for x in data["emerging_issues"] if x["intent"] == "Cancellation Request"
    )
    assert item["category"] == "Billing"
    assert item["conversation_count"] == 3
    assert item["average_priority"] in {"low", "medium", "high", "critical"}
    assert item["summary"]
    assert item["representative_ticket"]


def test_emerging_issues_do_not_flood_with_singleton_low_priority_conversations(
    monkeypatch,
):
    """Direct regression test for the reported noise bug: a lone, low-priority,
    non-growing conversation must never show up as an emerging issue."""
    rows = [
        _row(
            id=1,
            ticket_id="T-1",
            cluster_id=1,
            intent="General Inquiry",
            priority="low",
        ),
        _row(
            id=2,
            ticket_id="T-2",
            cluster_id=2,
            intent="Parking Inquiry",
            priority="low",
        ),
        _row(
            id=3,
            ticket_id="T-3",
            cluster_id=3,
            intent="Booking Inquiry",
            priority="medium",
        ),
    ]
    _patched_insights(monkeypatch, rows)

    data = generate_business_insights()
    assert data["emerging_issues"] == []


def test_automation_opportunities_group_by_intent(monkeypatch):
    rows = [
        _row(
            id=1,
            ticket_id="T-1",
            cluster_id=1,
            cluster_label="Password A",
            intent="Password Reset",
            status="duplicate",
        ),
        _row(
            id=2,
            ticket_id="T-2",
            cluster_id=2,
            cluster_label="Password B",
            intent="Password Reset",
            status="duplicate",
        ),
        _row(
            id=3,
            ticket_id="T-3",
            cluster_id=3,
            cluster_label="Password C",
            intent="Password Reset",
            status="duplicate",
        ),
    ]
    _patched_insights(monkeypatch, rows)

    data = generate_business_insights(
        InsightsConfig(duplicate_rate_threshold=0.5, large_cluster_min_size=100)
    )
    opportunities = {item["intent"]: item for item in data["automation_opportunities"]}
    assert "Password Reset" in opportunities
    assert opportunities["Password Reset"]["conversation_count"] == 3
