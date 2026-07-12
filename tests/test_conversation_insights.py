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

    reasons = {item["reason"] for item in data["emerging_issues"]}
    assert "mostly_unique_conversations" in reasons

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
