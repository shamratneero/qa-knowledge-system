"""UI smoke tests for dashboard sections and chart containers."""

from __future__ import annotations


def test_ui_contains_saas_dashboard_sections(client):
    response = client.get("/ui")
    assert response.status_code == 200
    html = response.text

    # Landing screen (guided flow entry point).
    assert "Upload Conversation Excel" in html
    # Report screen sections (business insights, now presented as cards).
    assert "Top Customer Issues" in html
    assert "Automation Opportunities" in html
    assert "Emerging Issues" in html
    assert "Explore Conversations" in html
    assert "Knowledge Base & Continuous Learning" in html
    assert "Ask AI Analyst" in html


def test_ui_contains_chart_canvas_elements(client):
    response = client.get("/ui")
    assert response.status_code == 200
    html = response.text

    assert "statusPieChart" in html
    assert "clusterBarChart" in html
    assert "dailyLineChart" in html
    assert "recurringChart" in html
    assert "knowledgeGrowthChart" in html
