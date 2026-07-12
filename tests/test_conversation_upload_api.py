"""API tests for uploading conversation Excel files."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def _build_excel_bytes() -> bytes:
    tickets_df = pd.DataFrame(
        {
            "Ticket ID": ["105", "106"],
            "Status": ["Open", "Closed"],
        }
    )
    conversations_df = pd.DataFrame(
        {
            "Ticket ID": ["105", "105", "105", "106"],
            "Subject": ["Password reset", "Password reset", "Password reset", "Refund"],
            "Message Number": [1, 2, 3, 1],
            "Direction": ["Guest", "Agent", "Guest", "Guest"],
            "Sender": ["Guest", "Support Agent", "Guest", "Guest"],
            "Message": ["Hi", "Hello", "I forgot my password", "Need a refund"],
            "Sent At": [
                "2026-07-10 10:00:00",
                "2026-07-10 10:01:00",
                "2026-07-10 10:02:00",
                "2026-07-11 08:00:00",
            ],
        }
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        tickets_df.to_excel(writer, sheet_name="Tickets", index=False)
        conversations_df.to_excel(writer, sheet_name="Conversations", index=False)

    return buffer.getvalue()


def test_upload_conversations_preview_success(client):
    response = client.post(
        "/admin/conversations/upload?preview_count=2",
        files={
            "file": (
                "conversations.xlsx",
                _build_excel_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["preview_count"] == 2
    assert data["report"]["file_name"] == "conversations.xlsx"
    assert data["report"]["total_source_rows"] == 4
    assert data["report"]["valid_conversations"] == 2
    assert data["report"]["stored_conversations"] == 2
    assert data["report"]["unique_ticket_ids"] == 2
    assert "duplicate_conversations" in data["report"]
    assert "similar_conversations" in data["report"]
    assert "unique_conversations" in data["report"]
    assert "total_clusters" in data["report"]
    assert data["report"]["total_clusters"] >= 1
    assert (
        data["report"]["duplicate_conversations"]
        + data["report"]["similar_conversations"]
        + data["report"]["unique_conversations"]
        == data["report"]["valid_conversations"]
    )
    assert len(data["preview_rows"]) == 2
    assert data["preview_rows"][0]["ticket_id"] == "105"
    assert "status" in data["preview_rows"][0]
    assert "cluster_id" in data["preview_rows"][0]
    assert "cluster_label" in data["preview_rows"][0]
    assert "Guest: Hi" in data["preview_rows"][0]["conversation_text"]


def test_upload_conversations_rejects_non_excel(client):
    response = client.post(
        "/admin/conversations/upload",
        files={"file": ("notes.txt", b"not excel", "text/plain")},
    )

    assert response.status_code == 400
    assert "Excel file" in response.json()["detail"]


def test_upload_conversations_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_upload_size_mb", 0)

    response = client.post(
        "/admin/conversations/upload",
        files={
            "file": (
                "conversations.xlsx",
                _build_excel_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
