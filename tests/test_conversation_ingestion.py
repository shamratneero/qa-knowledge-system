"""Tests for conversation Excel ingestion and reconstruction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.conversation_ingestion import (
    REQUIRED_CONVERSATION_COLUMNS,
    load_and_reconstruct_conversations,
)


def _write_valid_excel(path: Path) -> None:
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
            "Direction": ["Guest", "Agent", "Guest", "Guest"],
            "Sender": ["Guest", "Support Agent", "Guest", "Guest"],
            "Message": ["Hi", "Hello", "I forgot my password", "Need a refund"],
        }
    )
    with pd.ExcelWriter(path) as writer:
        tickets_df.to_excel(writer, sheet_name="Tickets", index=False)
        conversations_df.to_excel(writer, sheet_name="Conversations", index=False)


def test_reconstruct_groups_by_ticket_id(tmp_path: Path):
    file_path = tmp_path / "sample.xlsx"
    _write_valid_excel(file_path)

    out = load_and_reconstruct_conversations(file_path)

    assert len(out) == 2
    row_105 = out[out["ticket_id"] == "105"].iloc[0]
    assert row_105["subject"] == "Password reset"
    assert row_105["message_count"] == 3
    assert row_105["conversation_text"] == (
        "Guest: Hi\n\n" "Support Agent: Hello\n\n" "Guest: I forgot my password"
    )


def test_missing_required_sheet(tmp_path: Path):
    file_path = tmp_path / "missing_sheet.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({"Ticket ID": [1]}).to_excel(
            writer, sheet_name="Conversations", index=False
        )

    with pytest.raises(ValueError, match="Missing required sheet"):
        load_and_reconstruct_conversations(file_path)


def test_missing_required_conversation_columns(tmp_path: Path):
    file_path = tmp_path / "missing_columns.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({"Ticket ID": [1]}).to_excel(
            writer, sheet_name="Tickets", index=False
        )
        pd.DataFrame(
            {
                "Ticket ID": [1],
                "Subject": ["S"],
                "Direction": ["Guest"],
                "Sender": ["Guest"],
            }
        ).to_excel(
            writer,
            sheet_name="Conversations",
            index=False,
        )

    with pytest.raises(ValueError, match="Missing required conversation column"):
        load_and_reconstruct_conversations(file_path)


def test_optional_message_number_and_sent_at_can_be_missing(tmp_path: Path):
    file_path = tmp_path / "optional_columns.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({"Ticket ID": [1]}).to_excel(
            writer, sheet_name="Tickets", index=False
        )
        pd.DataFrame(
            {
                "Ticket ID": ["105", "105"],
                "Subject": ["Password reset", "Password reset"],
                "Direction": ["Guest", "Agent"],
                "Sender": ["Guest", "Support Agent"],
                "Message": ["Hi", "Hello"],
            }
        ).to_excel(writer, sheet_name="Conversations", index=False)

    out = load_and_reconstruct_conversations(file_path)
    assert len(out) == 1
    assert out.iloc[0]["message_count"] == 2
    assert out.iloc[0]["conversation_text"] == "Guest: Hi\n\nSupport Agent: Hello"


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_and_reconstruct_conversations(tmp_path / "does_not_exist.xlsx")


def test_required_columns_constant_is_complete():
    assert REQUIRED_CONVERSATION_COLUMNS == [
        "Ticket ID",
        "Subject",
        "Direction",
        "Sender",
        "Message",
    ]
