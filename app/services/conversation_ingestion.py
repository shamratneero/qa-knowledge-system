"""Conversation ingestion and reconstruction from support Excel exports."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.logging import logger

REQUIRED_SHEETS = ["Tickets", "Conversations"]

REQUIRED_CONVERSATION_COLUMNS = [
    "Ticket ID",
    "Subject",
    "Direction",
    "Sender",
    "Message",
]

OPTIONAL_CONVERSATION_COLUMNS = ["Message Number", "Sent At"]


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _speaker_label(row: pd.Series) -> str:
    sender = _normalize_text(row.get("Sender"))
    direction = _normalize_text(row.get("Direction"))
    if sender:
        return sender
    if direction:
        return direction
    return "Unknown"


def _validate_excel_structure(excel_file: pd.ExcelFile) -> None:
    available = set(excel_file.sheet_names)
    missing = [name for name in REQUIRED_SHEETS if name not in available]
    if missing:
        raise ValueError(f"Missing required sheet(s): {missing}")


def load_conversations_sheet(file_path: str | Path) -> pd.DataFrame:
    """Load and validate the Conversations sheet from an Excel file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    excel = pd.ExcelFile(path)
    _validate_excel_structure(excel)

    conversations = pd.read_excel(path, sheet_name="Conversations")
    missing_cols = [
        c for c in REQUIRED_CONVERSATION_COLUMNS if c not in conversations.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required conversation column(s): {missing_cols}")

    if conversations.empty:
        raise ValueError("Conversations sheet is empty.")

    return conversations


def load_conversations_sheet_from_bytes(
    file_bytes: bytes, file_name: str = "uploaded.xlsx"
) -> pd.DataFrame:
    """Load and validate the Conversations sheet from uploaded Excel bytes."""
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    excel = pd.ExcelFile(BytesIO(file_bytes))
    _validate_excel_structure(excel)

    conversations = pd.read_excel(BytesIO(file_bytes), sheet_name="Conversations")
    missing_cols = [
        c for c in REQUIRED_CONVERSATION_COLUMNS if c not in conversations.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required conversation column(s): {missing_cols}")

    if conversations.empty:
        raise ValueError("Conversations sheet is empty.")

    logger.info(
        "loaded_conversations_from_upload file=%s rows=%d",
        file_name,
        len(conversations),
    )
    return conversations


def reconstruct_conversations(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct full conversation text by grouping messages by Ticket ID."""
    work = conversations_df.copy()
    work["_row_order"] = range(len(work))
    work["Ticket ID"] = work["Ticket ID"].apply(_normalize_text)
    work["Message"] = work["Message"].apply(_normalize_text)
    work["Subject"] = work["Subject"].apply(_normalize_text)

    work = work[(work["Ticket ID"] != "") & (work["Message"] != "")]
    if work.empty:
        raise ValueError("No valid conversation messages found after normalization.")

    if "Sent At" in work.columns:
        work["_sent_at"] = pd.to_datetime(work["Sent At"], errors="coerce", utc=False)
    else:
        work["_sent_at"] = pd.NaT

    if "Message Number" in work.columns:
        work["_msg_num"] = pd.to_numeric(work["Message Number"], errors="coerce")
    else:
        work["_msg_num"] = work["_row_order"]

    work["_msg_num"] = work["_msg_num"].fillna(work["_row_order"])

    work = work.sort_values(
        by=["Ticket ID", "_sent_at", "_msg_num", "_row_order"], kind="stable"
    )

    reconstructed: list[dict[str, Any]] = []
    for ticket_id, group in work.groupby("Ticket ID", sort=False):
        lines = []
        for _, row in group.iterrows():
            label = _speaker_label(row)
            msg = _normalize_text(row.get("Message"))
            lines.append(f"{label}: {msg}")

        subject = ""
        subject_candidates = [
            s for s in group["Subject"].tolist() if _normalize_text(s)
        ]
        if subject_candidates:
            subject = _normalize_text(subject_candidates[0])

        first_sent_at = group["_sent_at"].min()
        last_sent_at = group["_sent_at"].max()

        reconstructed.append(
            {
                "ticket_id": ticket_id,
                "subject": subject,
                "message_count": int(len(lines)),
                "first_sent_at": first_sent_at,
                "last_sent_at": last_sent_at,
                "conversation_text": "\n\n".join(lines),
            }
        )

    out = pd.DataFrame(reconstructed)
    logger.info(
        "reconstructed_conversations tickets=%d messages=%d", len(out), len(work)
    )
    return out


def summarize_conversation_upload(conversations_df: pd.DataFrame) -> dict[str, int]:
    """Summarize row-level validity for an uploaded Conversations sheet."""
    work = conversations_df.copy()
    work["Ticket ID"] = work["Ticket ID"].apply(_normalize_text)
    work["Message"] = work["Message"].apply(_normalize_text)
    valid_mask = (work["Ticket ID"] != "") & (work["Message"] != "")
    valid_rows = int(valid_mask.sum())
    invalid_rows = int(len(work) - valid_rows)
    valid_work = work[valid_mask]
    unique_ticket_ids = (
        int(valid_work["Ticket ID"].nunique()) if not valid_work.empty else 0
    )

    return {
        "total_source_rows": int(len(work)),
        "valid_message_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "unique_ticket_ids": unique_ticket_ids,
    }


def load_and_reconstruct_conversations(file_path: str | Path) -> pd.DataFrame:
    """Read Excel and return reconstructed conversations grouped by Ticket ID."""
    source_df = load_conversations_sheet(file_path)
    return reconstruct_conversations(source_df)


def load_and_reconstruct_conversations_from_bytes(
    file_bytes: bytes, file_name: str = "uploaded.xlsx"
) -> pd.DataFrame:
    """Read uploaded Excel bytes and return reconstructed conversations grouped by Ticket ID."""
    source_df = load_conversations_sheet_from_bytes(file_bytes, file_name=file_name)
    return reconstruct_conversations(source_df)


__all__ = [
    "REQUIRED_CONVERSATION_COLUMNS",
    "OPTIONAL_CONVERSATION_COLUMNS",
    "REQUIRED_SHEETS",
    "load_and_reconstruct_conversations",
    "load_and_reconstruct_conversations_from_bytes",
    "load_conversations_sheet",
    "load_conversations_sheet_from_bytes",
    "reconstruct_conversations",
    "summarize_conversation_upload",
]
