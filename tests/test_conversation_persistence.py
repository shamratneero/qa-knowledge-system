"""Tests for conversation persistence into SQLite."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, ConversationThread
from app.services.database import save_conversations_to_db


def test_save_conversations_to_db_replaces_existing_rows():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    session = Session()
    try:
        df = pd.DataFrame(
            {
                "ticket_id": ["105", "106"],
                "subject": ["Password reset", "Refund"],
                "message_count": [3, 1],
                "first_sent_at": [pd.NaT, pd.NaT],
                "last_sent_at": [pd.NaT, pd.NaT],
                "conversation_text": ["Guest: Hi", "Guest: Need refund"],
            }
        )

        stored = save_conversations_to_db(df, source_file="sample.xlsx", db=session)
        assert stored == 2

        rows = session.query(ConversationThread).all()
        assert len(rows) == 2
        assert rows[0].source_file == "sample.xlsx"
        assert rows[0].ticket_id in {"105", "106"}

        df2 = pd.DataFrame(
            {
                "ticket_id": ["200"],
                "subject": ["Login"],
                "message_count": [1],
                "first_sent_at": [pd.NaT],
                "last_sent_at": [pd.NaT],
                "conversation_text": ["Guest: Help"],
            }
        )
        stored_again = save_conversations_to_db(
            df2, source_file="sample.xlsx", db=session
        )
        assert stored_again == 1

        rows = session.query(ConversationThread).all()
        assert len(rows) == 1
        assert rows[0].ticket_id == "200"
    finally:
        session.close()
