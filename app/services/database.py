"""SQLite database session management."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import settings
from ..core.logging import logger
from ..models.database import Base, ConversationThread, Knowledge

# Database URL
DB_FILE = settings.data_path / "knowledge.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.debug,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the database (create tables)."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", DB_FILE)


def get_db() -> Generator[Session, None, None]:
    """Get a database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_from_excel():
    """Migrate data from Excel knowledge base to SQLite."""
    from ..core.loader import load_knowledge_base

    logger.info("Migrating data from Excel to SQLite...")
    init_db()

    # Clear existing data
    db = SessionLocal()
    db.query(Knowledge).delete()
    db.commit()

    # Load from Excel
    df = load_knowledge_base()

    # Insert into database
    for _, row in df.iterrows():
        knowledge = Knowledge(
            id=int(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            category=str(row["category"]),
            keywords=str(row.get("keywords", "")),
            is_active=True,
        )
        db.add(knowledge)

    db.commit()
    db.close()
    logger.info("Migrated %d entries to SQLite", len(df))


def save_conversations_to_db(
    conversations_df, source_file: str, db: Session | None = None
) -> int:
    """Persist reconstructed conversations to SQLite as the current active dataset."""
    init_db()
    _ensure_conversation_schema()

    owns_session = db is None
    session = db or SessionLocal()

    try:
        session.query(ConversationThread).delete()

        rows = []
        for _, row in conversations_df.iterrows():
            rows.append(
                ConversationThread(
                    ticket_id=str(row["ticket_id"]),
                    subject=str(row.get("subject", "")),
                    conversation_text=str(row["conversation_text"]),
                    message_count=int(row.get("message_count", 0)),
                    first_sent_at=_to_datetime(row.get("first_sent_at")),
                    last_sent_at=_to_datetime(row.get("last_sent_at")),
                    source_file=source_file,
                    status=str(row.get("status", "unique")),
                    similarity_score=int(
                        round(float(row.get("similarity_score", 0.0)) * 1000)
                    ),
                    nearest_ticket_id=(
                        None
                        if row.get("nearest_ticket_id") in [None, "", "nan"]
                        else str(row.get("nearest_ticket_id"))
                    ),
                    cluster_id=int(row.get("cluster_id", 0)),
                    cluster_label=str(row.get("cluster_label", "Cluster 0")),
                    is_active=True,
                )
            )

        session.add_all(rows)
        session.commit()
        logger.info(
            "Saved %d reconstructed conversations from %s", len(rows), source_file
        )
        return len(rows)
    except Exception:
        session.rollback()
        logger.exception(
            "Failed to save conversations to SQLite source_file=%s", source_file
        )
        raise
    finally:
        if owns_session:
            session.close()


def _ensure_conversation_schema() -> None:
    """Apply minimal schema upgrades for conversation_threads on existing SQLite DBs."""
    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(conversation_threads)"))
        }

        if "status" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE conversation_threads ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'unique'"
                )
            )
        if "similarity_score" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE conversation_threads ADD COLUMN similarity_score INTEGER NOT NULL DEFAULT 0"
                )
            )
        if "nearest_ticket_id" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE conversation_threads ADD COLUMN nearest_ticket_id VARCHAR(100)"
                )
            )
        if "cluster_id" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE conversation_threads ADD COLUMN cluster_id INTEGER NOT NULL DEFAULT 0"
                )
            )
        if "cluster_label" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE conversation_threads ADD COLUMN cluster_label VARCHAR(100) NOT NULL DEFAULT 'Cluster 0'"
                )
            )


def _to_datetime(value: Any):
    """Convert pandas timestamps or strings to datetime for SQLite storage."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        return None if converted is None else converted
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed
    except Exception:
        return None


__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "migrate_from_excel",
    "save_conversations_to_db",
    "Knowledge",
    "DATABASE_URL",
]
