"""SQLite database session management."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..core.config import settings
from ..core.logging import logger
from ..models.database import Base, Knowledge


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


def get_db() -> Session:
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


__all__ = ["engine", "SessionLocal", "get_db", "init_db", "migrate_from_excel", "Knowledge", "DATABASE_URL"]
