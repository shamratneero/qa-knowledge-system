"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Knowledge(Base):
    """Knowledge base entry model."""

    __tablename__ = "knowledge"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String(500), nullable=False, index=True)
    answer = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    keywords = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Knowledge(id={self.id}, question='{self.question[:50]}...')>"


__all__ = ["Base", "Knowledge"]
