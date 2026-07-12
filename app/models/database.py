"""SQLAlchemy database models."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
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


class ConversationThread(Base):
    """Reconstructed conversation row stored in SQLite."""

    __tablename__ = "conversation_threads"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(100), nullable=False, index=True)
    subject = Column(String(500), nullable=False, default="")
    conversation_text = Column(Text, nullable=False)
    message_count = Column(Integer, nullable=False, default=0)
    first_sent_at = Column(DateTime, nullable=True)
    last_sent_at = Column(DateTime, nullable=True)
    source_file = Column(String(255), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="unique", index=True)
    similarity_score = Column(Integer, nullable=False, default=0)
    nearest_ticket_id = Column(String(100), nullable=True, index=True)
    cluster_id = Column(Integer, nullable=False, default=0, index=True)
    cluster_label = Column(String(100), nullable=False, default="Cluster 0", index=True)
    summary = Column(Text, nullable=False, default="")
    intent = Column(String(120), nullable=False, default="", index=True)
    keywords = Column(String(500), nullable=False, default="")
    category = Column(String(100), nullable=False, default="", index=True)
    sentiment = Column(String(20), nullable=False, default="neutral")
    priority = Column(String(20), nullable=False, default="low", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<ConversationThread(id={self.id}, ticket_id='{self.ticket_id}')>"


class KnowledgeConversation(Base):
    """Persistent historical knowledge conversation record."""

    __tablename__ = "knowledge_conversations"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(120), nullable=False, unique=True, index=True)
    ticket_id = Column(String(100), nullable=False, index=True)
    subject = Column(String(500), nullable=False, default="")
    reconstructed_conversation = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    cluster_id = Column(Integer, nullable=False, default=0, index=True)
    cluster_label = Column(String(120), nullable=False, default="Cluster 0", index=True)
    similarity = Column(Integer, nullable=False, default=0)
    classification = Column(
        String(32), nullable=False, default="new_intent", index=True
    )
    summary = Column(Text, nullable=False, default="")
    intent = Column(String(120), nullable=False, default="", index=True)
    keywords = Column(String(500), nullable=False, default="")
    category = Column(String(100), nullable=False, default="", index=True)
    sentiment = Column(String(20), nullable=False, default="neutral")
    priority = Column(String(20), nullable=False, default="low", index=True)
    upload_batch = Column(String(80), nullable=False, index=True)
    upload_timestamp = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<KnowledgeConversation(conversation_id='{self.conversation_id}', ticket_id='{self.ticket_id}')>"


__all__ = ["Base", "Knowledge", "ConversationThread", "KnowledgeConversation"]
