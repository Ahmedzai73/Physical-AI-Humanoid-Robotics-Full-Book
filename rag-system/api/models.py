"""
Database models for the Physical AI & Humanoid Robotics Book RAG system
"""
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
import uuid
from datetime import datetime
from enum import Enum


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class DocumentChunk(SQLModel, table=True):
    """Model for storing document chunks in the database"""
    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: str = Field(unique=True, index=True)  # Qdrant point ID
    chapter_id: str = Field(index=True)
    chapter_title: str
    content: str
    embedding_id: Optional[str] = Field(default=None, index=True)  # Reference to embedding storage
    metadata: Optional[dict] = Field(default={})
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(SQLModel, table=True):
    """Model for storing chat sessions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class ChatMessage(SQLModel, table=True):
    """Model for storing individual chat messages"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(index=True, foreign_key="chatsession.id")
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[list] = Field(default=[])
    grounding_score: Optional[float] = Field(default=None)


class User(SQLModel, table=True):
    """Model for storing user information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    username: str = Field(unique=True, index=True)
    email: Optional[str] = Field(default=None, unique=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class BookMetadata(SQLModel, table=True):
    """Model for storing book-related metadata"""
    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: str = Field(unique=True, index=True)
    title: str
    module: str  # Module number or name
    word_count: Optional[int] = Field(default=0)
    estimated_reading_time: Optional[int] = Field(default=0)  # in minutes
    difficulty: Optional[str] = Field(default="intermediate")  # beginner, intermediate, advanced
    prerequisites: Optional[list] = Field(default=[])
    learning_objectives: Optional[list] = Field(default=[])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)