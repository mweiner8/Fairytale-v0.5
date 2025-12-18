"""
Database models for the Fairy Tale Generator
"""
from datetime import datetime, timezone
from database import db
from sqlalchemy import Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List
from flask_login import UserMixin

def utc_now():
    """Return current UTC time with timezone awareness"""
    return datetime.now(timezone.utc)

class User(UserMixin, db.Model):
    """User model for authentication and tracking"""
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    oauth_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    # Relationships
    books: Mapped[List["Book"]] = relationship("Book", back_populates="user", cascade="all, delete-orphan")
    logs: Mapped[List["Log"]] = relationship("Log", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f'<User {self.email}>'

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'oauth_provider': self.oauth_provider,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Storyline(db.Model):
    """Storyline template model"""
    __tablename__ = 'storylines'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    story_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)  # 'Boy', 'Girl', or 'Any'
    pages_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # Store story pages as JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    # Relationships
    books: Mapped[List["Book"]] = relationship("Book", back_populates="storyline")

    def __repr__(self):
        return f'<Storyline {self.story_name}>'

    def to_dict(self):
        """Convert storyline to dictionary"""
        return {
            'id': self.id,
            'story_name': self.story_name,
            'gender': self.gender,
            'pages_json': self.pages_json,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Book(db.Model):
    """Book generation model"""
    __tablename__ = 'books'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    story_id: Mapped[int] = mapped_column(Integer, ForeignKey('storylines.id'), nullable=False, index=True)
    child_name: Mapped[str] = mapped_column(String(100), nullable=False)

    pdf_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Cloudinary URL
    cloudinary_public_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # For deletion

    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default='pending', nullable=False)  # pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="books")
    storyline: Mapped["Storyline"] = relationship("Storyline", back_populates="books")

    def __repr__(self):
        return f'<Book {self.id} - {self.child_name}>'

    def to_dict(self):
        """Convert book to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'story_id': self.story_id,
            'story_name': self.storyline.story_name if self.storyline else None,
            'child_name': self.child_name,
            'pdf_url': self.pdf_url,
            'session_id': self.session_id,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class Log(db.Model):
    """Application logging model"""
    __tablename__ = 'logs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, CRITICAL
    message: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="logs")

    def __repr__(self):
        return f'<Log {self.level} - {self.timestamp}>'

    def to_dict(self):
        """Convert log to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'level': self.level,
            'message': self.message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'session_id': self.session_id
        }