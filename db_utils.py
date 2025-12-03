"""
Database utility functions for common operations
"""
from datetime import datetime, timezone
from database import db
from models import User, Book, Storyline, Log
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ============= User Functions =============

def get_or_create_user(email: str, name: str, oauth_provider: Optional[str] = None) -> User:
    """
    Get existing user or create a new one

    Args:
        email: User's email address
        name: User's name
        oauth_provider: OAuth provider name (e.g., 'google', 'github')

    Returns:
        User object
    """
    user = db.session.query(User).filter_by(email=email).first()

    if not user:
        user = User(
            email=email,
            name=name,
            oauth_provider=oauth_provider
        )
        db.session.add(user)
        db.session.commit()
        logger.info(f"Created new user: {email}")

    return user

def get_user_by_id(user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.session.query(User).filter_by(id=user_id).first()

def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email"""
    return db.session.query(User).filter_by(email=email).first()

# ============= Storyline Functions =============

def get_storyline_by_name(story_name: str) -> Optional[Storyline]:
    """Get storyline by name"""
    return db.session.query(Storyline).filter_by(story_name=story_name).first()

def get_storyline_by_id(story_id: int) -> Optional[Storyline]:
    """Get storyline by ID"""
    return db.session.query(Storyline).filter_by(id=story_id).first()

def get_all_storylines():
    """Get all available storylines"""
    return db.session.query(Storyline).all()

def get_storylines_by_gender(gender: str):
    """Get storylines filtered by gender"""
    return db.session.query(Storyline).filter(
        (Storyline.gender == gender) | (Storyline.gender == 'Any')
    ).all()

# ============= Book Functions =============

def create_book(session_id: str, story_id: int, child_name: str, user_id: Optional[int] = None) -> Book:
    """
    Create a new book record

    Args:
        session_id: Unique session identifier
        story_id: Storyline ID
        child_name: Child's name for the book
        user_id: Optional user ID if authenticated

    Returns:
        Book object
    """
    book = Book(
        session_id=session_id,
        story_id=story_id,
        child_name=child_name,
        user_id=user_id,
        status='pending'
    )
    db.session.add(book)
    db.session.commit()
    logger.info(f"Created book record for session: {session_id}")
    return book

def update_book_status(session_id: str, status: str, pdf_path: Optional[str] = None,
                      error_message: Optional[str] = None):
    """
    Update book generation status

    Args:
        session_id: Session identifier
        status: New status ('pending', 'processing', 'completed', 'failed')
        pdf_path: Path to generated PDF (for completed books)
        error_message: Error message (for failed books)
    """
    book = db.session.query(Book).filter_by(session_id=session_id).first()
    if not book:
        logger.warning(f"Book not found for session: {session_id}")
        return

    book.status = status

    if pdf_path:
        book.pdf_path = pdf_path

    if error_message:
        book.error_message = error_message

    if status == 'completed':
        book.completed_at = datetime.now(timezone.utc)

    db.session.commit()
    logger.info(f"Updated book status to '{status}' for session: {session_id}")

def get_book_by_session(session_id: str) -> Optional[Book]:
    """Get book by session ID"""
    return db.session.query(Book).filter_by(session_id=session_id).first()

def get_user_books(user_id: int, limit: int = 10):
    """Get recent books for a user"""
    return db.session.query(Book).filter_by(user_id=user_id)\
        .order_by(Book.created_at.desc())\
        .limit(limit)\
        .all()

def delete_book(book_id: int) -> bool:
    """
    Delete a book record

    Args:
        book_id: Book ID to delete

    Returns:
        True if deleted, False if not found
    """
    book = db.session.query(Book).filter_by(id=book_id).first()
    if not book:
        return False

    db.session.delete(book)
    db.session.commit()
    logger.info(f"Deleted book: {book_id}")
    return True

# ============= Log Functions =============

def create_log(level: str, message: str, user_id: Optional[int] = None,
               session_id: Optional[str] = None):
    """
    Create a log entry

    Args:
        level: Log level (INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        user_id: Optional user ID
        session_id: Optional session ID
    """
    log = Log(
        level=level,
        message=message,
        user_id=user_id,
        session_id=session_id
    )
    db.session.add(log)
    db.session.commit()

def get_logs(level: Optional[str] = None, user_id: Optional[int] = None,
             limit: int = 100):
    """
    Get logs with optional filtering

    Args:
        level: Filter by log level
        user_id: Filter by user ID
        limit: Maximum number of logs to return

    Returns:
        List of log entries
    """
    query = db.session.query(Log)

    if level:
        query = query.filter_by(level=level)

    if user_id:
        query = query.filter_by(user_id=user_id)

    return query.order_by(Log.timestamp.desc()).limit(limit).all()

def get_session_logs(session_id: str):
    """Get all logs for a specific session"""
    return db.session.query(Log).filter_by(session_id=session_id)\
        .order_by(Log.timestamp.asc())\
        .all()

# ============= Statistics Functions =============

def get_user_stats(user_id: int) -> dict:
    """
    Get statistics for a user

    Args:
        user_id: User ID

    Returns:
        Dictionary with user statistics
    """
    total_books = db.session.query(Book).filter_by(user_id=user_id).count()
    completed_books = db.session.query(Book).filter_by(
        user_id=user_id,
        status='completed'
    ).count()
    failed_books = db.session.query(Book).filter_by(
        user_id=user_id,
        status='failed'
    ).count()

    return {
        'total_books': total_books,
        'completed_books': completed_books,
        'failed_books': failed_books,
        'success_rate': (completed_books / total_books * 100) if total_books > 0 else 0
    }

def get_app_stats() -> dict:
    """
    Get overall application statistics

    Returns:
        Dictionary with application statistics
    """
    total_users = db.session.query(User).count()
    total_books = db.session.query(Book).count()
    completed_books = db.session.query(Book).filter_by(status='completed').count()

    return {
        'total_users': total_users,
        'total_books': total_books,
        'completed_books': completed_books,
        'success_rate': (completed_books / total_books * 100) if total_books > 0 else 0
    }