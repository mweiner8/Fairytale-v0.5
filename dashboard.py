"""
Dashboard routes for authenticated users
"""
from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
import db_utils
import logging

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard - shows their books and create new book option"""
    return render_template('dashboard.html')


@dashboard_bp.route('/api/dashboard/books')
@login_required
def get_user_books():
    """API endpoint to get user's books"""
    try:
        books = db_utils.get_user_books(current_user.id, limit=50)

        books_data = []
        for book in books:
            book_dict = book.to_dict()
            books_data.append(book_dict)

        return jsonify({
            'books': books_data,
            'total': len(books_data)
        }), 200

    except Exception as e:
        logger.error(f"Error fetching user books: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch books'}), 500


@dashboard_bp.route('/api/dashboard/stats')
@login_required
def get_user_stats():
    """API endpoint to get user statistics"""
    try:
        stats = db_utils.get_user_stats(current_user.id)
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error fetching user stats: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch statistics'}), 500