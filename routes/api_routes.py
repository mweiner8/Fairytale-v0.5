"""
API routes
Handles JSON API endpoints for progress, books, etc.
"""
import os
import logging
from flask import Blueprint, jsonify, send_file
from session_manager import progress_tracker
import db_utils

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/progress/<session_id>')
def get_progress(session_id):
    """Get generation progress (API endpoint)"""
    if session_id not in progress_tracker:
        return jsonify({'error': 'Invalid session ID'}), 404

    return jsonify(progress_tracker[session_id])


@api_bp.route('/books/<session_id>')
def book_details(session_id):
    """Get book details from database"""
    try:
        book = db_utils.get_book_by_session(session_id)
        if not book:
            return jsonify({'error': 'Book not found'}), 404

        return jsonify(book.to_dict())
    except Exception as e:
        logger.error(f"Error fetching book details: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/page_image/<session_id>/<page_identifier>')
def page_image(session_id, page_identifier):
    """Serve individual page images"""
    from config import Config

    if session_id not in progress_tracker:
        return jsonify({'error': 'Invalid session ID'}), 404

    pages_dir = os.path.join(Config.OUTPUT_FOLDER, 'pages', session_id)

    # Handle cover or page number
    if page_identifier == 'cover':
        image_path = os.path.join(pages_dir, 'cover.png')
    else:
        try:
            page_num = int(page_identifier)
            image_path = os.path.join(pages_dir, f'page_{page_num}.png')
        except ValueError:
            return jsonify({'error': 'Invalid page identifier'}), 400

    if not os.path.exists(image_path):
        return jsonify({'error': 'Page image not found'}), 404

    return send_file(image_path, mimetype='image/png')