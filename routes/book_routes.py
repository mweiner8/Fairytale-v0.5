"""
Book management routes
Handles upload, progress, download, and progress API for polling
"""
import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from validators import validate_child_name, validate_file_upload, validate_image
from story_loader import validate_story_selection
from image_processor import convert_to_safe_format
from book_generator import generate_book_async
from session_manager import progress_tracker, cleanup_old_sessions
import db_utils

from gevent import spawn

logger = logging.getLogger(__name__)

book_bp = Blueprint('book', __name__)


@book_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    """Handle file upload and start book generation"""
    try:
        # Import here to avoid circular imports
        from app import app, socketio
        from openai import OpenAI

        openai_client = None
        if Config.OPENAI_API_KEY:
            openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        uploaded_file = request.files['image']
        story_type = request.form.get('story_type')
        gender = request.form.get('gender')
        child_name = request.form.get('child_name', '').strip()

        # Validate child's name
        name_validation_error = validate_child_name(child_name)
        if name_validation_error:
            return jsonify({'error': name_validation_error}), 400

        # Validate file
        is_valid, error = validate_file_upload(uploaded_file)
        if not is_valid:
            return jsonify({'error': error}), 400

        # Validate story selection
        is_valid, error = validate_story_selection(story_type, gender)
        if not is_valid:
            return jsonify({'error': error}), 400

        if not Config.OPENAI_API_KEY:
            return jsonify({
                'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'
            }), 500

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Save uploaded file
        filename = secure_filename(uploaded_file.filename)
        if not filename or '.' not in filename:
            filename = f'upload_{session_id[:8]}.jpg'

        file_path = os.path.join(Config.UPLOAD_FOLDER, f'{session_id}_{filename}')
        uploaded_file.save(file_path)
        logger.info(f"File uploaded: {filename} for session {session_id}")

        # Convert to safe PNG format
        file_path = convert_to_safe_format(file_path)

        # Validate image
        is_valid, validation_error = validate_image(file_path, openai_client)
        if not is_valid:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up invalid image: {cleanup_error}")
            return jsonify({'error': validation_error}), 400

        # Create book record in database
        try:
            story_title = Config.STORY_TEMPLATES[story_type]['title']
            storyline = db_utils.get_storyline_by_name(story_title)

            if not storyline:
                logger.error(f"Storyline not found in database: {story_title}")
                return jsonify({'error': 'Story template not found'}), 500

            book = db_utils.create_book(
                session_id=session_id,
                story_id=storyline.id,
                child_name=child_name,
                user_id=current_user.id
            )

            db_utils.update_book_status(session_id, 'processing')
            db_utils.create_log(
                level='INFO',
                message=f'Started book generation for {child_name}',
                session_id=session_id
            )

        except Exception as db_error:
            logger.error(f"Database error creating book record: {db_error}")

        # Clean up old sessions
        cleanup_old_sessions()

        # Initialize progress_tracker entry - start empty, pages fill as they complete
        progress_tracker[session_id] = {
            "progress": 0,
            "status": "Starting...",
            "completed": False,
            "pdf_path": None,
            "pages": {},  # Start empty, will be populated as pages complete
            "total_pages": 13
        }

        logger.info(f"Initialized progress tracker for session {session_id}")

        # Start async generation using gevent
        spawn(generate_book_async, session_id, file_path, story_type, gender,
              child_name, current_user.id, progress_tracker, socketio, app, openai_client)

        logger.info(f"Spawned async book generation for session {session_id}")

        return jsonify({
            'session_id': session_id,
            'message': 'Generation started',
            'redirect': f'/progress/{session_id}'
        })

    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your file'}), 500


@book_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    logger.error(error)
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@book_bp.route('/progress/<session_id>')
@login_required
def progress(session_id):
    """Display real-time progress page"""
    book = db_utils.get_book_by_session(session_id)
    if not book or book.user_id != current_user.id:
        return render_template('error.html', error='Unauthorized access'), 403

    logger.info(f"Rendering progress page for session {session_id}")

    return render_template(
        'progress.html',
        session_id=session_id,
        child_name=book.child_name,
        story_title=getattr(book, 'story_title', 'Your Storybook')
    )


@book_bp.route('/api/progress/<session_id>')
@login_required
def api_progress(session_id):
    """Return progress data for fallback polling"""
    if session_id not in progress_tracker:
        logger.warning(f"Session {session_id} not found in progress tracker")
        return jsonify({'error': 'Invalid session ID'}), 404

    book = db_utils.get_book_by_session(session_id)
    if not book or book.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403

    progress_info = progress_tracker[session_id]

    # Return the pages as-is (they're keyed by page_name like 'cover', 'page_1', etc.)
    return jsonify({
        'progress': progress_info.get('progress', 0),
        'status': progress_info.get('status', ''),
        'completed': progress_info.get('completed', False),
        'pages': progress_info.get('pages', {})
    })


@book_bp.route('/download/<session_id>')
@login_required
def download(session_id):
    """Download generated PDF"""
    book = db_utils.get_book_by_session(session_id)
    if not book or book.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403

    progress_info = progress_tracker.get(session_id)
    if not progress_info or not progress_info.get('completed'):
        return jsonify({'error': 'Book not ready yet'}), 400

    pdf_path = progress_info.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF file not found'}), 404

    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f'fairy_tale_book_{timestamp}.pdf'

    logger.info(f"Serving PDF download for session {session_id}")

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )


@book_bp.route('/page_image/<session_id>/<page_name>')
@login_required
def serve_page_image(session_id, page_name):
    """Serve generated page images for live preview"""
    # Ensure session exists and belongs to user
    book = db_utils.get_book_by_session(session_id)
    if not book or book.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403

    pages_dir = os.path.join(Config.OUTPUT_FOLDER, 'pages', session_id)
    if page_name == 'cover':
        filename = 'cover.png'
    else:
        filename = f'{page_name}.png'

    file_path = os.path.join(pages_dir, filename)
    if not os.path.exists(file_path):
        logger.warning(f"Image not found: {file_path}")
        return jsonify({'error': 'Image not found'}), 404

    return send_file(file_path, mimetype='image/png')