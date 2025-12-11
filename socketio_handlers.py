"""
SocketIO event handlers
Handles WebSocket connections for real-time updates
"""
import logging
from flask_socketio import join_room
from session_manager import progress_tracker

logger = logging.getLogger(__name__)


def register_socketio_handlers(socketio):
    """Register all Socket.IO event handlers"""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info('🔌 Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info('❌ Client disconnected')

    @socketio.on('join_session')
    def handle_join_session(data):
        """Handle client joining a session room"""
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            logger.info(f'✅ Client joined session room: {session_id}')

            # Send current progress if available
            if session_id in progress_tracker:
                current_progress = progress_tracker[session_id].get('progress', 0)
                current_status = progress_tracker[session_id].get('status', 'Starting...')

                logger.info(f'📤 Sending current progress to client: {current_progress}% - {current_status}')

                socketio.emit('progress_update', {
                    'session_id': session_id,
                    'progress': current_progress,
                    'status': current_status
                }, room=session_id)

                # Send all completed pages
                pages = progress_tracker[session_id].get('pages', {})
                completed_count = 0

                for page_name, page_info in pages.items():
                    if page_info.get('status') == 'complete':
                        completed_count += 1
                        page_number = page_info.get('page_number')

                        logger.info(f'📤 Sending completed page {page_number} to client')

                        socketio.emit('page_complete', {
                            'session_id': session_id,
                            'page_number': page_number,
                            'page_name': page_name,
                            'image_url': page_info.get('image_url'),
                            'status': 'complete'
                        }, room=session_id)

                if completed_count > 0:
                    logger.info(f'📤 Sent {completed_count} completed pages to client')

                # Check if generation is complete
                if progress_tracker[session_id].get('completed'):
                    logger.info('🎉 Generation already complete, sending completion event')
                    socketio.emit('generation_complete', {
                        'session_id': session_id,
                        'pdf_path': progress_tracker[session_id].get('pdf_path')
                    }, room=session_id)
            else:
                logger.warning(f'⚠️ Session {session_id} not found in progress tracker')
        else:
            logger.warning('⚠️ join_session called without session_id')