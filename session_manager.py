"""
Session and progress tracking management
Handles in-memory progress tracking and session cleanup
"""
import os
import glob
import logging
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

# Global progress tracker
progress_tracker = {}


def cleanup_old_sessions():
    """Clean up expired sessions and their files"""
    current_time = datetime.now()
    expired_sessions = []

    for session_id, info in list(progress_tracker.items()):
        if 'created_at' in info:
            created = info['created_at']
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if current_time - created > timedelta(hours=Config.CLEANUP_INTERVAL_HOURS):
                expired_sessions.append(session_id)

    for expired_session_id in expired_sessions:
        try:
            # Remove PDF if exists
            if 'pdf_path' in progress_tracker[expired_session_id]:
                pdf_path = progress_tracker[expired_session_id]['pdf_path']
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)

            # Remove uploaded files
            upload_pattern = os.path.join(
                Config.UPLOAD_FOLDER,
                f'{expired_session_id}_*'
            )
            for file_path in glob.glob(upload_pattern):
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Remove from tracker
            del progress_tracker[expired_session_id]
            logger.info(f"Cleaned up expired session: {expired_session_id}")

        except (OSError, KeyError) as cleanup_error:
            logger.error(f"Error cleaning up session {expired_session_id}: {cleanup_error}")