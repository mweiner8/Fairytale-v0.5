"""
Application configuration
Centralizes all configuration settings
"""
import os
import logging
from datetime import timedelta


class Config:
    """Application configuration class"""

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

    # Session Configuration (24 hour expiry)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    USE_DALLE_GENERATION = os.environ.get('USE_DALLE_GENERATION', 'true').lower() == 'true'

    # Rate Limiting Configuration
    MAX_LOGIN_ATTEMPTS = 5
    RATE_LIMIT_WINDOW = 60  # seconds
    DALLE_RATE_LIMIT = 5  # requests per minute

    # Cleanup Configuration
    CLEANUP_INTERVAL_HOURS = 24

    # Story Templates
    STORY_TEMPLATES = {
        'little_red_riding_hood': {
            'title': 'Little Red Riding Hood',
            'character_name': 'Little Red Riding Hood',
            'folder': 'LRRH'
        },
        'jack_and_the_beanstalk': {
            'title': 'Jack and the Beanstalk',
            'character_name': 'Jack',
            'folder': 'JATB'
        }
    }

    # Image Validation Thresholds
    BLUR_THRESHOLD = 25
    MIN_BRIGHTNESS = 35
    MAX_BRIGHTNESS = 245

    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        import os

        # Ensure directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.OUTPUT_FOLDER, 'pages'), exist_ok=True)

    @staticmethod
    def configure_logging():
        """Configure application logging to reduce noise"""

        # Set root logger level
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Completely silence SQLAlchemy logging (all variants)
        logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.engine.Engine').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)

        # Reduce HTTP request logging noise
        logging.getLogger('httpx').setLevel(logging.WARNING)

        # Reduce gevent websocket handler noise
        logging.getLogger('geventwebsocket.handler').setLevel(logging.WARNING)

        # Reduce werkzeug (Flask) request logging noise
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

        # Keep important loggers at INFO level
        logging.getLogger('book_generator').setLevel(logging.INFO)
        logging.getLogger('ai_generator').setLevel(logging.INFO)
        logging.getLogger('socketio_handlers').setLevel(logging.INFO)
        logging.getLogger('routes.book_routes').setLevel(logging.INFO)
        logging.getLogger('image_processor').setLevel(logging.INFO)
        logging.getLogger('auth').setLevel(logging.INFO)
        logging.getLogger('db_utils').setLevel(logging.INFO)
        logging.getLogger('story_loader').setLevel(logging.INFO)

        # Set __main__ (app.py) to INFO
        logging.getLogger('__main__').setLevel(logging.INFO)
        logging.getLogger('app').setLevel(logging.INFO)