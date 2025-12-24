"""
Main application entry point
Manages Flask app initialization and route registration
"""
from gevent import monkey
monkey.patch_all()

from psycogreen.gevent import patch_psycopg
try:
    patch_psycopg()
except ImportError:
    pass

import os
import logging
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_login import LoginManager
from flask_socketio import SocketIO

# Load environment variables
load_dotenv()

# Import configuration
from config import Config

# Configure logging FIRST before any other imports
Config.configure_logging()

logger = logging.getLogger(__name__)

# Import modules (after logging is configured)
from database import init_db
import db_utils

# Import blueprints
from auth import auth_bp
from dashboard import dashboard_bp
from routes.main_routes import main_bp
from routes.book_routes import book_bp
from routes.api_routes import api_bp

# Initialize Flask app
app = Flask(__name__, static_folder='templates/static')
app.config.from_object(Config)

# Initialize extensions with longer timeouts for long-running operations
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=120,  # 2 minutes before considering connection dead
    ping_interval=25,  # Ping every 25 seconds
    logger=False,      # Disable Socket.IO's own logging
    engineio_logger=False  # Disable Engine.IO's logging
)

# Initialize database
init_db(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = None
login_manager.login_message = None

# Initialize Cloudinary
from cloudinary_utils import init_cloudinary
init_cloudinary()
logger.info("✅ Cloudinary initialized")

# Initialize OAuth
from oauth_config import init_oauth
init_oauth(app)
logger.info("✅ OAuth initialized")


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return db_utils.get_user_by_id(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    """Handle unauthorized access"""
    return jsonify({'error': 'Authentication required'}), 401


# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(main_bp)
app.register_blueprint(book_bp)
app.register_blueprint(api_bp)

# Register SocketIO handlers AFTER socketio is created
from socketio_handlers import register_socketio_handlers
register_socketio_handlers(socketio)
logger.info("✅ Socket.IO handlers registered")

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring"""
    from database import db
    from sqlalchemy import text

    try:
        db.session.execute(text('SELECT 1'))
        db_status = 'healthy'
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = 'unhealthy'

    return jsonify({
        'status': 'healthy' if db_status == 'healthy' else 'degraded',
        'service': 'fairy_tale_generator',
        'database': db_status
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    logger.info(f"🚀 Starting Flask-SocketIO server on port {port}, debug={debug}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)