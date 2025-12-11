"""
Authentication routes and helpers
"""
import re
import time
from flask import Blueprint, request, jsonify, session, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from database import db
from models import User
import logging
from threading import Lock

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

# Rate limiting for login attempts
login_attempts = {}
login_lock = Lock()
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 60  # seconds


def is_rate_limited(ip_address):
    """Check if IP is rate limited for login attempts"""
    with login_lock:
        current_time = time.time()

        # Clean up old attempts
        if ip_address in login_attempts:
            login_attempts[ip_address] = [
                timestamp for timestamp in login_attempts[ip_address]
                if current_time - timestamp < RATE_LIMIT_WINDOW
            ]

        # Check if rate limited
        if ip_address in login_attempts and len(login_attempts[ip_address]) >= MAX_LOGIN_ATTEMPTS:
            return True

        return False


def record_login_attempt(ip_address):
    """Record a login attempt for rate limiting"""
    with login_lock:
        current_time = time.time()
        if ip_address not in login_attempts:
            login_attempts[ip_address] = []
        login_attempts[ip_address].append(current_time)


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """
    Validate password requirements:
    - Minimum 8 characters
    - At least one number
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one number"

    return True, None


@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        email = data.get('email', '').strip().lower()
        name = data.get('name', '').strip()
        password = data.get('password', '')

        # Validate inputs
        if not email or not name or not password:
            return jsonify({'error': 'Email, name, and password are required'}), 400

        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400

        if len(name) < 2 or len(name) > 100:
            return jsonify({'error': 'Name must be between 2 and 100 characters'}), 400

        is_valid, error_msg = validate_password(password)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Check if user already exists
        existing_user = db.session.query(User).filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'An account with this email already exists'}), 400

        # Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(
            email=email,
            name=name,
            password_hash=hashed_password,
            oauth_provider=None
        )

        db.session.add(new_user)
        db.session.commit()

        logger.info(f"New user registered: {email}")

        # Log the user in
        login_user(new_user, remember=False, duration=None)

        return jsonify({
            'message': 'Registration successful',
            'user': {
                'id': new_user.id,
                'email': new_user.email,
                'name': new_user.name
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({'error': 'Registration failed. Please try again.'}), 500


@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        # Get client IP for rate limiting
        ip_address = request.remote_addr

        # Check rate limiting
        if is_rate_limited(ip_address):
            return jsonify({
                'error': 'Too many login attempts. Please try again in a minute.'
            }), 429

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        # Record login attempt for rate limiting
        record_login_attempt(ip_address)

        # Find user
        user = db.session.query(User).filter_by(email=email).first()

        if not user or not user.password_hash:
            return jsonify({'error': 'Invalid email or password'}), 401

        # Check password
        if not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Log the user in (24-hour session)
        login_user(user, remember=False, duration=None)

        logger.info(f"User logged in: {email}")

        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name
            }
        }), 200

    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({'error': 'Login failed. Please try again.'}), 500


@auth_bp.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """Logout user — returns JSON for API calls, redirects for browser forms."""
    try:
        user_email = current_user.email
        logout_user()
        logger.info(f"User logged out: {user_email}")

        # If the client explicitly sends JSON, respond with JSON
        if request.is_json or request.headers.get('Content-Type') == 'application/json':
            return jsonify({'message': 'Logout successful'}), 200

        # Otherwise treat as a browser form POST → redirect
        return redirect('/')

    except Exception as e:
        logger.error(f"Logout error: {e}", exc_info=True)
        return jsonify({'error': 'Logout failed'}), 500


@auth_bp.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current user info"""
    return jsonify({
        'user': {
            'id': current_user.id,
            'email': current_user.email,
            'name': current_user.name
        }
    }), 200


@auth_bp.route('/api/auth/check', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'email': current_user.email,
                'name': current_user.name
            }
        }), 200
    else:
        return jsonify({'authenticated': False}), 200