"""
Database configuration and session management
"""
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


def init_db(app):
    """
    Initialize database with Flask app

    Args:
        app: Flask application instance
    """
    # Database URI - SQLite for dev, PostgreSQL for production
    database_url = os.environ.get('DATABASE_URL')

    if database_url:
        # Handle Heroku postgres:// vs postgresql:// issue
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    else:
        # Default to SQLite for development
        db_path = os.path.join(os.path.dirname(__file__), 'fairy_tale.db')
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

    # Disable modification tracking (saves resources)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Echo SQL queries in debug mode
    app.config['SQLALCHEMY_ECHO'] = os.environ.get('FLASK_ENV') != 'production'

    # Initialize the database
    db.init_app(app)

    return db