"""
Routes package
"""
from routes.main_routes import main_bp
from routes.book_routes import book_bp
from routes.api_routes import api_bp

__all__ = ['main_bp', 'book_bp', 'api_bp']