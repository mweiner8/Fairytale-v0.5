"""
Main page routes
Handles landing page and book creation page
"""
from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def landing():
    """Landing page for guests, creation page for logged-in users"""
    if current_user.is_authenticated:
        return render_template('index.html')
    return render_template('landing.html')


@main_bp.route('/create')
@login_required
def create():
    """Main creation page - requires authentication"""
    return render_template('index.html')