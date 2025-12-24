"""
OAuth configuration for Google authentication
"""
import os
from authlib.integrations.flask_client import OAuth  # type: ignore

oauth = OAuth()


def init_oauth(app):
    """
    Initialize OAuth with Flask app

    Args:
        app: Flask application instance

    Returns:
        OAuth: Configured OAuth instance
    """
    oauth.init_app(app)

    # Register Google OAuth
    oauth.register(
        name='google',
        client_id=os.environ.get('GOOGLE_CLIENT_ID'),
        client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )

    return oauth