"""
Database initialization script
Creates all tables and optionally seeds with initial data
"""
import os
import json
from flask import Flask
from database import db, init_db
from models import User, Book, Storyline, Log
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_tables(app):
    """Create all database tables"""
    with app.app_context():
        # CHANGE THIS: Don't drop tables in production
        db.drop_all()  # Only uncomment locally when you need to reset

        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")

def seed_storylines(app):
    """Seed database with default storylines"""
    with app.app_context():
        # Check if storylines already exist
        existing_count = db.session.query(Storyline).count()
        if existing_count > 0:
            print(f"ℹ️  Storylines already exist ({existing_count} found), skipping seed")
            return

        # Load storyline data from templates
        storylines_data = [
            {
                'story_name': 'Little Red Riding Hood',
                'gender': 'Girl',
                'template_folder': 'LRRH'
            },
            {
                'story_name': 'Jack and the Beanstalk',
                'gender': 'Boy',
                'template_folder': 'JATB'
            }
        ]

        for story_data in storylines_data:
            try:
                # Load pages from template
                template_folder = os.path.join('templates', story_data['template_folder'])
                text_json_path = os.path.join(template_folder, 'text.json')

                if not os.path.exists(text_json_path):
                    print(f"❌ Error: {text_json_path} not found!")
                    print(f"   Please ensure the file exists at this location.")
                    continue

                with open(text_json_path, 'r', encoding='utf-8') as f:
                    pages_json = json.load(f)

                # Validate JSON structure
                if 'pages' not in pages_json:
                    print(f"❌ Error: {text_json_path} missing 'pages' array!")
                    continue

                if len(pages_json['pages']) != 12:
                    print(f"⚠️  Warning: {story_data['story_name']} has {len(pages_json['pages'])} pages (expected 12)")

                # Validate each page
                for i, page in enumerate(pages_json['pages']):
                    if 'page_number' not in page or 'text' not in page:
                        print(f"❌ Error: Page {i+1} in {story_data['story_name']} missing required fields!")
                        continue

                # Create storyline
                storyline = Storyline(
                    story_name=story_data['story_name'],
                    gender=story_data['gender'],
                    pages_json=pages_json
                )
                db.session.add(storyline)
                print(f"✅ Added storyline: {story_data['story_name']} ({len(pages_json['pages'])} pages)")

            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON for {story_data['story_name']}: {e}")
            except Exception as e:
                print(f"❌ Error adding storyline {story_data['story_name']}: {e}")

        # Commit all storylines
        try:
            db.session.commit()
            print("✅ Storylines seeded successfully")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error committing storylines: {e}")

def create_test_user(app):
    """Create a test user for development"""
    with app.app_context():
        # Check if test user exists
        test_user = db.session.query(User).filter_by(email='test@example.com').first()
        if test_user:
            print("ℹ️  Test user already exists, skipping")
            return

        try:
            user = User(
                email='test@example.com',
                name='Test User',
                oauth_provider='dev'
            )
            db.session.add(user)
            db.session.commit()
            print("✅ Test user created successfully")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error creating test user: {e}")

def main():
    """Main initialization function"""
    print("🚀 Starting database initialization...")

    # Create Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

    # Initialize database
    init_db(app)

    # Create tables
    create_tables(app)

    # Seed data
    seed_storylines(app)

    # Create test user (only in development)
    if os.environ.get('FLASK_ENV') != 'production':
        create_test_user(app)

    print("✅ Database initialization complete!")
    print(f"📊 Database location: {app.config['SQLALCHEMY_DATABASE_URI']}")

if __name__ == '__main__':
    main()