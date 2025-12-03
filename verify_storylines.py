"""
Verification script to check storylines in database
Run this after init_db.py to verify everything is working
"""
from flask import Flask
from database import db, init_db
from models import Storyline
import db_utils
import json


def main():
    print("🔍 Verifying Storylines in Database\n")
    print("=" * 60)

    # Create Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test'
    init_db(app)

    with app.app_context():
        # Get all storylines
        storylines = db_utils.get_all_storylines()

        print(f"\n📊 Total Storylines: {len(storylines)}\n")

        if len(storylines) == 0:
            print("❌ No storylines found in database!")
            print("   Run: python init_db.py")
            return

        # Verify each storyline
        for storyline in storylines:
            print(f"📖 Story: {storyline.story_name}")
            print(f"   ID: {storyline.id}")
            print(f"   Gender: {storyline.gender}")
            print(f"   Created: {storyline.created_at}")

            # Check pages_json structure
            pages_json = storyline.pages_json
            print(f"   Title: {pages_json.get('title', 'N/A')}")
            print(f"   Subtitle: {pages_json.get('subtitle', 'N/A')}")

            pages = pages_json.get('pages', [])
            print(f"   Pages: {len(pages)}")

            if len(pages) > 0:
                print(f"   ✅ First page preview: {pages[0].get('text', '')[:100]}...")

                # Verify all pages have required fields
                all_valid = True
                for i, page in enumerate(pages):
                    if 'page_number' not in page or 'text' not in page:
                        print(f"   ❌ Page {i + 1} missing required fields!")
                        all_valid = False

                if all_valid:
                    print(f"   ✅ All {len(pages)} pages have valid structure")
            else:
                print("   ❌ No pages found in pages_json!")

            print()

        # Test retrieval by name
        print("=" * 60)
        print("\n🔍 Testing Story Retrieval by Name:\n")

        test_stories = [
            'Little Red Riding Hood',
            'Jack and the Beanstalk'
        ]

        for story_name in test_stories:
            storyline = db_utils.get_storyline_by_name(story_name)
            if storyline:
                print(f"✅ Found: {story_name}")
                print(f"   - Gender: {storyline.gender}")
                print(f"   - Pages: {len(storyline.pages_json.get('pages', []))}")
            else:
                print(f"❌ Not Found: {story_name}")

        # Test gender filtering
        print("\n" + "=" * 60)
        print("\n🔍 Testing Gender Filtering:\n")

        boy_stories = db_utils.get_storylines_by_gender('Boy')
        girl_stories = db_utils.get_storylines_by_gender('Girl')

        print(f"Boy stories: {len(boy_stories)}")
        for story in boy_stories:
            print(f"   - {story.story_name}")

        print(f"\nGirl stories: {len(girl_stories)}")
        for story in girl_stories:
            print(f"   - {story.story_name}")

        print("\n" + "=" * 60)
        print("\n✅ Verification Complete!\n")


if __name__ == '__main__':
    main()