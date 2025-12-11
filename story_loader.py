"""
Story template loading
Handles loading story data and template images
"""
import os
import logging
from PIL import Image
from config import Config
import db_utils

logger = logging.getLogger(__name__)


def load_template_story(story_type, child_name):
    """
    Load story template from database

    Args:
        story_type: Story type key (e.g., 'little_red_riding_hood')
        child_name: Child's name to substitute in subtitle

    Returns:
        dict: Story data with title, subtitle, and pages
    """
    # Get story title from config
    template = Config.STORY_TEMPLATES[story_type]
    story_title = template['title']

    # Fetch storyline from database
    storyline = db_utils.get_storyline_by_name(story_title)

    if not storyline:
        logger.error(f"Storyline not found in database: {story_title}")
        raise ValueError(f"Story template '{story_title}' not found in database")

    # Get story data from database
    story_data = storyline.pages_json.copy()

    # Substitute child's name in subtitle
    subtitle = story_data.get('subtitle', '')
    subtitle = subtitle.replace("(child's name)", child_name)
    story_data['subtitle'] = subtitle

    logger.info(
        f"Loaded story '{story_title}' from database with "
        f"{len(story_data.get('pages', []))} pages"
    )

    return story_data


def load_template_images(story_type):
    """
    Load template images for a story

    Args:
        story_type: Story type key

    Returns:
        list: List of tuples (page_name, PIL Image)
    """
    template = Config.STORY_TEMPLATES[story_type]
    template_folder = os.path.join('templates', template['folder'])
    images = []

    # Load cover image
    cover_path = os.path.join(template_folder, 'cover.png')
    if os.path.exists(cover_path):
        cover_img = Image.open(cover_path).convert('RGB')
        images.append(('cover', cover_img))
    else:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")

    # Load page images (1-12)
    for page_num in range(1, 13):
        page_path = os.path.join(template_folder, f'Page {page_num}.png')
        if os.path.exists(page_path):
            page_img = Image.open(page_path).convert('RGB')
            images.append((f'page_{page_num}', page_img))
        else:
            raise FileNotFoundError(f"Page {page_num} image not found: {page_path}")

    return images


def validate_story_selection(story_type, gender):
    """
    Validate that story selection matches gender

    Args:
        story_type (str): Story type key
        gender (str): 'Boy' or 'Girl'

    Returns:
        tuple: (is_valid, error_message)
    """
    if story_type not in Config.STORY_TEMPLATES:
        return False, 'Invalid story type'

    if gender not in ['Boy', 'Girl']:
        return False, 'Invalid gender selection'

    # Validate gender-story pairing
    if gender == 'Boy' and story_type != 'jack_and_the_beanstalk':
        return False, (
            'Story selection does not match gender selection. '
            'Boys can only select "Jack and the Beanstalk".'
        )

    if gender == 'Girl' and story_type != 'little_red_riding_hood':
        return False, (
            'Story selection does not match gender selection. '
            'Girls can only select "Little Red Riding Hood".'
        )

    return True, None