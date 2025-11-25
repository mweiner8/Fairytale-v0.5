import io
import os
import re
import cv2
import glob
import json
import time
import uuid
import base64
import logging
import requests
import textwrap
import threading
import numpy as np
from openai import OpenAI
from threading import Lock
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from better_profanity import profanity
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from common_names import COMMON_FIRST_NAMES
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from flask_socketio import SocketIO, join_room
from werkzeug.exceptions import RequestEntityTooLarge
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template, jsonify, send_file

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['OUTPUT_FOLDER'], 'pages'), exist_ok=True)

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
USE_DALLE_GENERATION = os.environ.get('USE_DALLE_GENERATION', 'true').lower() == 'true'

try:
    openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
except Exception as init_error:
    print(f"Warning: Could not initialize OpenAI client: {init_error}")
    openai_client = None

# Story templates
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

rate_limit_lock = Lock()
rate_limit_timestamps = []

progress_tracker = {}
CLEANUP_INTERVAL_HOURS = 24


def wait_for_rate_limit():
    """Ensure we don't exceed 5 requests per minute"""
    with rate_limit_lock:
        now = time.time()
        # Remove timestamps older than 60 seconds
        rate_limit_timestamps[:] = [ts for ts in rate_limit_timestamps if now - ts < 60]

        # If we've made 5 requests in the last minute, wait
        while len(rate_limit_timestamps) >= 5:
            sleep_time = 60 - (now - rate_limit_timestamps[0]) + 1
            logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

            # Re-check after sleeping
            now = time.time()
            rate_limit_timestamps[:] = [ts for ts in rate_limit_timestamps if now - ts < 60]

        # Record this request timestamp BEFORE releasing the lock
        rate_limit_timestamps.append(time.time())


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_child_name(name):
    """
    Validate child's name input.
    
    Requirements:
    - Must be 2-20 characters long
    - Must contain only alphabetic characters (A-Z, case-insensitive)
    - Must not contain profanity
    - Must not contain digits or symbols
    - Must be a real first name (not a nonsense word)
    
    Args:
        name (str): The child's name to validate
        
    Returns:
        str or None: Error message if invalid, None if valid
    """
    if not name:
        return "Please enter a real first name (letters only, 2–20 characters)."
    
    # Check length
    if len(name) < 2 or len(name) > 20:
        return "Please enter a real first name (letters only, 2–20 characters)."
    
    # Check for only alphabetic characters (case-insensitive)
    if not re.match(r'^[A-Za-z]+$', name):
        return "Please enter a real first name (letters only, 2–20 characters)."
    
    # Check for profanity using better_profanity
    try:
        if profanity.contains_profanity(name):
            return "Please enter a real first name (letters only, 2–20 characters)."
    except Exception as e:
        logger.warning(f"Error checking profanity: {e}")
        # If profanity check fails, we'll still allow the name but log the warning
    
    # Check if it's a real first name (not a nonsense word like "Pizza", "Moo", "Keyboard")
    name_lower = name.lower()
    if name_lower not in COMMON_FIRST_NAMES:
        return "Please enter a real first name (letters only, 2–20 characters)."
    
    return None  # Name is valid


def validate_image(image_path):
    """
    Validate uploaded image for appropriateness and usability.
    
    Requirements:
    - Face detection: Confirm exactly one human face is visible
    - Image quality: Reject if blurry, underexposed, or overexposed
    - Content safety: Use OpenAI's moderation API to block unsafe content
    
    Args:
        image_path (str): Path to the image file to validate
        
    Returns:
        tuple: (True, None) if valid, (False, error_message) if invalid
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read image file. Please upload a valid image."
        
        # 1. Face Detection - Check for exactly one face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return False, "Please upload a photo with exactly one face visible."
        
        if len(faces) > 1:
            return False, "Please upload a photo with exactly one face visible."
        
        # 2. Image Quality Checks
        
        # 2a. Blur Detection using Laplacian variance
        gray_for_blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_for_blur, cv2.CV_64F).var()
        # Threshold: values below 100 typically indicate blurry images
        if laplacian_var < 100:
            return False, "Image too blurry or dark."
        
        # 2b. Exposure Check (brightness)
        # Convert to HSV and check V (value/brightness) channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        mean_brightness = np.mean(v_channel)
        
        # Underexposed: mean brightness < 50 (out of 255)
        if mean_brightness < 50:
            return False, "Image too blurry or dark."
        
        # Overexposed: mean brightness > 220 (out of 255)
        if mean_brightness > 220:
            return False, "Image too blurry or dark."
        
        # 3. Content Safety using OpenAI Moderation API
        if openai_client:
            try:
                # Read image as base64 for moderation API
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Use OpenAI's moderation endpoint for images
                # Note: OpenAI moderation API works with text, but we can use the vision API
                # to check content. However, moderation API is primarily for text.
                # For image content safety, we'll use a combination approach:
                # 1. Check if we can use the image analysis to detect inappropriate content
                # 2. For now, we'll use a basic check - in production, you might want to use
                #    a dedicated image moderation service
                
                # Convert image to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Get image format
                img_pil = Image.open(io.BytesIO(image_data))
                img_format = img_pil.format.lower() if img_pil.format else 'jpeg'
                img_pil.close()
                mime_type = f'image/{img_format}'
                image_url = f"data:{mime_type};base64,{base64_image}"
                
                # Use GPT-4 Vision to check for inappropriate content
                # This is a safety check - we ask the model to identify if content is inappropriate
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Is this image appropriate for a children's book? Respond with only 'YES' if appropriate, or 'NO' followed by a brief reason if inappropriate. Focus on: violence, adult content, inappropriate themes, or anything unsuitable for children."},
                                    {"type": "image_url", "image_url": {"url": image_url}}
                                ]
                            }
                        ],
                        max_tokens=50
                    )
                    
                    result = response.choices[0].message.content.strip().upper()
                    if result.startswith('NO'):
                        return False, "Image not appropriate for a children's book."
                    
                except Exception as vision_error:
                    logger.warning(f"Vision API check failed: {vision_error}")
                    # If vision check fails, we'll still proceed but log the warning
                    # In production, you might want to be more strict here
                    
            except Exception as moderation_error:
                logger.warning(f"Content safety check failed: {moderation_error}")
                # If moderation check fails, we'll still proceed but log the warning
                # In production, you might want to be more strict here
        
        # All checks passed
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating image: {e}", exc_info=True)
        return False, f"Error validating image: {str(e)}"


def cleanup_old_sessions():
    current_time = datetime.now()
    expired_sessions = []

    for session_id, info in list(progress_tracker.items()):
        if 'created_at' in info:
            created = info['created_at']
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if current_time - created > timedelta(hours=CLEANUP_INTERVAL_HOURS):
                expired_sessions.append(session_id)

    for expired_session_id in expired_sessions:
        try:
            if 'pdf_path' in progress_tracker[expired_session_id]:
                pdf_path = progress_tracker[expired_session_id]['pdf_path']
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)

            upload_pattern = os.path.join(app.config['UPLOAD_FOLDER'], f'{expired_session_id}_*')
            for file_path in glob.glob(upload_pattern):
                if os.path.exists(file_path):
                    os.remove(file_path)

            del progress_tracker[expired_session_id]
            logger.info(f"Cleaned up expired session: {expired_session_id}")
        except (OSError, KeyError) as cleanup_error:
            logger.error(f"Error cleaning up session {expired_session_id}: {cleanup_error}")


def analyze_image(image_path):
    if not openai_client:
        return "a child with kind features"
    try:
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()

        img = Image.open(io.BytesIO(image_data))
        img_format = img.format.lower() if img.format else 'jpeg'
        mime_type = f'image/{img_format}'
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:{mime_type};base64,{base64_image}"

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Describe only this child's face and hair in detail. Focus exclusively on: hair color and style, eye color, skin tone, facial features (nose, mouth, cheeks, face shape), and any distinctive facial characteristics. Do NOT describe clothing, body, or hands. Be specific and consistent. Format as: 'A child with [hair description], [eye color] eyes, [skin tone], [face shape], and [other facial features].'"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=200
        )

        result = response.choices[0].message.content.strip()
        
        # Check if the API refused to analyze the image
        refusal_indicators = [
            "i'm sorry",
            "i can't",
            "i cannot",
            "unable to",
            "cannot describe",
            "can't describe",
            "refuse",
            "not appropriate"
        ]
        
        result_lower = result.lower()
        if any(indicator in result_lower for indicator in refusal_indicators):
            logger.warning(f"OpenAI refused to analyze image, using fallback description. Response: {result}")
            return "a child with kind features, warm smile, and friendly appearance"
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return "a child with kind features"


def load_template_story(story_type, child_name):
    template = STORY_TEMPLATES[story_type]
    template_folder = os.path.join('templates', template['folder'])
    text_json_path = os.path.join(template_folder, 'text.json')

    with open(text_json_path, 'r', encoding='utf-8') as json_file:
        story_data = json.load(json_file)

    subtitle = story_data.get('subtitle', '')
    subtitle = subtitle.replace("(child's name)", child_name)
    story_data['subtitle'] = subtitle
    return story_data


def load_template_images(story_type):
    template = STORY_TEMPLATES[story_type]
    template_folder = os.path.join('templates', template['folder'])
    images = []

    cover_path = os.path.join(template_folder, 'cover.png')
    if os.path.exists(cover_path):
        cover_img = Image.open(cover_path).convert('RGB')
        images.append(('cover', cover_img))
    else:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")

    for page_num in range(1, 13):
        page_path = os.path.join(template_folder, f'Page {page_num}.png')
        if os.path.exists(page_path):
            page_img = Image.open(page_path).convert('RGB')
            images.append((f'page_{page_num}', page_img))
        else:
            raise FileNotFoundError(f"Page {page_num} image not found: {page_path}")

    return images


def add_text_to_image(source_img, text):
    if not text or not text.strip():
        return source_img

    result_img = source_img.copy().convert('RGB')
    width, height = result_img.size
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text_height = int(height * 0.2)
    padding = 30
    draw.rectangle([(0, 0), (width, text_height)], fill=(255, 255, 255, 230))

    try:
        font_size = max(24, int(height * 0.03))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception as exc:
        print(exc)
        font = ImageFont.load_default()

    chars_per_line = (width - 2 * padding) // 12
    lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=chars_per_line)
            lines.extend(wrapped.split('\n'))
    lines = lines[:3]

    y_pos = padding
    for line in lines:
        draw.text((padding + 2, y_pos + 2), line, fill=(0, 0, 0, 100), font=font)
        draw.text((padding, y_pos), line, fill=(40, 40, 40, 255), font=font)
        y_pos += font.size + 8

    result_img = Image.alpha_composite(result_img.convert('RGBA'), overlay).convert('RGB')
    return result_img


def resize_image(img, target_size_in_pixels):
    img_width, img_height = img.size
    if img_width != img_height:
        min_dim = min(img_width, img_height)
        left = (img_width - min_dim) // 2
        top = (img_height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((target_size_in_pixels, target_size_in_pixels), Image.Resampling.LANCZOS)
    return img


def generate_page_with_ai(template_img, child_img_path, page_text, char_description, story_type):
    """
    Generate a complete page with child's face and text overlay using GPT Image editing.
    Compatible with OpenAI Python SDK v2.7.1.
    Includes detailed logging for progress tracking.
    Now targets specific character by name to avoid editing other characters' faces.
    """
    if not openai_client or not USE_DALLE_GENERATION:
        logger.info("Using fallback text overlay (DALL-E disabled or unavailable)")
        return add_text_to_image(template_img, page_text)

    try:
        target_size = 1024
        template_work = template_img.copy().convert('RGB')

        # Crop to square if needed and to 1024x1024 for gpt-image-1
        template_work = resize_image(template_work, target_size)

        # Save temporary image for API upload
        temp_template_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            f'temp_template_{uuid.uuid4().hex[:8]}.png'
        )
        template_work.save(temp_template_path, 'PNG')

        # Determine character name based on story type
        character_name = STORY_TEMPLATES[story_type]['character_name']

        # Build prompt with specific character targeting
        prompt = f"""IMPORTANT: Only replace the face of the MAIN CHARACTER ({character_name}) in this image.
Do NOT modify any other characters' faces - leave grandmothers, wolves, giants, mothers, and all other characters exactly as they are.

Replace ONLY {character_name}'s face with a face matching this description: {char_description}.
Maintain {character_name}'s exact same pose, facial expression, and style as the original image.
Keep all other elements and all other characters identical.

Additionally, overlay the following text at the TOP of the image:
{page_text}

Don't block out any of the image. Just make sure the text color is easily legible over the image."""

        logger.info(f"Running GPT Image edit for {character_name} on page...")

        # RATE LIMITING
        wait_for_rate_limit()

        with open(temp_template_path, "rb") as img_file:
            response = openai_client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=prompt,
                size="1024x1024",
                n=1
            )

        # Decode and return the image
        image_data = response.data[0].b64_json
        edited_img = Image.open(io.BytesIO(base64.b64decode(image_data)))

        os.remove(temp_template_path)

        logger.info(f"✅ GPT Image edit completed successfully for {character_name}.")
        return edited_img.convert('RGB')

    except Exception as e:
        logger.error(f"❌ Error editing page with AI: {e}")

        # Safe fallback
        try:
            fallback_img = add_text_to_image(template_img, page_text)
            logger.info("✅ Used fallback text overlay instead (manual text applied).")
            return fallback_img
        except Exception as font_error:
            logger.error(f"❌ Fallback text overlay failed: {font_error}")
            return template_img


def generate_page_image(page_data, session_id, image_path, character_description, story_type, template_images, pages_dir):
    """
    Generate a single page image with retry logic and error handling.
    This function is designed to be called concurrently by ThreadPoolExecutor.
    
    Args:
        page_data: Dictionary containing page information (page_number, text, etc.)
        session_id: Session identifier for progress tracking
        image_path: Path to the child's uploaded image
        character_description: Description of the child's features
        story_type: Type of story being generated
        template_images: List of template images
        pages_dir: Directory to save page images
        
    Returns:
        tuple: (page_number, success, result_image, error_message, timing_info)
    """
    start_time = time.time()
    page_num = page_data.get('page_number', 0)
    page_text = page_data.get('text', '')
    thread_id = threading.current_thread().ident
    
    logger.info(f"[Thread {thread_id}] Starting generation for page {page_num}")
    
    try:
        # Get template image
        if page_num == 0:
            # Cover page
            if len(template_images) > 0:
                template_img = template_images[0][1]
            else:
                raise ValueError("Cover template image not found")
        elif 1 <= page_num <= 12:
            # Story pages (index 1-12 in template_images correspond to pages 1-12)
            if page_num < len(template_images):
                template_img = template_images[page_num][1]
            else:
                raise ValueError(f"Template image not found for page {page_num}")
        else:
            raise ValueError(f"Invalid page number: {page_num}")
        
        # Retry logic: attempt generation up to 2 times (initial + 1 retry)
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Generate complete page with AI (face + text)
                complete_page = generate_page_with_ai(
                    template_img,
                    image_path,
                    page_text,
                    character_description,
                    story_type
                )
                
                # Save page image
                if page_num == 0:
                    page_path = os.path.join(pages_dir, 'cover.png')
                    page_url = f'/page_image/{session_id}/cover'
                    page_name = 'cover'
                else:
                    page_path = os.path.join(pages_dir, f'page_{page_num}.png')
                    page_url = f'/page_image/{session_id}/{page_num}'
                    page_name = f'page_{page_num}'
                
                complete_page.save(page_path, 'PNG')
                
                elapsed_time = time.time() - start_time
                logger.info(f"[Thread {thread_id}] ✅ Page {page_num} completed in {elapsed_time:.2f}s")
                
                return (page_num, True, complete_page, None, {
                    'thread_id': thread_id,
                    'page_number': page_num,
                    'elapsed_time': elapsed_time,
                    'attempts': attempt + 1
                })
                
            except Exception as api_error:
                last_error = api_error
                error_str = str(api_error).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = any(keyword in error_str for keyword in [
                    'rate limit', 'rate_limit', '429', 'too many requests',
                    'quota', 'limit exceeded'
                ])
                
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = (attempt + 1) * 2  # 2s, 4s, etc.
                    if is_rate_limit:
                        wait_time = (attempt + 1) * 5  # Longer wait for rate limits: 5s, 10s
                    
                    logger.warning(
                        f"[Thread {thread_id}] ⚠️ Page {page_num} attempt {attempt + 1} failed: {last_error}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"[Thread {thread_id}] ❌ Page {page_num} failed after {max_retries} attempts: {last_error}"
                    )
        
        # All retries failed - use fallback
        logger.warning(f"[Thread {thread_id}] Using fallback for page {page_num}")
        fallback_img = add_text_to_image(template_img, page_text)
        
        # Save fallback image
        if page_num == 0:
            page_path = os.path.join(pages_dir, 'cover.png')
            page_url = f'/page_image/{session_id}/cover'
            page_name = 'cover'
        else:
            page_path = os.path.join(pages_dir, f'page_{page_num}.png')
            page_url = f'/page_image/{session_id}/{page_num}'
            page_name = f'page_{page_num}'
        
        fallback_img.save(page_path, 'PNG')
        
        elapsed_time = time.time() - start_time
        logger.info(f"[Thread {thread_id}] ✅ Page {page_num} completed (fallback) in {elapsed_time:.2f}s")
        
        return (page_num, True, fallback_img, f"Used fallback after {max_retries} failed attempts: {str(last_error)}", {
            'thread_id': thread_id,
            'page_number': page_num,
            'elapsed_time': elapsed_time,
            'attempts': max_retries,
            'used_fallback': True
        })
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Failed to generate page {page_num}: {str(e)}"
        logger.error(f"[Thread {thread_id}] ❌ {error_msg} (took {elapsed_time:.2f}s)")
        
        return (page_num, False, None, error_msg, {
            'thread_id': thread_id,
            'page_number': page_num,
            'elapsed_time': elapsed_time,
            'error': str(e)
        })


def create_simple_pdf(images, output_path):
    page_size = 8.5 * inch
    target_pixels = 1020
    pdf_canvas = canvas.Canvas(output_path, pagesize=(page_size, page_size))

    for source_img in images:
        source_img = source_img.convert('RGB')
        source_img = resize_image(source_img, target_pixels)

        img_buffer = io.BytesIO()
        source_img.save(img_buffer, format='PNG', dpi=(120, 120))
        img_buffer.seek(0)
        pdf_canvas.drawImage(ImageReader(img_buffer), 0, 0, width=page_size, height=page_size)
        pdf_canvas.showPage()

    pdf_canvas.save()


def generate_book_async(session_id, image_path, story_type, gender, child_name):
    """Async function to generate the entire book using AI-generated pages"""
    try:
        progress_tracker[session_id] = {
            'progress': 0,
            'status': 'Starting...',
            'error': None,
            'created_at': datetime.now().isoformat(),
            'pages': {},
            'total_pages': 13  # Cover + 12 pages
        }
        logger.info(f"Starting book generation for session {session_id}")

        # Step 1: Analyze child's image
        progress_tracker[session_id].update({'progress': 5, 'status': 'Analyzing child\'s photo...'})
        socketio.emit('progress_update', {
            'session_id': session_id,
            'progress': 5,
            'status': 'Analyzing child\'s photo...'
        }, room=session_id)
        character_description = analyze_image(image_path)
        logger.info(f"Character description: {character_description}")

        # Step 2: Load template story
        progress_tracker[session_id].update({'progress': 10, 'status': 'Loading story template...'})
        socketio.emit('progress_update', {
            'session_id': session_id,
            'progress': 10,
            'status': 'Loading story template...'
        }, room=session_id)
        story_data = load_template_story(story_type, child_name)

        # Step 3: Load template images
        progress_tracker[session_id].update({'progress': 15, 'status': 'Loading template images...'})
        socketio.emit('progress_update', {
            'session_id': session_id,
            'progress': 15,
            'status': 'Loading template images...'
        }, room=session_id)
        template_images = load_template_images(story_type)

        # Step 4: Generate complete pages with AI (face + text)
        images_for_pdf = []
        pages_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'pages', session_id)
        os.makedirs(pages_dir, exist_ok=True)

        # Process cover image (page 0)
        progress_tracker[session_id].update({'progress': 20, 'status': 'Creating cover page...'})
        socketio.emit('progress_update', {
            'session_id': session_id,
            'progress': 20,
            'status': 'Creating cover page...'
        }, room=session_id)
        cover_img = template_images[0][1]
        cover_text = f"{story_data.get('title', '')}\n{story_data.get('subtitle', '')}"

        # Generate complete cover with AI
        cover_with_face_and_text = generate_page_with_ai(
            cover_img,
            image_path,
            cover_text,
            character_description,
            story_type
        )
        images_for_pdf.append(cover_with_face_and_text)
        
        # Save cover image
        cover_path = os.path.join(pages_dir, 'cover.png')
        cover_with_face_and_text.save(cover_path, 'PNG')
        cover_url = f'/page_image/{session_id}/cover'
        
        # Update progress tracker and emit event
        progress_tracker[session_id]['pages']['cover'] = {
            'page_number': 0,
            'image_url': cover_url,
            'status': 'complete'
        }
        socketio.emit('page_complete', {
            'session_id': session_id,
            'page_number': 0,
            'image_url': cover_url,
            'status': 'complete',
            'page_name': 'cover'
        }, room=session_id)

        # Step 4b: Generate all story pages concurrently using ThreadPoolExecutor
        pages = story_data.get('pages', [])
        total_pages = len(pages)
        
        # Update status to indicate concurrent generation (cover is already done, so start at 25%)
        progress_tracker[session_id].update({
            'progress': 25,
            'status': f'Generating {total_pages} story pages concurrently...'
        })
        socketio.emit('progress_update', {
            'session_id': session_id,
            'progress': 25,
            'status': f'Generating {total_pages} story pages concurrently...'
        }, room=session_id)
        
        # Prepare page data for concurrent processing
        page_tasks = []
        for idx, page_data in enumerate(pages):
            page_num = page_data.get('page_number', idx + 1)
            page_tasks.append({
                'page_number': page_num,
                'text': page_data.get('text', ''),
                'index': idx
            })
        
        # Use ThreadPoolExecutor to generate pages concurrently
        # Limit to 5 workers at a time to avoid overwhelming the API and reaching usage limits
        max_workers = min(5, total_pages)
        page_results = {}  # Dictionary to store results by page number
        completed_count = 0
        failed_count = 0
        timing_logs = []
        
        logger.info(f"Starting concurrent generation of {total_pages} pages with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page generation tasks
            future_to_page = {}
            for page_task in page_tasks:
                future = executor.submit(
                    generate_page_image,
                    page_task,
                    session_id,
                    image_path,
                    character_description,
                    story_type,
                    template_images,
                    pages_dir
                )
                future_to_page[future] = page_task['page_number']
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    page_num_result, success, result_image, error_msg, timing_info = result
                    
                    # Log timing information
                    timing_logs.append(timing_info)
                    logger.info(
                        f"Page {page_num_result} timing: {timing_info.get('elapsed_time', 0):.2f}s, "
                        f"attempts: {timing_info.get('attempts', 1)}, "
                        f"thread: {timing_info.get('thread_id', 'N/A')}"
                    )
                    
                    if success and result_image:
                        # Store result in dictionary (ordered by page number)
                        page_results[page_num_result] = result_image
                        completed_count += 1
                        
                        # Update progress in real-time (cover is 5%, pages are 25-95%)
                        progress_per_page = 70 / total_pages
                        current_progress = 25 + int(completed_count * progress_per_page)
                        
                        # Determine page name and URL
                        if page_num_result == 0:
                            page_name = 'cover'
                            page_url = f'/page_image/{session_id}/cover'
                        else:
                            page_name = f'page_{page_num_result}'
                            page_url = f'/page_image/{session_id}/{page_num_result}'
                        
                        # Update progress tracker
                        progress_tracker[session_id]['pages'][page_name] = {
                            'page_number': page_num_result,
                            'image_url': page_url,
                            'status': 'complete',
                            'timing': timing_info
                        }
                        
                        # Emit real-time progress update
                        socketio.emit('page_complete', {
                            'session_id': session_id,
                            'page_number': page_num_result,
                            'image_url': page_url,
                            'status': 'complete',
                            'page_name': page_name
                        }, room=session_id)
                        
                        socketio.emit('progress_update', {
                            'session_id': session_id,
                            'progress': current_progress,
                            'status': f'Completed {completed_count} of {total_pages} pages...'
                        }, room=session_id)
                        
                        logger.info(f"✅ Page {page_num_result} completed successfully")
                    else:
                        failed_count += 1
                        logger.error(f"❌ Page {page_num_result} failed: {error_msg}")
                        
                        # Even if generation failed, we should still try to continue
                        # The page_results dictionary will be missing this page
                        progress_tracker[session_id]['pages'][f'page_{page_num_result}'] = {
                            'page_number': page_num_result,
                            'status': 'failed',
                            'error': error_msg
                        }
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Exception processing page {page_num}: {e}", exc_info=True)
                    progress_tracker[session_id]['pages'][f'page_{page_num}'] = {
                        'page_number': page_num,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        # Log performance summary
        if timing_logs:
            total_time = sum(t.get('elapsed_time', 0) for t in timing_logs)
            avg_time = total_time / len(timing_logs)
            max_time = max(t.get('elapsed_time', 0) for t in timing_logs)
            min_time = min(t.get('elapsed_time', 0) for t in timing_logs)
            logger.info(
                f"Performance Summary - Total: {total_time:.2f}s, "
                f"Avg: {avg_time:.2f}s, Max: {max_time:.2f}s, Min: {min_time:.2f}s, "
                f"Completed: {completed_count}/{total_pages}, Failed: {failed_count}"
            )
        
        # Check if we have enough pages to continue
        if completed_count < total_pages:
            logger.warning(
                f"Only {completed_count} of {total_pages} pages completed successfully. "
                f"Continuing with available pages."
            )
        
        # Build images_for_pdf list in correct order (cover already added, then pages 1-12)
        # Cover is already in images_for_pdf, so just add story pages in order
        for page_num in range(1, 13):
            if page_num in page_results:
                images_for_pdf.append(page_results[page_num])
            else:
                logger.warning(f"Page {page_num} missing from results, skipping in PDF")

        # Step 5: Create PDF (simplified - no text needed)
        progress_tracker[session_id].update({'progress': 95, 'status': 'Creating PDF...'})
        socketio.emit('progress_update', {
            'session_id': session_id,
            'progress': 95,
            'status': 'Creating PDF...'
        }, room=session_id)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}.pdf')
        create_simple_pdf(images_for_pdf, output_path)

        progress_tracker[session_id].update({
            'progress': 100,
            'status': 'Complete!',
            'pdf_path': output_path,
            'completed': True,
            'completed_at': datetime.now().isoformat()
        })
        socketio.emit('generation_complete', {
            'session_id': session_id,
            'pdf_path': output_path
        }, room=session_id)
        logger.info(f"Book generation completed for session {session_id}")

    except (OSError, IOError, FileNotFoundError) as file_error:
        logger.error(f"File error in book generation for session {session_id}: {file_error}", exc_info=True)
        progress_tracker[session_id] = {
            'progress': 0,
            'status': f'Error: {str(file_error)}',
            'error': str(file_error),
            'completed': False
        }
    except (json.JSONDecodeError, KeyError, ValueError) as data_error:
        logger.error(f"Data error in book generation for session {session_id}: {data_error}", exc_info=True)
        progress_tracker[session_id] = {
            'progress': 0,
            'status': f'Error: {str(data_error)}',
            'error': str(data_error),
            'completed': False
        }
    except requests.exceptions.RequestException as api_error:
        logger.error(f"API error in book generation for session {session_id}: {api_error}", exc_info=True)
        progress_tracker[session_id] = {
            'progress': 0,
            'status': f'Error: {str(api_error)}',
            'error': str(api_error),
            'completed': False
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and start book generation"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        uploaded_file = request.files['image']
        story_type = request.form.get('story_type')
        gender = request.form.get('gender')
        child_name = request.form.get('child_name', '').strip()

        if uploaded_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate child's name
        name_validation_error = validate_child_name(child_name)
        if name_validation_error:
            return jsonify({'error': name_validation_error}), 400

        # Validate file extension
        if not allowed_file(uploaded_file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Validate file is actually an image
        try:
            uploaded_file.seek(0)
            test_img = Image.open(io.BytesIO(uploaded_file.read()))
            test_img.verify()
            uploaded_file.seek(0)  # Reset file pointer
        except (OSError, IOError) as img_error:
            logger.warning(f"Invalid image file uploaded: {img_error}")
            return jsonify({'error': 'File is not a valid image'}), 400

        if story_type not in STORY_TEMPLATES:
            return jsonify({'error': 'Invalid story type'}), 400

        if gender not in ['Boy', 'Girl']:
            return jsonify({'error': 'Invalid gender selection'}), 400

        # Validate gender-story pairing
        if gender == 'Boy' and story_type != 'jack_and_the_beanstalk':
            return jsonify({'error': 'Story selection does not match gender selection. Boys can only select "Jack and the Beanstalk".'}), 400
        
        if gender == 'Girl' and story_type != 'little_red_riding_hood':
            return jsonify({'error': 'Story selection does not match gender selection. Girls can only select "Little Red Riding Hood".'}), 400

        if not openai_api_key:
            return jsonify(
                {'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'}), 500

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Save uploaded file
        filename = secure_filename(uploaded_file.filename)
        if not filename or '.' not in filename:
            filename = f'upload_{session_id[:8]}.jpg'

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
        uploaded_file.save(file_path)
        logger.info(f"File uploaded: {filename} for session {session_id}")

        # --- Convert uploaded image to safe PNG format for OpenAI ---
        try:
            img = Image.open(file_path).convert("RGB")
            safe_path = os.path.splitext(file_path)[0] + "_safe.png"
            img.save(safe_path, format="PNG")
            os.remove(file_path)  # remove original AVIF/HEIC/etc.
            file_path = safe_path
            logger.info(f"Converted uploaded image to safe PNG: {file_path}")
        except Exception as e:
            logger.error(f"Failed to convert uploaded image to PNG: {e}")
            return jsonify({'error': 'Failed to convert uploaded image to supported format.'}), 400

        # --- Validate image before starting generation ---
        is_valid, validation_error = validate_image(file_path)
        if not is_valid:
            # Clean up the uploaded file if validation fails
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up invalid image: {cleanup_error}")
            return jsonify({'error': validation_error}), 400

        # Clean up old sessions periodically
        cleanup_old_sessions()

        # Start async generation
        generation_thread = threading.Thread(
            target=generate_book_async,
            args=(session_id, file_path, story_type, gender, child_name)
        )
        generation_thread.daemon = True
        generation_thread.start()

        # Redirect to progress page
        return jsonify({
            'session_id': session_id,
            'message': 'Generation started',
            'redirect': f'/progress/{session_id}'
        })

    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413
    except (OSError, IOError) as file_error:
        logger.error(f"File upload error: {file_error}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your file'}), 500
    except ValueError as value_error:
        logger.error(f"Value error in upload: {value_error}", exc_info=True)
        return jsonify({'error': 'Invalid request data'}), 400


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    logger.error(error)
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.route('/progress/<session_id>')
def progress(session_id):
    """Display real-time progress page"""
    if session_id not in progress_tracker:
        return render_template('error.html', error='Invalid session ID'), 404
    
    return render_template('progress.html', session_id=session_id)


@app.route('/api/progress/<session_id>')
def api_progress(session_id):
    """Get generation progress (API endpoint)"""
    if session_id not in progress_tracker:
        return jsonify({'error': 'Invalid session ID'}), 404

    return jsonify(progress_tracker[session_id])


@app.route('/page_image/<session_id>/<page_identifier>')
def page_image(session_id, page_identifier):
    """Serve individual page images"""
    if session_id not in progress_tracker:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    pages_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'pages', session_id)
    
    # Handle cover or page number
    if page_identifier == 'cover':
        image_path = os.path.join(pages_dir, 'cover.png')
    else:
        try:
            page_num = int(page_identifier)
            image_path = os.path.join(pages_dir, f'page_{page_num}.png')
        except ValueError:
            return jsonify({'error': 'Invalid page identifier'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Page image not found'}), 404
    
    return send_file(image_path, mimetype='image/png')


@app.route('/download/<session_id>')
def download(session_id):
    """Download generated PDF"""
    if session_id not in progress_tracker:
        return jsonify({'error': 'Invalid session ID'}), 404

    progress_info = progress_tracker[session_id]

    if not progress_info.get('completed'):
        return jsonify({'error': 'Book not ready yet'}), 400

    pdf_path = progress_info.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF file not found'}), 404

    # Generate friendly filename
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f'fairy_tale_book_{timestamp}.pdf'

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')


@socketio.on('join_session')
def handle_join_session(data):
    """Handle client joining a session room"""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        logger.info(f'Client joined session: {session_id}')
        
        # Send current progress if available
        if session_id in progress_tracker:
            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': progress_tracker[session_id].get('progress', 0),
                'status': progress_tracker[session_id].get('status', 'Starting...')
            }, room=session_id)
            
            # Send all completed pages
            pages = progress_tracker[session_id].get('pages', {})
            for page_name, page_info in pages.items():
                if page_info.get('status') == 'complete':
                    socketio.emit('page_complete', {
                        'session_id': session_id,
                        'page_number': page_info.get('page_number'),
                        'image_url': page_info.get('image_url'),
                        'status': 'complete',
                        'page_name': page_name
                    }, room=session_id)


@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({'status': 'healthy', 'service': 'fairy_tale_generator'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)