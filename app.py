import os
import json
import time
import uuid
import logging
import glob
import io
import base64
import textwrap
from datetime import datetime, timedelta
import threading
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import requests
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from dotenv import load_dotenv
from flask_socketio import SocketIO

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

progress_tracker = {}
CLEANUP_INTERVAL_HOURS = 24


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

        return response.choices[0].message.content
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

        # Crop to square if needed
        img_width, img_height = template_work.size
        if img_width != img_height:
            min_dim = min(img_width, img_height)
            left = (img_width - min_dim) // 2
            top = (img_height - min_dim) // 2
            template_work = template_work.crop((left, top, left + min_dim, top + min_dim))

        # Resize to 1024x1024 for gpt-image-1
        template_work = template_work.resize((target_size, target_size), Image.Resampling.LANCZOS)

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

        # ✅ Correct API for OpenAI Python SDK v2.7.1
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


def create_simple_pdf(images, output_path):
    page_size = 8.5 * inch
    target_pixels = 1020
    pdf_canvas = canvas.Canvas(output_path, pagesize=(page_size, page_size))

    for source_img in images:
        source_img = source_img.convert('RGB')
        img_width, img_height = source_img.size
        if img_width != img_height:
            min_dim = min(img_width, img_height)
            left = (img_width - min_dim) // 2
            top = (img_height - min_dim) // 2
            source_img = source_img.crop((left, top, left + min_dim, top + min_dim))
        source_img = source_img.resize((target_pixels, target_pixels), Image.Resampling.LANCZOS)

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
            'created_at': datetime.now().isoformat()
        }
        logger.info(f"Starting book generation for session {session_id}")

        # Step 1: Analyze child's image
        progress_tracker[session_id] = {'progress': 5, 'status': 'Analyzing child\'s photo...'}
        character_description = analyze_image(image_path)
        logger.info(f"Character description: {character_description}")

        # Step 2: Load template story
        progress_tracker[session_id] = {'progress': 10, 'status': 'Loading story template...'}
        story_data = load_template_story(story_type, child_name)

        # Step 3: Load template images
        progress_tracker[session_id] = {'progress': 15, 'status': 'Loading template images...'}
        template_images = load_template_images(story_type)

        # Step 4: Generate complete pages with AI (face + text)
        images_for_pdf = []

        # Process cover image
        progress_tracker[session_id] = {'progress': 20, 'status': 'Creating cover page...'}
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

        # Process story pages
        pages = story_data.get('pages', [])
        for idx, page_data in enumerate(pages):
            page_num = page_data.get('page_number', idx + 1)
            the_progress = 20 + int((idx + 1) / len(pages) * 75)
            progress_tracker[session_id] = {
                'progress': the_progress,
                'status': f'Creating page {page_num} of {len(pages)}...'
            }

            # Get template image
            if 1 <= page_num <= 12 and page_num < len(template_images):
                template_img = template_images[page_num][1]
                page_text = page_data.get('text', '')

                # Generate complete page with AI (face + text)
                complete_page = generate_page_with_ai(
                    template_img,
                    image_path,
                    page_text,
                    character_description,
                    story_type
                )
                images_for_pdf.append(complete_page)
            else:
                logger.warning(f"No image found for page {page_num}")

            # Small delay to avoid rate limiting
            time.sleep(1)

        # Step 5: Create PDF (simplified - no text needed)
        progress_tracker[session_id] = {'progress': 95, 'status': 'Creating PDF...'}
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{session_id}.pdf')
        create_simple_pdf(images_for_pdf, output_path)

        progress_tracker[session_id] = {
            'progress': 100,
            'status': 'Complete!',
            'pdf_path': output_path,
            'completed': True,
            'completed_at': datetime.now().isoformat()
        }
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

        if not child_name:
            return jsonify({'error': 'Child\'s name is required'}), 400

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

        # Clean up old sessions periodically
        cleanup_old_sessions()

        # Start async generation
        generation_thread = threading.Thread(
            target=generate_book_async,
            args=(session_id, file_path, story_type, gender, child_name)
        )
        generation_thread.daemon = True
        generation_thread.start()

        return jsonify({
            'session_id': session_id,
            'message': 'Generation started'
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
    """Get generation progress"""
    if session_id not in progress_tracker:
        return jsonify({'error': 'Invalid session ID'}), 404

    return jsonify(progress_tracker[session_id])


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


@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({'status': 'healthy', 'service': 'fairy_tale_generator'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)