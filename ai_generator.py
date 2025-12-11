"""
AI-powered image generation using OpenAI DALL-E
Handles rate limiting and image editing
"""
import os
import time
import uuid
import base64
import logging
from threading import Lock
from PIL import Image
import io
from config import Config
from image_processor import resize_image, add_text_to_image

logger = logging.getLogger(__name__)

# Rate limiting for DALL-E API
rate_limit_lock = Lock()
rate_limit_timestamps = []


def wait_for_rate_limit():
    """Ensure we don't exceed 5 requests per minute"""
    with rate_limit_lock:
        now = time.time()

        # Remove timestamps older than 60 seconds
        rate_limit_timestamps[:] = [
            ts for ts in rate_limit_timestamps if now - ts < 60
        ]

        # If we've made 5 requests in the last minute, wait
        while len(rate_limit_timestamps) >= Config.DALLE_RATE_LIMIT:
            sleep_time = 60 - (now - rate_limit_timestamps[0]) + 1
            logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

            # Re-check after sleeping
            now = time.time()
            rate_limit_timestamps[:] = [
                ts for ts in rate_limit_timestamps if now - ts < 60
            ]

        # Record this request timestamp
        rate_limit_timestamps.append(time.time())


def generate_page_with_ai(
        template_img,
        child_img_path,
        page_text,
        char_description,
        story_type,
        openai_client,
        upload_folder
):
    """
    Generate a complete page with child's face and text overlay using GPT Image editing.

    Args:
        template_img: PIL Image of template
        child_img_path: Path to child's photo
        page_text: Text to overlay
        char_description: Description of child's features
        story_type: Type of story (for character name)
        openai_client: OpenAI client instance
        upload_folder: Folder for temporary files

    Returns:
        PIL Image: Generated page
    """
    if not openai_client or not Config.USE_DALLE_GENERATION:
        logger.info("Using fallback text overlay (DALL-E disabled or unavailable)")
        return add_text_to_image(template_img, page_text)

    try:
        target_size = 1024
        template_work = template_img.copy().convert('RGB')

        # Crop to square and resize to 1024x1024 for gpt-image-1
        template_work = resize_image(template_work, target_size)

        # Save temporary image for API upload
        temp_template_path = os.path.join(
            upload_folder,
            f'temp_template_{uuid.uuid4().hex[:8]}.png'
        )
        template_work.save(temp_template_path, 'PNG')

        # Determine character name based on story type
        character_name = Config.STORY_TEMPLATES[story_type]['character_name']

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

        # Apply rate limiting
        wait_for_rate_limit()

        # Call OpenAI API
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

        # Clean up temporary file
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