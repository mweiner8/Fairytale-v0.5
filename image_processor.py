"""
Image processing utilities
Handles image analysis, editing, resizing, and text overlay
"""
import io
import os
import base64
import logging
import textwrap
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def analyze_image(image_path, openai_client):
    """
    Analyze child's image to extract facial features

    Args:
        image_path (str): Path to image file
        openai_client: OpenAI client instance

    Returns:
        str: Description of child's features
    """
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
                        {
                            "type": "text",
                            "text": (
                                "Describe only this child's face and hair in detail. "
                                "Focus exclusively on: hair color and style, eye color, skin tone, "
                                "facial features (nose, mouth, cheeks, face shape), and any distinctive "
                                "facial characteristics. Do NOT describe clothing, body, or hands. "
                                "Be specific and consistent. Format as: 'A child with [hair description], "
                                "[eye color] eyes, [skin tone], [face shape], and [other facial features].'"
                            )
                        },
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=200
        )

        result = response.choices[0].message.content.strip()

        # Check if the API refused to analyze the image
        refusal_indicators = [
            "i'm sorry", "i can't", "i cannot", "unable to",
            "cannot describe", "can't describe", "refuse", "not appropriate"
        ]

        result_lower = result.lower()
        if any(indicator in result_lower for indicator in refusal_indicators):
            logger.warning(f"OpenAI refused to analyze image, using fallback. Response: {result}")
            return "a child with kind features, warm smile, and friendly appearance"

        return result

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return "a child with kind features"


def resize_image(img, target_size_in_pixels):
    """
    Resize image to target size, cropping to square if needed

    Args:
        img: PIL Image object
        target_size_in_pixels (int): Target size

    Returns:
        PIL Image: Resized image
    """
    img_width, img_height = img.size

    # Crop to square if needed
    if img_width != img_height:
        min_dim = min(img_width, img_height)
        left = (img_width - min_dim) // 2
        top = (img_height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

    # Resize
    img = img.resize(
        (target_size_in_pixels, target_size_in_pixels),
        Image.Resampling.LANCZOS
    )
    return img


def add_text_to_image(source_img, text):
    """
    Add text overlay to image

    Args:
        source_img: PIL Image object
        text (str): Text to overlay

    Returns:
        PIL Image: Image with text overlay
    """
    if not text or not text.strip():
        return source_img

    result_img = source_img.copy().convert('RGB')
    width, height = result_img.size
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Create text area
    text_height = int(height * 0.2)
    padding = 30
    draw.rectangle([(0, 0), (width, text_height)], fill=(255, 255, 255, 230))

    # Load font
    try:
        font_size = max(24, int(height * 0.03))
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size
        )
    except Exception as e:
        logger.warning(f"Could not load font: {e}")
        font = ImageFont.load_default()

    # Wrap and draw text
    chars_per_line = (width - 2 * padding) // 12
    lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=chars_per_line)
            lines.extend(wrapped.split('\n'))
    lines = lines[:3]

    y_pos = padding
    for line in lines:
        # Shadow
        draw.text((padding + 2, y_pos + 2), line, fill=(0, 0, 0, 100), font=font)
        # Text
        draw.text((padding, y_pos), line, fill=(40, 40, 40, 255), font=font)
        y_pos += font.size + 8

    result_img = Image.alpha_composite(
        result_img.convert('RGBA'),
        overlay
    ).convert('RGB')

    return result_img


def convert_to_safe_format(file_path):
    """
    Convert image to safe PNG format for OpenAI

    Args:
        file_path (str): Path to image file

    Returns:
        str: Path to converted file
    """
    try:
        img = Image.open(file_path).convert("RGB")
        safe_path = os.path.splitext(file_path)[0] + "_safe.png"
        img.save(safe_path, format="PNG")

        # Remove original if different
        if safe_path != file_path:
            os.remove(file_path)

        logger.info(f"Converted image to safe PNG: {safe_path}")
        return safe_path

    except Exception as e:
        logger.error(f"Failed to convert image to PNG: {e}")
        raise