"""
Input validation functions
Handles validation for names, images, files, etc.
"""
import re
import cv2
import base64
import logging
from PIL import Image
import io
from better_profanity import profanity
from common_names import COMMON_FIRST_NAMES
from config import Config

logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


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

    # Check if it's a real first name
    name_lower = name.lower()
    if name_lower not in COMMON_FIRST_NAMES:
        return "Please enter a real first name (letters only, 2–20 characters)."

    return None  # Name is valid


def validate_image(image_path, openai_client=None):
    """
    Improved validation of uploaded child portrait photos.
    More tolerant blur + brightness thresholds to avoid rejecting normal images.

    Args:
        image_path (str): Path to image file
        openai_client: OpenAI client for content safety check

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read the image file. Please upload a valid image."

        # Face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return False, "No face detected. Please upload a clear photo of one child."
        if len(faces) > 1:
            return False, "Multiple faces detected. Please upload a photo with exactly one child."

        # Blur check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < Config.BLUR_THRESHOLD:
            return False, "The photo is too blurry. Please upload a sharper image."

        # Exposure check
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()

        if brightness < Config.MIN_BRIGHTNESS:
            return False, "The image is too dark. Try a brighter photo."
        if brightness > Config.MAX_BRIGHTNESS:
            return False, "The image is too bright. Try a more evenly lit photo."

        # OpenAI content safety check
        if openai_client:
            try:
                with open(image_path, "rb") as f:
                    data = f.read()

                img_pil = Image.open(io.BytesIO(data))
                fmt = img_pil.format.lower() if img_pil.format else "jpeg"
                base64_img = base64.b64encode(data).decode("utf-8")
                url = f"data:image/{fmt};base64,{base64_img}"

                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Is this image safe for inclusion in a children's storybook? "
                                        "Respond with ONLY 'YES' if safe. "
                                        "Respond with 'NO: <reason>' if unsafe. "
                                        "Check for nudity, violence, injury, blood, or disturbing content."
                                    )
                                },
                                {"type": "image_url", "image_url": {"url": url}}
                            ]
                        }
                    ],
                    max_tokens=50
                )

                result = response.choices[0].message.content.strip().upper()
                if result.startswith("NO"):
                    return False, "Image not appropriate for children's content."

            except Exception as e:
                logger.warning(f"Content safety check failed: {e}")

        return True, None

    except Exception as e:
        logger.error(f"Error validating image: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"


def validate_file_upload(uploaded_file):
    """
    Validate uploaded file before processing

    Args:
        uploaded_file: Flask file upload object

    Returns:
        tuple: (is_valid, error_message)
    """
    if not uploaded_file or uploaded_file.filename == '':
        return False, 'No file selected'

    if not allowed_file(uploaded_file.filename):
        return False, f'Invalid file type. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'

    # Validate file is actually an image
    try:
        uploaded_file.seek(0)
        test_img = Image.open(io.BytesIO(uploaded_file.read()))
        test_img.verify()
        uploaded_file.seek(0)
        return True, None
    except Exception as e:
        logger.warning(f"Invalid image file uploaded: {e}")
        return False, 'File is not a valid image'