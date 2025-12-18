"""
Cloudinary utility functions for PDF storage
"""
import logging
import cloudinary
import cloudinary.uploader
from config import Config

logger = logging.getLogger(__name__)


def init_cloudinary():
    """Initialize Cloudinary with credentials"""
    cloudinary.config(
        cloud_name=Config.CLOUDINARY_CLOUD_NAME,
        api_key=Config.CLOUDINARY_API_KEY,
        api_secret=Config.CLOUDINARY_API_SECRET,
        secure=True
    )
    logger.info("✅ Cloudinary initialized")


def upload_pdf(file_path: str, public_id: str) -> dict:
    """
    Upload PDF to Cloudinary

    Args:
        file_path: Local path to PDF file
        public_id: Unique identifier for the file (use session_id)

    Returns:
        dict with 'url' and 'secure_url'
    """
    try:
        result = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            resource_type="raw",  # For PDFs
            folder="fairy_tale_pdfs",  # Organize in a folder
            overwrite=True
        )

        logger.info(f"✅ Uploaded PDF to Cloudinary: {public_id}")
        return {
            'url': result.get('url'),
            'secure_url': result.get('secure_url'),
            'public_id': result.get('public_id')
        }
    except Exception as e:
        logger.error(f"❌ Failed to upload PDF to Cloudinary: {e}")
        raise


def delete_pdf(public_id: str) -> bool:
    """
    Delete PDF from Cloudinary

    Args:
        public_id: Cloudinary public ID

    Returns:
        True if successful
    """
    try:
        result = cloudinary.uploader.destroy(
            public_id,
            resource_type="raw"
        )
        logger.info(f"✅ Deleted PDF from Cloudinary: {public_id}")
        return result.get('result') == 'ok'
    except Exception as e:
        logger.error(f"❌ Failed to delete PDF from Cloudinary: {e}")
        return False


def get_pdf_url(public_id: str) -> str:
    """
    Get secure URL for a PDF

    Args:
        public_id: Cloudinary public ID

    Returns:
        Secure HTTPS URL
    """
    return cloudinary.CloudinaryResource(
        public_id,
        resource_type="raw"
    ).url