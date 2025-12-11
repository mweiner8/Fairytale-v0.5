"""
PDF generation utilities
Creates print-ready PDFs from images
"""
import io
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from image_processor import resize_image

logger = logging.getLogger(__name__)


def create_simple_pdf(images, output_path):
    """
    Create a PDF from a list of images

    Args:
        images (list): List of PIL Image objects
        output_path (str): Path to save PDF
    """
    page_size = 8.5 * inch
    target_pixels = 1020
    pdf_canvas = canvas.Canvas(output_path, pagesize=(page_size, page_size))

    for source_img in images:
        # Convert to RGB and resize
        source_img = source_img.convert('RGB')
        source_img = resize_image(source_img, target_pixels)

        # Convert to buffer for PDF
        img_buffer = io.BytesIO()
        source_img.save(img_buffer, format='PNG', dpi=(120, 120))
        img_buffer.seek(0)

        # Add to PDF
        pdf_canvas.drawImage(
            ImageReader(img_buffer),
            0, 0,
            width=page_size,
            height=page_size
        )
        pdf_canvas.showPage()

    pdf_canvas.save()
    logger.info(f"PDF created successfully: {output_path}")