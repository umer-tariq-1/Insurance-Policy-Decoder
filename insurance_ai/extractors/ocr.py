# insurance_ai/extractors/ocr.py
import pytesseract
from PIL import Image
import os

from PIL import Image, ImageFilter, ImageEnhance

def configure_tesseract():
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False


configure_tesseract()


def ocr_image_by_columns(image: Image.Image):
    """
    Split image into left & right columns and OCR separately
    """
    image = preprocess_image(image)
    width, height = image.size

    left_img = image.crop((0, 0, width // 2, height))
    right_img = image.crop((width // 2, 0, width, height))

    config = r"--oem 3 --psm 4"

    left_text = pytesseract.image_to_string(left_img, config=config)
    right_text = pytesseract.image_to_string(right_img, config=config)

    return left_text, right_text

def preprocess_image(image: Image.Image) -> Image.Image:
    # Convert to grayscale
    image = image.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Sharpen text
    image = image.filter(ImageFilter.SHARPEN)

    return image