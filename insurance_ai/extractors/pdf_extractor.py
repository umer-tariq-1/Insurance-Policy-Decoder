# insurance_ai/extractors/pdf_extractor.py
import pdfplumber
from extractors.ocr import ocr_image_by_columns


def extract_text_from_pdf(file_path):
    final_text = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            width = page.width
            height = page.height

            # Try text extraction first (column-aware)
            left_bbox = (0, 0, width / 2, height)
            right_bbox = (width / 2, 0, width, height)

            left_text = page.crop(left_bbox).extract_text()
            right_text = page.crop(right_bbox).extract_text()

            if left_text or right_text:
                if left_text:
                    final_text.append(left_text)
                if right_text:
                    final_text.append(right_text)
            else:
                # OCR fallback (COLUMN-AWARE)
                image = page.to_image(resolution=300).original
                left_ocr, right_ocr = ocr_image_by_columns(image)

                if left_ocr.strip():
                    final_text.append(left_ocr)
                if right_ocr.strip():
                    final_text.append(right_ocr)

    return "\n".join(final_text).strip()
