# insurance_ai/extractors/docx_extractor.py
from docx import Document


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs).strip()
