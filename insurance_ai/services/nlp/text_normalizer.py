# insurance_ai/services/nlp/text_normalizer.py
import re


def normalize_text(text: str) -> str:
    """
    Generic normalization that PRESERVES paragraph structure.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Remove trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Detect paragraph breaks:
    # A paragraph break is:
    # - empty line OR
    # - line ending with punctuation followed by capital letter
    text = re.sub(
        r"([.!?])\n(?=[A-Z])",
        r"\1\n\n",
        text
    )

    # Reduce excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
