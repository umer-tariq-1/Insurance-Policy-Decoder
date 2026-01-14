# insurance_ai/services/nlp/chunk_filter.py
import re


def is_informative_chunk(text: str) -> bool:
    text = text.strip()

    # Too short
    if len(text) < 200:
        return False

    # Mostly numbers or symbols
    if len(re.findall(r"[A-Za-z]", text)) < 50:
        return False

    # Likely header / index / boilerplate
    if text.isupper():
        return False

    return True
