# insurance_ai/services/nlp/chunker.py
import re


MAX_CHUNK_SIZE = 800  # characters


def split_by_sections(text):
    """
    Split text by insurance-style section headings
    """
    pattern = r"(Section\s+\d+.*|We Cover|We do not cover|Exclusions|Benefits|Claims|Premiums)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = []
    current = ""

    for part in parts:
        if re.match(pattern, part, flags=re.IGNORECASE):
            if current:
                sections.append(current.strip())
            current = part
        else:
            current += " " + part

    if current:
        sections.append(current.strip())

    return sections


def chunk_text(text):
    """
    Chunk text intelligently for LLM processing
    """
    cleaned_chunks = []
    sections = split_by_sections(text)

    for section in sections:
        if len(section) <= MAX_CHUNK_SIZE:
            cleaned_chunks.append(section)
        else:
            # Further split long sections
            sentences = re.split(r"(?<=[.!?])\s+", section)
            temp = ""

            for sentence in sentences:
                if len(temp) + len(sentence) < MAX_CHUNK_SIZE:
                    temp += " " + sentence
                else:
                    cleaned_chunks.append(temp.strip())
                    temp = sentence

            if temp:
                cleaned_chunks.append(temp.strip())

    return cleaned_chunks
