import re


def split_into_sentences(text: str):
    """
    Generic sentence splitter.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences
