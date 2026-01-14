# insurance_ai/services/nlp/semantic_chunker.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk_text(text, similarity_threshold=0.65):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) <= 1:
        return paragraphs

    embeddings = model.encode(paragraphs)
    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if sim >= similarity_threshold:
            current_chunk.append(paragraphs[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraphs[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
