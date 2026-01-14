import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")  # very strong


def rank_sentences(sentences, top_k=5):
    """
    Rank sentences by semantic importance.
    """
    embeddings = model.encode(sentences)
    centroid = np.mean(embeddings, axis=0)

    scores = embeddings @ centroid
    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [s for s, _ in ranked[:top_k]]
