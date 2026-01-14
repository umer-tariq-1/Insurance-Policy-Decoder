# insurance_ai/services/nlp/importance.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def score_importance(summaries):
    embeddings = model.encode(summaries)
    centroid = np.mean(embeddings, axis=0)

    scores = []
    for emb in embeddings:
        score = np.dot(emb, centroid)
        scores.append(float(score))

    return scores
