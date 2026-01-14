# insurance_ai/services/nlp/extractive_summarizer.py
from sentence_transformers import SentenceTransformer
import numpy as np
import re

model = SentenceTransformer("all-mpnet-base-v2")

def clean_sentence(sentence: str) -> str:
    """Clean up a sentence for display."""
    sentence = sentence.strip()
    # Remove excessive whitespace
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def is_meaningful_sentence(sentence: str) -> bool:
    """Check if a sentence is meaningful content."""
    sentence = sentence.strip()
    
    # Too short
    if len(sentence) < 40:
        return False
    
    # Too long (likely merged sentences)
    if len(sentence) > 500:
        return False
    
    # Must have enough letters
    alpha_count = len(re.findall(r'[a-zA-Z]', sentence))
    if alpha_count < 20:
        return False
    
    # Skip common boilerplate patterns
    skip_patterns = [
        r'^page\s+\d+',
        r'^\d+\s*$',
        r'^section\s+\d+\s*$',
        r'^article\s+\d+\s*$',
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, sentence.lower()):
            return False
    
    return True

def extract_key_information(text: str, num_sentences: int = 15) -> list:
    """
    Extract key sentences from insurance policy using extractive summarization.
    This preserves the original wording which is important for insurance docs.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = clean_sentence(sent)
        if is_meaningful_sentence(sent):
            cleaned_sentences.append(sent)
    
    if len(cleaned_sentences) == 0:
        return []
    
    # If we have fewer sentences than requested, return all
    if len(cleaned_sentences) <= num_sentences:
        return cleaned_sentences
    
    # Compute embeddings
    embeddings = model.encode(cleaned_sentences)
    
    # Calculate importance scores based on:
    # 1. Similarity to document centroid (central themes)
    # 2. Keyword presence (insurance-specific terms)
    
    centroid = np.mean(embeddings, axis=0)
    centroid_scores = embeddings @ centroid
    
    # Keyword scoring
    important_keywords = [
        'coverage', 'covered', 'benefit', 'premium', 'deductible',
        'exclusion', 'limit', 'claim', 'policy', 'insured',
        'payment', 'termination', 'cancellation', 'renewal',
        'co-pay', 'copay', 'coinsurance', 'out-of-pocket',
        'network', 'provider', 'emergency', 'pre-existing',
        'waiting period', 'effective date', 'grace period'
    ]
    
    keyword_scores = np.zeros(len(cleaned_sentences))
    for i, sent in enumerate(cleaned_sentences):
        sent_lower = sent.lower()
        for keyword in important_keywords:
            if keyword in sent_lower:
                keyword_scores[i] += 1
    
    # Normalize keyword scores
    if keyword_scores.max() > 0:
        keyword_scores = keyword_scores / keyword_scores.max()
    
    # Combine scores (70% semantic, 30% keywords)
    final_scores = 0.7 * centroid_scores + 0.3 * keyword_scores
    
    # Get top sentences
    top_indices = np.argsort(final_scores)[-num_sentences:][::-1]
    
    # Sort by original order to maintain document flow
    top_indices = sorted(top_indices)
    
    key_sentences = [cleaned_sentences[i] for i in top_indices]
    
    return key_sentences