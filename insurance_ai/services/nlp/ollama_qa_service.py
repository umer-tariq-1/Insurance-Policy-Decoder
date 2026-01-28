# insurance_ai/services/nlp/ollama_qa_service.py
"""
RAG-based Question Answering service using local LLM (Ollama).

Uses retrieval (FAISS + embeddings) to find relevant document sections,
then generates comprehensive answers using Ollama.

This provides better answers than extractive QA because:
1. Can synthesize information from multiple sections
2. Can explain and elaborate on the answer
3. Can handle questions that require reasoning
"""

import re
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try to import ML dependencies
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML dependencies not available. Install torch, sentence-transformers, faiss-cpu")


@dataclass
class QAConfig:
    """Configuration for the QA service."""
    model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.2  # Lower for factual answers
    num_ctx: int = 8192
    num_predict: int = 1024  # Reasonable answer length
    top_k_chunks: int = 5  # Number of chunks to retrieve


# Global config
_qa_config = QAConfig()


def configure_qa(
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    temperature: Optional[float] = None
):
    """Configure the QA service settings."""
    global _qa_config
    if model:
        _qa_config.model = model
    if ollama_url:
        _qa_config.ollama_url = ollama_url
    if temperature is not None:
        _qa_config.temperature = temperature


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Make a request to Ollama API."""
    payload = {
        "model": _qa_config.model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": _qa_config.temperature,
            "num_ctx": _qa_config.num_ctx,
            "num_predict": _qa_config.num_predict,
        }
    }

    response = requests.post(
        f"{_qa_config.ollama_url}/api/generate",
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    return response.json().get("response", "")


def check_ollama_available() -> Tuple[bool, str]:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{_qa_config.ollama_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server not responding"

        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        target_base = _qa_config.model.split(":")[0]
        if not any(target_base in name for name in model_names):
            return False, f"Model '{_qa_config.model}' not found. Run: ollama pull {_qa_config.model}"

        return True, "OK"
    except requests.exceptions.ConnectionError:
        return False, "Ollama not running. Start with: ollama serve"
    except Exception as e:
        return False, f"Error: {str(e)}"


# System prompt for insurance Q&A
QA_SYSTEM_PROMPT = """You are an expert insurance policy analyst answering questions about insurance documents.

CRITICAL RULES:
1. ONLY answer based on the provided context from the document
2. If the answer is not in the context, say "This information is not found in the provided document sections"
3. Be specific - include exact numbers, dates, amounts, and conditions from the context
4. If there are multiple relevant pieces of information, include all of them
5. Explain insurance terms in plain language when they appear in your answer
6. If the context is ambiguous or incomplete, acknowledge the uncertainty
7. NEVER make up information that's not in the context"""


QA_PROMPT_TEMPLATE = """Based on the following sections from an insurance policy document, answer the user's question.

RELEVANT DOCUMENT SECTIONS:
{context}

---

USER QUESTION: {question}

Provide a clear, comprehensive answer based ONLY on the information in the document sections above. Include specific details like numbers, dates, conditions, and requirements. If the answer involves multiple points, organize them clearly.

If the information to answer this question is not in the provided sections, clearly state that the information was not found in the document.

ANSWER:"""


class OllamaQAService:
    """
    RAG-based Question Answering service using Ollama.
    """

    def __init__(self):
        if not ML_AVAILABLE:
            raise RuntimeError("ML dependencies not available. Install torch, sentence-transformers, faiss-cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Ollama QA Service initializing on device: {self.device}")

        # Load embedding model for retrieval
        print("Loading embedding model for retrieval...")
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.embedding_model.to(self.device)

        # Storage
        self.chunks = []
        self.chunk_metadata = []  # Store additional info about each chunk
        self.embeddings = None
        self.index = None
        self.document_prepared = False

    def prepare_document(self, text: str) -> Dict:
        """
        Process and index the document for Q&A.
        """
        print("Preparing document for Q&A...")

        # Split into chunks with overlap for better context
        self.chunks = []
        self.chunk_metadata = []

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunk_id = 0
        for para_idx, para in enumerate(paragraphs):
            if len(para) < 50:  # Too short
                continue

            # For long paragraphs, split into smaller chunks with overlap
            if len(para) > 800:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""

                for sent in sentences:
                    if len(current_chunk) + len(sent) <= 800:
                        current_chunk += " " + sent if current_chunk else sent
                    else:
                        if current_chunk and self._is_meaningful(current_chunk):
                            self.chunks.append(current_chunk.strip())
                            self.chunk_metadata.append({
                                "id": chunk_id,
                                "paragraph": para_idx,
                                "type": "split"
                            })
                            chunk_id += 1
                        current_chunk = sent

                if current_chunk and self._is_meaningful(current_chunk):
                    self.chunks.append(current_chunk.strip())
                    self.chunk_metadata.append({
                        "id": chunk_id,
                        "paragraph": para_idx,
                        "type": "split"
                    })
                    chunk_id += 1
            else:
                if self._is_meaningful(para):
                    self.chunks.append(para)
                    self.chunk_metadata.append({
                        "id": chunk_id,
                        "paragraph": para_idx,
                        "type": "full"
                    })
                    chunk_id += 1

        print(f"Created {len(self.chunks)} searchable chunks")

        if not self.chunks:
            return {"error": "No meaningful content found in document", "chunks": 0}

        # Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Create FAISS index
        print("Building search index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        self.document_prepared = True
        print("Document ready for Q&A!")

        return {
            "status": "ready",
            "chunks": len(self.chunks),
            "embedding_dimension": dimension
        }

    def _is_meaningful(self, text: str) -> bool:
        """Check if chunk contains meaningful content."""
        text = text.strip()
        if len(text) < 50:
            return False

        alpha_count = len(re.findall(r'[a-zA-Z]', text))
        if alpha_count < 30:
            return False

        # Skip boilerplate
        skip_patterns = [
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^table of contents',
            r'^index$',
        ]

        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False

        return True

    def _retrieve_relevant_chunks(
        self,
        question: str,
        top_k: int = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve most relevant chunks for the question.
        Returns list of (chunk_text, similarity_score, metadata) tuples.
        """
        if top_k is None:
            top_k = _qa_config.top_k_chunks

        # Encode question
        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
        faiss.normalize_L2(question_embedding)

        # Search
        scores, indices = self.index.search(question_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks) and score > 0.2:  # Minimum relevance threshold
                results.append((
                    self.chunks[idx],
                    float(score),
                    self.chunk_metadata[idx]
                ))

        return results

    def answer_question(
        self,
        question: str,
        detailed: bool = False,
        include_sources: bool = True
    ) -> Dict:
        """
        Answer a question about the document using RAG.

        Args:
            question: The question to answer
            detailed: If True, include more context and explanation
            include_sources: If True, include the source chunks used

        Returns:
            Dictionary with answer, confidence, and optionally sources
        """
        if not self.document_prepared:
            return {
                "error": "Document not prepared. Call prepare_document() first.",
                "answer": None
            }

        # Check Ollama availability
        available, message = check_ollama_available()
        if not available:
            return {
                "error": f"Ollama not available: {message}",
                "answer": None
            }

        # Step 1: Retrieve relevant chunks
        top_k = 7 if detailed else 5
        relevant_chunks = self._retrieve_relevant_chunks(question, top_k=top_k)

        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the document for this question.",
                "confidence": "none",
                "sources": []
            }

        # Step 2: Build context from retrieved chunks
        context_parts = []
        for i, (chunk, score, meta) in enumerate(relevant_chunks):
            context_parts.append(f"[Section {i+1}] (Relevance: {score:.2f})\n{chunk}")

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Generate answer using Ollama
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        try:
            answer = _call_ollama(prompt, QA_SYSTEM_PROMPT)
        except Exception as e:
            return {
                "error": f"Failed to generate answer: {str(e)}",
                "answer": None
            }

        # Step 4: Calculate confidence based on retrieval scores
        avg_relevance = np.mean([score for _, score, _ in relevant_chunks])
        max_relevance = max([score for _, score, _ in relevant_chunks])

        if max_relevance > 0.6 and avg_relevance > 0.4:
            confidence = "high"
        elif max_relevance > 0.4 and avg_relevance > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        # Step 5: Build response
        response = {
            "answer": answer.strip(),
            "confidence": confidence,
            "relevance_score": round(max_relevance, 3),
            "model": _qa_config.model
        }

        if include_sources:
            response["sources"] = [
                {
                    "text": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "relevance": round(score, 3)
                }
                for chunk, score, meta in relevant_chunks[:3]
            ]

        if confidence == "low":
            response["note"] = "Low confidence - the question may not be directly addressed in the document, or the relevant section wasn't found."

        return response

    def get_suggested_questions(self) -> List[str]:
        """
        Generate suggested questions based on document content.
        """
        if not self.document_prepared or not self.chunks:
            return []

        # Sample some chunks and generate questions
        sample_chunks = self.chunks[:10] if len(self.chunks) > 10 else self.chunks
        context = "\n\n".join(sample_chunks[:5])

        prompt = f"""Based on this insurance policy content, suggest 5-7 important questions a policyholder might want to ask.

POLICY CONTENT:
{context}

Generate practical questions about coverage, costs, exclusions, claims, and important conditions. Format as a numbered list:"""

        try:
            result = _call_ollama(prompt, QA_SYSTEM_PROMPT)

            # Parse questions from result
            questions = []
            for line in result.strip().split("\n"):
                line = line.strip()
                # Remove numbering and clean up
                cleaned = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
                if cleaned and len(cleaned) > 10 and "?" in cleaned:
                    questions.append(cleaned)

            return questions[:7]
        except:
            return [
                "What is covered under this policy?",
                "What are the exclusions?",
                "What is the deductible amount?",
                "How do I file a claim?",
                "What is the coverage limit?"
            ]

    def get_stats(self) -> Dict:
        """Get statistics about the indexed document."""
        return {
            "document_prepared": self.document_prepared,
            "total_chunks": len(self.chunks),
            "indexed": self.index is not None,
            "model": _qa_config.model
        }


# Global instance (lazy loaded)
_ollama_qa_service = None


def get_ollama_qa_service() -> OllamaQAService:
    """Get or create the global Ollama QA service instance."""
    global _ollama_qa_service
    if _ollama_qa_service is None:
        _ollama_qa_service = OllamaQAService()
    return _ollama_qa_service


def reset_qa_service():
    """Reset the QA service (useful when switching documents)."""
    global _ollama_qa_service
    _ollama_qa_service = None
