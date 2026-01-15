# insurance_ai/services/nlp/qa_service.py
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import faiss
from typing import List, Dict, Tuple


class InsuranceQAService:
    """
    Question Answering service for insurance documents.
    Uses retrieval + extractive QA for accurate answers.
    """
    
    def __init__(self):
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"QA Service initializing on device: {self.device}")
        
        # Load embedding model for retrieval (very strong semantic model)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.embedding_model.to(self.device)
        
        # Load QA model (fine-tuned on SQuAD 2.0 - handles unanswerable questions)
        print("Loading QA model...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            device=0 if self.device == "cuda" else -1
        )
        
        # Storage for document chunks and embeddings
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def prepare_document(self, text: str) -> None:
        """
        Process and index the document for QA.
        Splits into semantic chunks and creates FAISS index.
        """
        print("Preparing document for QA...")
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        # Further split large paragraphs into sentences
        self.chunks = []
        for para in paragraphs:
            if len(para) < 100:  # Too short, skip
                continue
            
            if len(para) <= 600:  # Good size, keep as is
                self.chunks.append(para)
            else:  # Too long, split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= 600:
                        current_chunk += " " + sent
                    else:
                        if current_chunk:
                            self.chunks.append(current_chunk.strip())
                        current_chunk = sent
                
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
        
        # Filter out meaningless chunks
        self.chunks = [c for c in self.chunks if self._is_meaningful_chunk(c)]
        
        print(f"Created {len(self.chunks)} searchable chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Create FAISS index for fast retrieval
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print("Document ready for QA!")
    
    def _is_meaningful_chunk(self, text: str) -> bool:
        """Check if chunk contains meaningful content."""
        text = text.strip()
        
        # Too short
        if len(text) < 50:
            return False
        
        # Must have enough letters
        alpha_count = len(re.findall(r'[a-zA-Z]', text))
        if alpha_count < 30:
            return False
        
        # Skip boilerplate
        skip_patterns = [
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^table of contents',
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        return True
    
    def _retrieve_relevant_chunks(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant chunks for the question.
        Returns list of (chunk_text, similarity_score) tuples.
        """
        # Encode question
        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
        faiss.normalize_L2(question_embedding)
        
        # Search
        scores, indices = self.index.search(question_embedding, top_k)
        
        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def _extract_answer_from_context(
        self, 
        question: str, 
        context: str,
        max_answer_length: int = 200
    ) -> Dict:
        """
        Use QA model to extract answer from context.
        """
        try:
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=max_answer_length,
                handle_impossible_answer=True
            )
            return result
        except Exception as e:
            print(f"QA extraction error: {e}")
            return {"answer": "", "score": 0.0}
    
    def _calculate_confidence(self, qa_score: float, retrieval_score: float) -> str:
        """
        Calculate overall confidence level.
        """
        # Weighted combination
        combined_score = (qa_score * 0.7) + (retrieval_score * 0.3)
        
        if combined_score > 0.6:
            return "high"
        elif combined_score > 0.35:
            return "medium"
        else:
            return "low"
    
    def answer_question(
        self, 
        question: str, 
        detailed: bool = False
    ) -> Dict:
        """
        Answer a question based on the document.
        
        Args:
            question: The question to answer
            detailed: If True, return detailed answer with context
        
        Returns:
            Dictionary with answer, confidence, and context
        """
        if not self.chunks or self.index is None:
            return {
                "error": "Document not prepared. Call prepare_document() first.",
                "answer": None,
                "confidence": "none"
            }
        
        # Step 1: Retrieve relevant chunks
        relevant_chunks = self._retrieve_relevant_chunks(question, top_k=5)
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "confidence": "none",
                "source_context": None
            }
        
        best_answer = None
        best_score = 0.0
        best_chunk = None
        best_retrieval_score = 0.0
        
        # Step 2: Try to extract answer from each relevant chunk
        for chunk, retrieval_score in relevant_chunks[:3]:  # Check top 3 chunks
            qa_result = self._extract_answer_from_context(question, chunk)
            
            if qa_result["score"] > best_score:
                best_score = qa_result["score"]
                best_answer = qa_result["answer"]
                best_chunk = chunk
                best_retrieval_score = retrieval_score
        
        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(best_score, best_retrieval_score)
        
        # Step 4: Format response
        if detailed:
            # Return full context with answer highlighted
            response = {
                "answer": best_answer if best_answer else "Answer not explicitly stated in document.",
                "confidence": confidence,
                "confidence_score": round(best_score, 3),
                "source_context": best_chunk,
                "additional_relevant_sections": [
                    {
                        "text": chunk,
                        "relevance_score": round(score, 3)
                    }
                    for chunk, score in relevant_chunks[:3]
                ]
            }
        else:
            # Return short, direct answer
            response = {
                "answer": best_answer if best_answer else "Answer not found in document.",
                "confidence": confidence,
                "confidence_score": round(best_score, 3)
            }
        
        # Add explanation if confidence is low
        if confidence == "low":
            response["note"] = "Low confidence answer. The most relevant section is provided, but exact answer may not be in document."
            if not detailed:
                response["relevant_section"] = best_chunk
        
        return response
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the indexed document."""
        return {
            "total_chunks": len(self.chunks),
            "indexed": self.index is not None,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0
        }


# Global instance (lazy loaded)
_qa_service = None


def get_qa_service() -> InsuranceQAService:
    """Get or create the global QA service instance."""
    global _qa_service
    if _qa_service is None:
        _qa_service = InsuranceQAService()
    return _qa_service