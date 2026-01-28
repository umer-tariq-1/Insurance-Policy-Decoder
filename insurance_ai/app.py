# insurance_ai/app.py
import os
import hashlib


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from extractors.pdf_extractor import extract_text_from_pdf
from extractors.docx_extractor import extract_text_from_docx

from services.nlp.chunker import chunk_text

from services.nlp.text_normalizer import normalize_text
from services.nlp.semantic_chunker import semantic_chunk_text
from services.nlp.summarizer import summarize_chunk
from services.nlp.importance import score_importance
from services.nlp.chunk_filter import is_informative_chunk

from services.nlp.sentence_utils import split_into_sentences
from services.nlp.sentence_ranker import rank_sentences
from services.nlp.summarizer_bart import rewrite_to_plain_language
from services.nlp.extractive_summarizer import extract_key_information
from services.nlp.qa_service import get_qa_service
from services.nlp.ollama_summarizer import (
    summarize_insurance_policy,
    generate_quick_summary,
    check_ollama_available,
    configure_summarizer
)
from services.nlp.ollama_qa_service import (
    get_ollama_qa_service,
    reset_qa_service,
    check_ollama_available as check_qa_ollama,
    configure_qa
)
from services.nlp.ollama_comparator import (
    compare_policies,
    quick_compare,
    check_ollama_available as check_compare_ollama
)
from google import genai

load_dotenv()
app = Flask(__name__)

# QA service cache: stores prepared documents by hash
qa_document_cache = {}

# Ollama QA cache: stores which documents have been prepared for Ollama QA
ollama_qa_cache = {}

UPLOAD_FOLDER = "../uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx","doc"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Flask server running"})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF, DOCX or DOC files allowed"}), 400

    # Read file content to calculate hash
    file_content = file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    # Get original filename and size
    original_filename = secure_filename(file.filename)
    file_size = len(file_content)
    
    # Save file with hash as filename, keeping the extension
    ext = original_filename.rsplit(".", 1)[-1].lower()
    hash_filename = f"{file_hash}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], hash_filename)
    
    # Write the file content
    with open(save_path, 'wb') as f:
        f.write(file_content)

    return jsonify({
        "hash": file_hash,
        "filename": original_filename,
        "size": file_size
    })

@app.route("/content", methods=["POST"])
def get_file_content():
    data = request.get_json()
    
    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400
    
    file_hash = data["hash"]
    
    # Try to find file with either pdf or docx extension
    file_path = None
    ext = None
    
    for extension in ["pdf", "docx","doc"]:
        potential_path = os.path.join(UPLOAD_FOLDER, f"{file_hash}.{extension}")
        if os.path.exists(potential_path):
            file_path = potential_path
            ext = extension
            break
    
    if not file_path:
        return jsonify({"error": "File not found"}), 404
    
    # Extract text based on extension
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_docx(file_path)
    
    # Chunk the text
    chunks = chunk_text(text)
    chunks = [chunk.replace('"', "'") for chunk in chunks]
    chunks_string = "\n\n".join(chunks)
    
    return jsonify({
        "total_chunks": len(chunks),
        "chunks_string": chunks_string
    })

@app.route("/scratch-summary", methods=["POST"])
def generate_scratch_summary():
    data = request.get_json()
    
    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400
    
    file_hash = data["hash"]
    
    # Try to find file with pdf, docx, or doc extension
    file_path = None
    ext = None
    
    for extension in ["pdf", "docx", "doc"]:
        potential_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_hash}.{extension}")
        if os.path.exists(potential_path):
            file_path = potential_path
            ext = extension
            break
    
    if not file_path:
        return jsonify({"error": "File not found"}), 404

    try:
        # Extract text
        if ext == "pdf":
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = extract_text_from_docx(file_path)

        # Normalize text
        normalized = normalize_text(raw_text)
        
        # Extract key sentences (this preserves original wording)
        key_points = extract_key_information(normalized, num_sentences=25)
        
        return jsonify({
            "total_text_length": len(raw_text),
            "important_points": key_points
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500


@app.route("/local-summary", methods=["POST"])
def generate_local_summary():
    """
    Generate insurance policy summary using local LLM (Ollama).

    Request body:
    {
        "hash": "document_hash",
        "mode": "standard"  // optional: "quick", "standard", or "comprehensive"
    }

    Requires Ollama running locally with a model installed.
    Recommended: ollama pull llama3.2:3b
    """
    data = request.get_json()

    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400

    file_hash = data["hash"]
    mode = data.get("mode", "standard")

    if mode not in ["quick", "standard", "comprehensive"]:
        return jsonify({"error": "Mode must be 'quick', 'standard', or 'comprehensive'"}), 400

    # Check Ollama availability first
    available, ollama_message = check_ollama_available()
    if not available:
        return jsonify({
            "error": f"Local LLM not available: {ollama_message}",
            "setup_instructions": {
                "1": "Install Ollama from https://ollama.ai",
                "2": "Start Ollama: ollama serve",
                "3": "Pull a model: ollama pull llama3.2:3b"
            }
        }), 503

    # Find the file
    file_path = None
    ext = None

    for extension in ["pdf", "docx", "doc"]:
        potential_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_hash}.{extension}")
        if os.path.exists(potential_path):
            file_path = potential_path
            ext = extension
            break

    if not file_path:
        return jsonify({"error": "File not found"}), 404

    try:
        # Extract text
        if ext == "pdf":
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = extract_text_from_docx(file_path)

        # Normalize text
        normalized = normalize_text(raw_text)

        # Generate summary using local LLM
        result = summarize_insurance_policy(normalized, mode=mode)

        return jsonify({
            "hash": file_hash,
            "mode": mode,
            **result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500


@app.route("/ollama/status", methods=["GET"])
def ollama_status():
    """Check if Ollama is running and which models are available."""
    available, message = check_ollama_available()

    status = {
        "available": available,
        "message": message
    }

    if not available:
        status["setup_instructions"] = {
            "1": "Install Ollama from https://ollama.ai",
            "2": "Start Ollama: ollama serve",
            "3": "Pull a model: ollama pull llama3.2:3b (recommended for 4GB VRAM)",
            "alternative_models": [
                "phi3:mini (fast, good quality)",
                "qwen2.5:3b (good for structured output)",
                "mistral:7b (if you have 8GB+ VRAM)"
            ]
        }

    return jsonify(status), 200 if available else 503


@app.route("/ollama/configure", methods=["POST"])
def configure_ollama():
    """
    Configure Ollama settings.

    Request body:
    {
        "model": "llama3.2:3b",  // optional
        "ollama_url": "http://localhost:11434",  // optional
        "temperature": 0.3  // optional, 0.0-1.0
    }
    """
    data = request.get_json() or {}

    try:
        configure_summarizer(
            model=data.get("model"),
            ollama_url=data.get("ollama_url"),
            temperature=data.get("temperature")
        )

        return jsonify({
            "message": "Configuration updated",
            "current_settings": {
                "model": data.get("model", "unchanged"),
                "ollama_url": data.get("ollama_url", "unchanged"),
                "temperature": data.get("temperature", "unchanged")
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to configure: {str(e)}"}), 400


@app.route("/gemini-api-summary", methods=["POST"])
def generate_summary():
    data = request.get_json()
    
    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400
    
    file_hash = data["hash"]
    
    # Try to find file with pdf, docx, or doc extension
    file_path = None
    
    for extension in ["pdf", "docx", "doc"]:
        potential_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_hash}.{extension}")
        if os.path.exists(potential_path):
            file_path = potential_path
            break
    
    if not file_path:
        return jsonify({"error": "File not found"}), 404

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Upload the file using file= parameter (simple approach)
        uploaded_file = client.files.upload(file=file_path)
        
        # Create prompt for insurance policy summary
        prompt = """You are an insurance policy analyzer. Analyze this insurance policy document and provide a comprehensive summary that highlights all the critical information a policyholder MUST know.

            Structure your response as follows:

            **Policy Overview:**
            - Policy type and coverage summary
            - Key benefits and what's covered

            **Important Coverage Details:**
            - Coverage limits and amounts
            - Deductibles and co-payments
            - Exclusions (what's NOT covered)

            **Policyholder Responsibilities:**
            - Premium payment details
            - Claim filing procedures
            - Important deadlines or waiting periods

            **Critical Terms & Conditions:**
            - Renewal and cancellation policies
            - Any penalties or fees
            - Grace periods

            **Key Dates & Deadlines:**
            - Policy effective dates
            - Important milestones or review dates

            **Red Flags & Important Warnings:**
            - Any clauses that could result in claim denial
            - Limitations or restrictions to be aware of

            Extract and present ONLY the information that actually exists in this document. Be specific with numbers, dates, and amounts. Make it easy to understand for someone who doesn't want to read the entire policy."""

        # Generate summary with file
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Could you summarize this insurance policy?", uploaded_file, prompt]
        )
        
        # Delete the uploaded file (cleanup)
        client.files.delete(name=uploaded_file.name)
        
        return jsonify({
            "hash": file_hash,
            "summary": response.text
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

@app.route("/scratch-qa", methods=["POST"])
def question_answer():
    """
    Answer questions about an insurance document.
    
    Request body:
    {
        "hash": "document_hash",
        "question": "What is my deductible?",
        "detailed": false  // optional, defaults to false
    }
    """
    data = request.get_json()
    
    # Validate request
    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400
    
    if "question" not in data or not data["question"].strip():
        return jsonify({"error": "Question required in request body"}), 400
    
    file_hash = data["hash"]
    question = data["question"].strip()
    detailed = data.get("detailed", False)
    
    # Try to find file
    file_path = None
    ext = None
    
    for extension in ["pdf", "docx", "doc"]:
        potential_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_hash}.{extension}")
        if os.path.exists(potential_path):
            file_path = potential_path
            ext = extension
            break
    
    if not file_path:
        return jsonify({"error": "File not found"}), 404
    
    try:
        # Get QA service
        qa_service = get_qa_service()
        
        # Check if document is already prepared in cache
        if file_hash not in qa_document_cache:
            print(f"Preparing document {file_hash} for QA...")
            
            # Extract text
            if ext == "pdf":
                raw_text = extract_text_from_pdf(file_path)
            else:
                raw_text = extract_text_from_docx(file_path)
            
            # Normalize text
            normalized_text = normalize_text(raw_text)
            
            # Prepare document for QA
            qa_service.prepare_document(normalized_text)
            
            # Cache it
            qa_document_cache[file_hash] = True
            print(f"Document {file_hash} prepared and cached")
        
        # Answer the question
        result = qa_service.answer_question(question, detailed=detailed)
        
        # Add metadata
        result["hash"] = file_hash
        result["question"] = question
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to answer question: {str(e)}"}), 500

@app.route("/qa/clear-cache", methods=["POST"])
def clear_qa_cache():
    """
    Clear the QA document cache.
    Useful when you want to force re-processing of documents.
    
    Request body (optional):
    {
        "hash": "specific_document_hash"  // if provided, only clear this document
    }
    """
    data = request.get_json() or {}
    
    if "hash" in data:
        # Clear specific document
        file_hash = data["hash"]
        if file_hash in qa_document_cache:
            del qa_document_cache[file_hash]
            return jsonify({
                "message": f"Cleared cache for document {file_hash}",
                "remaining_cached": len(qa_document_cache)
            })
        else:
            return jsonify({
                "message": f"Document {file_hash} not in cache",
                "remaining_cached": len(qa_document_cache)
            })
    else:
        # Clear all cache
        count = len(qa_document_cache)
        qa_document_cache.clear()
        return jsonify({
            "message": f"Cleared all QA cache ({count} documents)",
            "remaining_cached": 0
        })


@app.route("/local-qa", methods=["POST"])
def local_question_answer():
    """
    Answer questions about an insurance document using local LLM (Ollama).

    Uses RAG (Retrieval Augmented Generation):
    1. Retrieves relevant document sections using semantic search
    2. Generates comprehensive answers using Ollama

    Request body:
    {
        "hash": "document_hash",
        "question": "What is my deductible?",
        "detailed": false  // optional, includes source sections if true
    }

    Requires Ollama running locally with a model installed.
    """
    data = request.get_json()

    # Validate request
    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400

    if "question" not in data or not data["question"].strip():
        return jsonify({"error": "Question required in request body"}), 400

    file_hash = data["hash"]
    question = data["question"].strip()
    detailed = data.get("detailed", False)

    # Check Ollama availability
    available, ollama_message = check_qa_ollama()
    if not available:
        return jsonify({
            "error": f"Local LLM not available: {ollama_message}",
            "setup_instructions": {
                "1": "Install Ollama from https://ollama.ai",
                "2": "Start Ollama: ollama serve",
                "3": "Pull a model: ollama pull llama3.2:3b"
            }
        }), 503

    # Find the file
    file_path = None
    ext = None

    for extension in ["pdf", "docx", "doc"]:
        potential_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_hash}.{extension}")
        if os.path.exists(potential_path):
            file_path = potential_path
            ext = extension
            break

    if not file_path:
        return jsonify({"error": "File not found"}), 404

    try:
        # Get Ollama QA service
        qa_service = get_ollama_qa_service()

        # Check if document needs to be prepared
        if file_hash not in ollama_qa_cache:
            print(f"Preparing document {file_hash} for Ollama QA...")

            # Extract text
            if ext == "pdf":
                raw_text = extract_text_from_pdf(file_path)
            else:
                raw_text = extract_text_from_docx(file_path)

            # Normalize text
            normalized_text = normalize_text(raw_text)

            # Prepare document
            prep_result = qa_service.prepare_document(normalized_text)

            if "error" in prep_result:
                return jsonify({"error": prep_result["error"]}), 400

            # Cache it
            ollama_qa_cache[file_hash] = True
            print(f"Document {file_hash} prepared for Ollama QA ({prep_result['chunks']} chunks)")

        # Answer the question
        result = qa_service.answer_question(
            question,
            detailed=detailed,
            include_sources=detailed
        )

        # Add metadata
        result["hash"] = file_hash
        result["question"] = question

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to answer question: {str(e)}"}), 500


@app.route("/local-qa/suggestions", methods=["POST"])
def get_qa_suggestions():
    """
    Get suggested questions for a document.

    Request body:
    {
        "hash": "document_hash"
    }
    """
    data = request.get_json()

    if not data or "hash" not in data:
        return jsonify({"error": "Hash required in request body"}), 400

    file_hash = data["hash"]

    # Check if document is prepared
    if file_hash not in ollama_qa_cache:
        return jsonify({
            "error": "Document not prepared for Q&A. Send a question first to prepare it.",
            "suggestions": []
        }), 400

    try:
        qa_service = get_ollama_qa_service()
        suggestions = qa_service.get_suggested_questions()

        return jsonify({
            "hash": file_hash,
            "suggestions": suggestions
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate suggestions: {str(e)}"}), 500


@app.route("/local-qa/clear-cache", methods=["POST"])
def clear_ollama_qa_cache():
    """
    Clear the Ollama QA document cache.

    Request body (optional):
    {
        "hash": "specific_document_hash"  // if provided, only clear this document
    }
    """
    data = request.get_json() or {}

    if "hash" in data:
        file_hash = data["hash"]
        if file_hash in ollama_qa_cache:
            del ollama_qa_cache[file_hash]
            # Also reset the service to clear document data
            reset_qa_service()
            return jsonify({
                "message": f"Cleared Ollama QA cache for document {file_hash}",
                "remaining_cached": len(ollama_qa_cache)
            })
        else:
            return jsonify({
                "message": f"Document {file_hash} not in Ollama QA cache",
                "remaining_cached": len(ollama_qa_cache)
            })
    else:
        count = len(ollama_qa_cache)
        ollama_qa_cache.clear()
        reset_qa_service()
        return jsonify({
            "message": f"Cleared all Ollama QA cache ({count} documents)",
            "remaining_cached": 0
        })


@app.route("/compare", methods=["POST"])
def compare_documents():
    """
    Compare two insurance policy documents side-by-side.

    Request body:
    {
        "hash1": "first_document_hash",
        "hash2": "second_document_hash",
        "quick": false,  // optional: true for faster but less detailed comparison
        "include_verdict": true  // optional: include AI recommendation
    }

    Returns comparison data structured for table display:
    {
        "categories": ["Policy Type", "Premium", ...],
        "policy1": {
            "hash": "hash1",
            "values": ["Health Insurance", "$500/month", ...]
        },
        "policy2": {
            "hash": "hash2",
            "values": ["Health Insurance", "$450/month", ...]
        },
        "highlights": [...],  // Key differences
        "verdict": "AI analysis..."  // Optional recommendation
    }
    """
    data = request.get_json()

    # Validate request
    if not data:
        return jsonify({"error": "Request body required"}), 400

    if "hash1" not in data or "hash2" not in data:
        return jsonify({"error": "Both hash1 and hash2 are required"}), 400

    hash1 = data["hash1"]
    hash2 = data["hash2"]
    quick_mode = data.get("quick", False)
    include_verdict = data.get("include_verdict", True)

    if hash1 == hash2:
        return jsonify({"error": "Cannot compare a document with itself"}), 400

    # Check Ollama availability
    available, ollama_message = check_compare_ollama()
    if not available:
        return jsonify({
            "error": f"Local LLM not available: {ollama_message}",
            "setup_instructions": {
                "1": "Install Ollama from https://ollama.ai",
                "2": "Start Ollama: ollama serve",
                "3": "Pull a model: ollama pull llama3.2:3b"
            }
        }), 503

    # Find both files
    def find_file(file_hash):
        for extension in ["pdf", "docx", "doc"]:
            potential_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_hash}.{extension}")
            if os.path.exists(potential_path):
                return potential_path, extension
        return None, None

    file_path1, ext1 = find_file(hash1)
    file_path2, ext2 = find_file(hash2)

    if not file_path1:
        return jsonify({"error": f"File not found for hash1: {hash1}"}), 404

    if not file_path2:
        return jsonify({"error": f"File not found for hash2: {hash2}"}), 404

    try:
        # Extract text from both documents
        print(f"Extracting text from document 1 ({hash1})...")
        if ext1 == "pdf":
            raw_text1 = extract_text_from_pdf(file_path1)
        else:
            raw_text1 = extract_text_from_docx(file_path1)

        print(f"Extracting text from document 2 ({hash2})...")
        if ext2 == "pdf":
            raw_text2 = extract_text_from_pdf(file_path2)
        else:
            raw_text2 = extract_text_from_docx(file_path2)

        # Normalize text
        normalized1 = normalize_text(raw_text1)
        normalized2 = normalize_text(raw_text2)

        # Perform comparison
        if quick_mode:
            print("Performing quick comparison...")
            result = quick_compare(normalized1, normalized2, hash1, hash2)
        else:
            print("Performing full comparison...")
            result = compare_policies(
                normalized1,
                normalized2,
                hash1,
                hash2,
                include_verdict=include_verdict
            )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to compare documents: {str(e)}"}), 500


@app.route("/compare/quick", methods=["POST"])
def quick_compare_documents():
    """
    Quick comparison of two policies - faster but less detailed.
    Focuses on top 10 most important differences.

    Request body:
    {
        "hash1": "first_document_hash",
        "hash2": "second_document_hash"
    }
    """
    data = request.get_json()

    if not data or "hash1" not in data or "hash2" not in data:
        return jsonify({"error": "Both hash1 and hash2 are required"}), 400

    # Reuse the main compare endpoint with quick=true
    data["quick"] = True
    return compare_documents()


if __name__ == "__main__":
    app.run(debug=True)