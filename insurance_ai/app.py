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
from google import genai

load_dotenv()
app = Flask(__name__)

# QA service cache: stores prepared documents by hash
qa_document_cache = {}

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


@app.route("/summary", methods=["POST"])
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


if __name__ == "__main__":
    app.run(debug=True)