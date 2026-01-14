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
from google import genai

load_dotenv()
app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
