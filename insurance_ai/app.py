# insurance_ai/app.py
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

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

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        return jsonify({"error": "Only PDF and DOCX allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Extract text
    ext = filename.rsplit(".", 1)[1].lower()

    if ext == "pdf":
        text = extract_text_from_pdf(save_path)
    else:
        text = extract_text_from_docx(save_path)

    
    chunks = chunk_text(text)
    chunks = [chunk.replace('"', "'") for chunk in chunks]
    chunks_string = "\n\n".join(chunks)
    

    return jsonify({
        "filename": filename,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "chunks_string": chunks_string
    })

# @app.route("/summary", methods=["POST"])
# def generate_summary():
    # if "file" not in request.files:
    #     return jsonify({"error": "File required"}), 400

    # file = request.files["file"]
    # filename = secure_filename(file.filename)
    # path = os.path.join(UPLOAD_FOLDER, filename)
    # file.save(path)

    # # Extract text
    # ext = filename.rsplit(".", 1)[1].lower()
    # if ext == "pdf":
    #     raw_text = extract_text_from_pdf(path)
    # else:
    #     raw_text = extract_text_from_docx(path)

    # # Normalize & chunk
    # normalized = normalize_text(raw_text)
    # chunks = semantic_chunk_text(normalized)

    # # Summarize each chunk
    # filtered_chunks = [c for c in chunks if is_informative_chunk(c)]

    # summaries = [summarize_chunk(c) for c in filtered_chunks]       

    # # Importance scoring
    # scores = score_importance(summaries)

    # # Sort by importance
    # results = sorted(
    #     zip(summaries, scores),
    #     key=lambda x: x[1],
    #     reverse=True
    # )

    # return jsonify({
    #     "total_chunks": len(chunks),
    #     "important_points": [
    #         {"summary": s, "importance": sc}
    #         for s, sc in results[:8]  # top points
    #     ]
    # })

@app.route("/summary", methods=["POST"])
def generate_summary():
    if "file" not in request.files:
        return jsonify({"error": "File required"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Extract text
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        raw_text = extract_text_from_pdf(path)
    else:
        raw_text = extract_text_from_docx(path)

    # Normalize & semantic chunking
    normalized = normalize_text(raw_text)
    chunks = semantic_chunk_text(normalized)

    final_points = []

    for chunk in chunks:
        sentences = split_into_sentences(chunk)
        if len(sentences) < 3:
            continue

        key_sentences = rank_sentences(sentences, top_k=4)
        combined = " ".join(key_sentences)

        rewritten = rewrite_to_plain_language(combined)
        final_points.append(rewritten)

    return jsonify({
        "total_chunks": len(chunks),
        "important_points": final_points[:10]  # top results
    })

if __name__ == "__main__":
    app.run(debug=True)
