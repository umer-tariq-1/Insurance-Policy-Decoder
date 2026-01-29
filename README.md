# Insurance Policy Decoder

> **AI-Powered Insurance Policy Analysis Tool**
> Decode complex insurance documents with advanced NLP and local AI processing

---

## Overview

**Insurance Policy Decoder** is an intelligent document analysis system that uses cutting-edge AI and Natural Language Processing to help users understand insurance policies. Upload any insurance document and get instant summaries, ask questions in natural language, and compare multiple policies side-by-side.

### Key Features

- **AI-Powered Summarization** - Generate comprehensive or quick summaries of insurance documents using local LLMs (Ollama/Llama 3.2)
- **Intelligent Q&A System** - Ask questions about your policy in natural language and get accurate answers with source references
- **Policy Comparison** - Compare two policies side-by-side across 20+ categories with AI-powered recommendations
- **Multi-Format Support** - Process PDF, DOCX, and DOC files with advanced text extraction and OCR capabilities
- **Privacy-First** - All processing happens locally using Ollama (optional cloud-based Gemini API also available)
- **Fast & Efficient** - Optimized with semantic chunking, vector embeddings (FAISS), and caching

---

## Technology Stack

### Backend
- **Python 3.10+** with Flask framework
- **Ollama** - Local LLM inference (Llama 3.2)
- **Google Gemini API** - Optional cloud-based AI

### AI/ML Libraries
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained NLP models
- **Sentence Transformers** - Semantic text embeddings
- **FAISS** - Fast similarity search for RAG (Retrieval Augmented Generation)

### Document Processing
- **pdfplumber** - PDF text extraction
- **python-docx** - DOCX/DOC processing
- **Tesseract OCR** - Scanned document processing
- **Pillow** - Image processing

### NLP Techniques
- Semantic chunking with embeddings
- Extractive summarization (BERT-based)
- Question answering with RAG architecture
- Vector similarity search

---

## Project Structure

```
Insurance Policy Decoder/
├── insurance_ai/              # Main application directory
│   ├── app.py                # Flask server and API routes
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # Environment variables (API keys)
│   │
│   ├── extractors/           # Document extraction modules
│   │   ├── pdf_extractor.py
│   │   ├── docx_extractor.py
│   │   └── ocr.py
│   │
│   └── services/nlp/         # NLP and AI services
│       ├── chunker.py                 # Text chunking
│       ├── semantic_chunker.py        # Semantic-based chunking
│       ├── extractive_summarizer.py   # BERT extractive summary
│       ├── ollama_summarizer.py       # Ollama-based summary
│       ├── ollama_qa_service.py       # Q&A with RAG
│       ├── ollama_comparator.py       # Policy comparison
│       ├── qa_service.py              # BERT-based Q&A
│       └── ...
│
├── setup.md                  # Detailed setup guide
├── run-project.md            # How to run the project
├── api-documentation.md      # Complete API reference
└── README.md                 # This file
```

---

## Quick Start

### Prerequisites

- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Installation & Setup

For detailed installation instructions, see **[setup.md](setup.md)**

**Quick summary:**

1. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/)
2. **Install Ollama** from [ollama.ai](https://ollama.ai/download)
3. **Download AI Model:**
   ```cmd
   ollama serve
   ollama pull llama3.2:3b
   ```
4. **Install Dependencies:**
   ```cmd
   cd "Insurance Policy Decoder"
   python -m venv venv
   venv\Scripts\activate
   cd insurance_ai
   pip install -r requirements.txt
   ```
5. **Configure Environment (Optional):**
   - Create [.env](insurance_ai/.env) file with `GEMINI_API_KEY=your_key` for cloud AI features

### Running the Application

For detailed run instructions, see **[run-project.md](run-project.md)**

**Quick start:**

1. **Terminal 1 - Start Ollama:**
   ```cmd
   ollama serve
   ```

2. **Terminal 2 - Start Flask Server:**
   ```cmd
   cd "Insurance Policy Decoder"
   venv\Scripts\activate
   cd insurance_ai
   python app.py
   ```

3. **Verify:**
   - Server: http://localhost:5000/health
   - AI Status: http://localhost:5000/ollama/status

---

## API Documentation

The application provides a RESTful API for document analysis. For complete API documentation, see **[api-documentation.md](api-documentation.md)**

### Core Endpoints

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **Upload** | `POST /upload` | Upload insurance document (PDF/DOCX/DOC) |
| **Summary** | `POST /local-summary` | Generate AI summary (quick/standard/comprehensive) |
| **Q&A** | `POST /local-qa` | Ask questions about the document |
| **Compare** | `POST /compare` | Compare two policies side-by-side |
| **Status** | `GET /ollama/status` | Check AI service availability |

### Example Usage

**Upload a document:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:5000/upload', {
  method: 'POST',
  body: formData
});
const { hash } = await response.json();
```

**Generate summary:**
```javascript
const response = await fetch('http://localhost:5000/local-summary', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    hash: 'document_hash',
    mode: 'standard'  // quick | standard | comprehensive
  })
});
const { summary } = await response.json();
```

**Ask a question:**
```javascript
const response = await fetch('http://localhost:5000/local-qa', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    hash: 'document_hash',
    question: 'What is my deductible amount?',
    detailed: true
  })
});
const { answer, confidence, sources } = await response.json();
```

For more examples and detailed API specifications, see **[api-documentation.md](api-documentation.md)**

---

## Features in Detail

### 1. Document Summarization

Generate three levels of summaries:

- **Quick Mode** (~30s) - 20 categorized bullet points (COVERAGE, EXCLUSION, WARNING, etc.)
- **Standard Mode** (~1-2 min) - Structured sections covering all key aspects
- **Comprehensive Mode** (~3-5 min) - Detailed analysis with complete policy breakdown

**Example output sections:**
- Policy Overview
- Coverage Details & Benefits
- Costs & Financial Terms
- Exclusions & Limitations
- Claim Procedures
- Important Warnings

### 2. Intelligent Q&A

Ask natural language questions about your policy:

- "What is my deductible?"
- "Are pre-existing conditions covered?"
- "How do I file a claim?"

**Features:**
- Confidence levels (high/medium/low)
- Source references from the document
- Suggested questions based on document content
- Uses RAG (Retrieval Augmented Generation) for accurate answers

### 3. Policy Comparison

Compare two policies across 20+ categories:

- Premium Amount, Deductible, Co-payment
- Coverage Limits, Exclusions, Waiting Periods
- Claim Process, Cancellation Terms, Special Benefits
- AI-powered verdict and recommendations
- Highlighted key differences

### 4. Multi-Format Document Processing

- **PDF**: Direct text extraction with pdfplumber
- **DOCX/DOC**: Native Word document parsing
- **Scanned PDFs**: OCR with Tesseract
- **Smart Chunking**: Semantic-based text chunking for better context

---

## Documentation Files

This project includes comprehensive documentation:

| File | Purpose | When to Use |
|------|---------|-------------|
| **[setup.md](setup.md)** | Complete setup guide | First-time installation |
| **[run-project.md](run-project.md)** | How to run the project | Daily startup |
| **[api-documentation.md](api-documentation.md)** | API reference | Frontend integration |
| **[README.md](README.md)** | Project overview | Understanding the project |

---

## Configuration

### Ollama Configuration

Change AI model or adjust settings:

```javascript
POST /ollama/configure
{
  "model": "llama3.2:3b",
  "temperature": 0.3
}
```

**Recommended models:**
- `llama3.2:3b` - Best for 4GB VRAM (default)
- `phi3:mini` - Fast, good quality
- `mistral:7b` - Better quality (requires 8GB+ VRAM)

### Environment Variables

Create [insurance_ai/.env](insurance_ai/.env):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get Gemini API key from: https://aistudio.google.com/apikey

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Ollama not running" | Run `ollama serve` in separate terminal |
| "Model not found" | Download model: `ollama pull llama3.2:3b` |
| "Python not recognized" | Add Python to PATH (reinstall with checkbox) |
| Port 5000 in use | Use different port: `python app.py --port 5001` |

For more troubleshooting tips, see **[setup.md](setup.md)**

---

## System Requirements

### Minimum
- Windows 10/11 (64-bit)
- 8GB RAM
- 4GB VRAM (for GPU acceleration) or CPU
- 10GB free disk space

### Recommended
- 16GB RAM
- 6GB+ VRAM (NVIDIA GPU)
- SSD storage

---

## Performance

| Operation | Time (approx.) | Notes |
|-----------|---------------|-------|
| Document upload | 1-3 seconds | Depends on file size |
| Quick summary | 30-60 seconds | 20 bullet points |
| Standard summary | 1-2 minutes | Full structured summary |
| First Q&A | 30-60 seconds | Includes document preparation |
| Follow-up Q&A | 10-30 seconds | Uses cached embeddings |
| Policy comparison | 2-4 minutes | Full 20-category comparison |

*Times based on Llama 3.2 3B model with 4GB VRAM*

---

## Technical Highlights

- **RAG Architecture** - Retrieval Augmented Generation for accurate Q&A
- **Semantic Search** - FAISS vector database for fast similarity search
- **Smart Caching** - Document embeddings cached for faster responses
- **Hybrid NLP** - Combines extractive and generative AI techniques
- **Confidence Scoring** - Provides answer reliability metrics
- **Source Attribution** - Links answers back to source text

---

## License

This project is provided as-is for educational and personal use.

---

## Support

For issues, questions, or feature requests:

1. Check the documentation files ([setup.md](setup.md), [run-project.md](run-project.md), [api-documentation.md](api-documentation.md))
2. Review the troubleshooting sections
3. Ensure Ollama is running and model is downloaded
4. Verify all dependencies are installed correctly

---

## Credits

**AI Models:**
- Llama 3.2 (Meta AI)
- Google Gemini
- RoBERTa (Hugging Face)

**Libraries:**
- PyTorch, Transformers, Sentence-Transformers
- FAISS (Facebook AI)
- Flask, pdfplumber, python-docx

---

**Built with advanced AI and NLP techniques for practical insurance policy analysis**

*Last updated: January 2026*
