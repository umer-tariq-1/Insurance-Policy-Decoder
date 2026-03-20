# Insurance Policy Decoder

> **AI-Powered Insurance Policy Analysis Tool**
> Decode complex insurance documents using Google Gemini AI

---

## Overview

**Insurance Policy Decoder** is an intelligent document analysis system that uses Google Gemini AI to help users understand insurance policies. Upload any insurance document and get instant summaries, ask questions in natural language, compare multiple policies side-by-side, and get a risk assessment.

### Key Features

- **AI-Powered Summarization** - Generate comprehensive summaries of insurance documents using Google Gemini
- **Intelligent Q&A System** - Ask questions about your policy in natural language and get accurate answers with source references
- **Policy Comparison** - Compare two policies side-by-side across 20 categories with AI-powered recommendations
- **Risk Scoring** - Get a transparent, rule-based risk assessment of any policy
- **Multi-Format Support** - Process PDF, DOCX, and DOC files with advanced text extraction
- **Cloud AI** - Powered by Google Gemini 2.5 Flash (no local GPU required)

---

## Technology Stack

### Backend
- **Python 3.10+** with Flask framework
- **Google Gemini API** - Primary AI engine (cloud-based, no GPU needed)
- **Ollama** - Optional local LLM fallback (Llama 3.2)

### AI/ML Libraries
- **PyTorch** - Deep learning framework (used by research/academic routes)
- **Transformers** (Hugging Face) - Pre-trained NLP models (research routes)
- **Sentence Transformers** - Semantic text embeddings
- **FAISS** - Fast similarity search

### Document Processing
- **pdfplumber** - PDF text extraction
- **python-docx** - DOCX/DOC processing
- **Tesseract OCR** - Scanned document processing
- **Pillow** - Image processing

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
│       ├── extractive_summarizer.py   # BERT extractive summary (research)
│       ├── ollama_summarizer.py       # Ollama-based summary (fallback)
│       ├── ollama_qa_service.py       # Q&A with RAG (fallback)
│       ├── ollama_comparator.py       # Policy comparison (fallback)
│       ├── qa_service.py              # BERT-based Q&A (research)
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
- 4GB RAM minimum
- Internet connection (for Gemini API calls)
- Google Gemini API key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Installation & Setup

For detailed installation instructions, see **[setup.md](setup.md)**

**Quick summary:**

1. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/)
2. **Get a Gemini API key** from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
3. **Install Dependencies:**
   ```cmd
   cd "Insurance Policy Decoder"
   python -m venv venv
   venv\Scripts\activate
   cd insurance_ai
   pip install -r requirements.txt
   ```
4. **Configure API key** in `insurance_ai/.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

For detailed run instructions, see **[run-project.md](run-project.md)**

**Quick start:**

```cmd
cd "Insurance Policy Decoder"
venv\Scripts\activate
cd insurance_ai
python app.py
```

**Verify:**
- Server: http://localhost:5000/health

---

## API Documentation

The application provides a RESTful API for document analysis. For complete API documentation, see **[api-documentation.md](api-documentation.md)**

### Core Endpoints

| Feature          | Endpoint                | Description                                         |
| ---------------- | ----------------------- | --------------------------------------------------- |
| **Upload**       | `POST /upload`          | Upload insurance document (PDF/DOCX/DOC)            |
| **Summary**      | `POST /gemini-api-summary`  | Generate AI summary using Gemini                |
| **Q&A**          | `POST /gemini-api-qa`       | Ask questions about the document                |
| **Compare**      | `POST /gemini-api-compare`  | Compare two policies side-by-side               |
| **Risk Score**   | `POST /gemini-risk-score`   | Get a risk assessment of the policy             |

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
const response = await fetch('http://localhost:5000/gemini-api-summary', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ hash: 'document_hash' })
});
const { summary } = await response.json();
```

**Ask a question:**
```javascript
const response = await fetch('http://localhost:5000/gemini-api-qa', {
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

Gemini reads the full document and returns a structured markdown summary covering:

- Policy Overview
- Coverage Details & Benefits
- Costs & Financial Terms
- Exclusions & Limitations
- Claim Procedures
- Important Warnings & Red Flags

### 2. Intelligent Q&A

Ask natural language questions about your policy:

- "What is my deductible?"
- "Are pre-existing conditions covered?"
- "How do I file a claim?"

**Features:**
- Confidence levels (high/medium/low/none)
- Optional source excerpts from the document (`detailed: true`)

### 3. Policy Comparison

Compare two policies across 20 categories:

- Premium Amount, Deductible, Co-payment
- Coverage Limits, Exclusions, Waiting Periods
- Claim Process, Cancellation Terms, Special Benefits
- Key differences highlighted
- AI verdict and recommendation

### 4. Risk Scoring

Transparent, rule-based risk scoring:

- Gemini extracts 5 key features (exclusions count, coverage amount, premium balance, waiting period, claim complexity)
- Python applies fixed scoring rules to each feature
- Final score maps to Low / Medium / High risk
- Full breakdown with per-factor reasoning included

### 5. Multi-Format Document Processing

- **PDF**: Direct text extraction with pdfplumber
- **DOCX/DOC**: Native Word document parsing
- **Scanned PDFs**: OCR with Tesseract

---

## Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **[setup.md](setup.md)** | Complete setup guide | First-time installation |
| **[run-project.md](run-project.md)** | How to run the project | Daily startup |
| **[api-documentation.md](api-documentation.md)** | API reference | Frontend integration |
| **[README.md](README.md)** | Project overview | Understanding the project |

---

## Configuration

### Environment Variables

Create `insurance_ai/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free Gemini API key from: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Optional: Ollama (Local Fallback)

If you want to use the `/local-summary`, `/local-qa`, or `/compare` fallback routes, install Ollama:

```cmd
ollama serve
ollama pull llama3.2:3b
```

**Recommended models:**
- `llama3.2:3b` - Best for 4GB VRAM (default)
- `phi3:mini` - Fast, good quality
- `mistral:7b` - Better quality (requires 8GB+ VRAM)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not configured" | Add `GEMINI_API_KEY` to `insurance_ai/.env` |
| "Python not recognized" | Add Python to PATH (reinstall with checkbox) |
| Port 5000 in use | Use different port: `python app.py --port 5001` |
| "Ollama not running" | Only needed for fallback routes — use Gemini endpoints instead |

For more troubleshooting tips, see **[setup.md](setup.md)**

---

## System Requirements

### Minimum (Gemini API mode)
- Windows 10/11 (64-bit)
- 4GB RAM
- Internet connection
- No GPU required

### If using Ollama (local fallback)
- 8GB RAM
- 4GB VRAM (for GPU acceleration) or CPU
- 10GB free disk space for models

---

## Performance

| Operation               | Time (approx.) | Notes                        |
| ----------------------- | -------------- | ---------------------------- |
| Document upload         | 1-3 seconds    | Depends on file size         |
| Gemini summary          | 10-30 seconds  | Full structured summary      |
| Gemini Q&A              | 5-15 seconds   | Per question                 |
| Gemini comparison       | 15-40 seconds  | Full 20-category comparison  |
| Gemini risk score       | 8-20 seconds   | Includes scoring breakdown   |

---

## Technical Highlights

- **Gemini Cloud AI** - Full document understanding without local GPU
- **Rule-Based Risk Scoring** - Deterministic, auditable scoring on top of AI extraction
- **Research Routes Included** - BERT-based extractive summarization and QA for academic comparison (`/scratch-*`)
- **Ollama Fallback** - Local LLM routes available when cloud AI is unavailable
- **Source Attribution** - Answers linked back to source text excerpts

---

## License

This project is provided as-is for educational and personal use.

---

## Support

For issues, questions, or feature requests:

1. Check the documentation files ([setup.md](setup.md), [run-project.md](run-project.md), [api-documentation.md](api-documentation.md))
2. Review the troubleshooting sections
3. Verify `GEMINI_API_KEY` is set in `insurance_ai/.env`
4. Ensure all dependencies are installed correctly

---

## Credits

**AI Models:**
- Google Gemini 2.5 Flash (primary)
- Llama 3.2 (Meta AI) — optional fallback
- RoBERTa (Hugging Face) — research routes

**Libraries:**
- PyTorch, Transformers, Sentence-Transformers
- FAISS (Facebook AI)
- Flask, pdfplumber, python-docx

---

**Built with Google Gemini AI for practical insurance policy analysis**

*Last updated: March 2026*