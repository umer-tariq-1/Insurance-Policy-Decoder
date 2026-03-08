# Insurance Policy Decoder - API Documentation

Complete API reference for frontend integration.

**Base URL:** `http://localhost:5000`

---

## Table of Contents

1. [Overview](#overview)
2. [Document Management](#document-management)
   - [Upload Document](#upload-document)
   - [Get Document Content](#get-document-content)
3. [Document Summarization](#document-summarization)
   - [Gemini API Summary (Recommended)](#gemini-api-summary-recommended)
   - [Research: Extractive Summary](#research-extractive-summary)
   - [Local AI Summary](#local-ai-summary)
4. [Question & Answer](#question--answer)
   - [Gemini API Q&A (Recommended)](#gemini-api-qa-recommended)
   - [Research: BERT Q&A](#research-bert-qa)
   - [Local AI Q&A](#local-ai-qa)
   - [Get Suggested Questions](#get-suggested-questions)
5. [Document Comparison](#document-comparison)
   - [Gemini API Comparison (Recommended)](#gemini-api-comparison-recommended)
   - [Local AI Comparison](#local-ai-comparison)
   - [Quick Comparison](#quick-comparison)
6. [System & Configuration](#system--configuration)
   - [Health Check](#health-check)
   - [Ollama Status](#ollama-status)
   - [Configure Ollama](#configure-ollama)
   - [Clear Caches](#clear-caches)
7. [Error Handling](#error-handling)
8. [Integration Examples](#integration-examples)

---

## Overview

### Route Categories

| Category            | Prefix                 | Description                                        |
| ------------------- | ---------------------- | -------------------------------------------------- |
| Document Management | `/upload`, `/content`  | Upload and extract document content                |
| Gemini Cloud        | `/gemini-*`            | Cloud-based AI (primary — use these in frontend)   |
| Local AI (Ollama)   | `/local-*`             | Local LLM features (requires Ollama installed)     |
| Research/Academic   | `/scratch-*`           | BERT-based implementations, do not use in frontend |
| System              | `/health`, `/ollama/*` | Health checks and configuration                    |

### Recommended Routes for Frontend

Use these Gemini endpoints as the primary implementation:

| Feature    | Primary Route           | Fallback (if no API key) |
| ---------- | ----------------------- | ------------------------ |
| Summary    | `/gemini-api-summary`   | `/local-summary`         |
| Q&A        | `/gemini-api-qa`        | `/local-qa`              |
| Comparison | `/gemini-api-compare`   | `/compare`               |

> **Why Gemini?** The local machine does not have sufficient GPU resources to run local pretrained models reliably. Gemini API provides fast, high-quality results without local hardware requirements.

---

## Document Management

### Upload Document

Upload an insurance policy document (PDF, DOCX, or DOC).

**Endpoint:** `POST /upload`

**Content-Type:** `multipart/form-data`

**Request:**

| Field | Type | Required | Description            |
| ----- | ---- | -------- | ---------------------- |
| file  | File | Yes      | PDF, DOCX, or DOC file |

**Example (using FormData):**

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:5000/upload", {
  method: "POST",
  body: formData,
});
```

**Response (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "filename": "my_policy.pdf",
  "size": 245678
}
```

| Field    | Type    | Description                                                                        |
| -------- | ------- | ---------------------------------------------------------------------------------- |
| hash     | string  | Unique identifier for the document (SHA-256). Use this in all subsequent API calls |
| filename | string  | Original filename                                                                  |
| size     | integer | File size in bytes                                                                 |

**Response (Error - 400):**

```json
{
  "error": "Only PDF, DOCX or DOC files allowed"
}
```

**Purpose:** This is the first step in the workflow. Upload a document to get a unique hash that identifies the document for all other operations.

---

### Get Document Content

Extract and retrieve the raw text content from an uploaded document.

**Endpoint:** `POST /content`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789..."
}
```

| Field | Type   | Required | Description               |
| ----- | ------ | -------- | ------------------------- |
| hash  | string | Yes      | Document hash from upload |

**Response (Success - 200):**

```json
{
  "total_chunks": 15,
  "chunks_string": "This insurance policy provides coverage for...\n\nSection 2: Coverage Details..."
}
```

| Field         | Type    | Description                                         |
| ------------- | ------- | --------------------------------------------------- |
| total_chunks  | integer | Number of text chunks extracted                     |
| chunks_string | string  | Full extracted text with chunks separated by `\n\n` |

**Purpose:** Use this to display the raw document content to users or for debugging. The text is extracted and chunked for easier processing.

---

## Document Summarization

### Gemini API Summary (Recommended)

Generate a comprehensive insurance policy summary using Google's Gemini AI.

**Endpoint:** `POST /gemini-api-summary`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789..."
}
```

| Field | Type   | Required | Description               |
| ----- | ------ | -------- | ------------------------- |
| hash  | string | Yes      | Document hash from upload |

**Response (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "summary": "**Policy Overview:**\n- Policy type: Health Insurance...\n\n**Important Coverage Details:**\n- Coverage limit: $1,000,000..."
}
```

| Field   | Type   | Description                              |
| ------- | ------ | ---------------------------------------- |
| hash    | string | Document hash                            |
| summary | string | Markdown-formatted comprehensive summary |

**Response (Error - 500):**

```json
{
  "error": "Failed to generate summary: API key not configured"
}
```

**Purpose:** Generate a detailed, structured summary of an insurance policy using Gemini cloud AI. Covers policy overview, coverage details, costs, exclusions, claim procedures, and red flags. **Use this as the primary summary feature.**

---

### Research: Extractive Summary (do not integrate with Frontend)

_Academic/Research implementation using BERT-based extractive summarization._

**Endpoint:** `POST /scratch-summary`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789..."
}
```

**Response (Success - 200):**

```json
{
  "total_text_length": 45678,
  "important_points": [
    "This policy provides comprehensive health coverage for the insured and dependents.",
    "The annual deductible is $500 per individual or $1,000 per family.",
    "Coverage includes hospitalization, surgery, and prescription drugs."
  ]
}
```

| Field             | Type    | Description                                     |
| ----------------- | ------- | ----------------------------------------------- |
| total_text_length | integer | Length of original document text                |
| important_points  | array   | Array of extracted key sentences (25 sentences) |

**Purpose:** Academic implementation showing extractive summarization using BERT embeddings and semantic similarity. Extracts existing sentences rather than generating new text. **For production, use `/gemini-api-summary` instead.**

---

### Local AI Summary

Generate a summary using local Ollama LLM. Requires Ollama running locally.

**Endpoint:** `POST /local-summary`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "mode": "standard"
}
```

| Field | Type   | Required | Default    | Description                                                         |
| ----- | ------ | -------- | ---------- | ------------------------------------------------------------------- |
| hash  | string | Yes      | -          | Document hash from upload                                           |
| mode  | string | No       | "standard" | Summary detail level: `"quick"`, `"standard"`, or `"comprehensive"` |

**Response - Standard/Comprehensive Mode (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "mode": "standard",
  "summary": "## POLICY OVERVIEW\n\nThis is a health insurance policy...",
  "model": "llama3.2:3b",
  "sections_processed": 8,
  "sentences_analyzed": 80
}
```

**Response (Ollama Not Running - 503):**

```json
{
  "error": "Local LLM not available: Ollama not running. Start with: ollama serve",
  "setup_instructions": {
    "1": "Install Ollama from https://ollama.ai",
    "2": "Start Ollama: ollama serve",
    "3": "Pull a model: ollama pull llama3.2:3b"
  }
}
```

**Purpose:** Fallback summary option using a local Ollama model. Use only when Gemini API is unavailable.

---

## Question & Answer

### Gemini API Q&A (Recommended)

Ask questions about an insurance document and get AI-generated answers using Gemini.

**Endpoint:** `POST /gemini-api-qa`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible amount?",
  "detailed": false
}
```

| Field    | Type    | Required | Default | Description                                          |
| -------- | ------- | -------- | ------- | ---------------------------------------------------- |
| hash     | string  | Yes      | -       | Document hash from upload                            |
| question | string  | Yes      | -       | Question to ask about the document                   |
| detailed | boolean | No       | false   | If true, includes source text excerpts in response   |

**Response - Basic (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible amount?",
  "answer": "Your annual deductible is $500 per individual. For family coverage, the combined deductible is $1,000. The deductible applies to most covered services except preventive care, which is covered at 100%.",
  "confidence": "high",
  "model": "gemini-2.5-flash"
}
```

**Response - Detailed (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible amount?",
  "answer": "Your annual deductible is $500 per individual...",
  "confidence": "high",
  "model": "gemini-2.5-flash",
  "sources": [
    {
      "text": "Section 4: Cost Sharing. The annual deductible for individual coverage is $500...",
      "relevance": 1.0
    }
  ]
}
```

| Field               | Type   | Description                                                  |
| ------------------- | ------ | ------------------------------------------------------------ |
| answer              | string | AI-generated answer to the question                          |
| confidence          | string | Confidence level: `"high"`, `"medium"`, `"low"`, or `"none"` |
| model               | string | AI model used                                                |
| sources             | array  | Source excerpts from document (only if `detailed=true`)      |
| sources[].text      | string | Verbatim text excerpt from the document                      |
| sources[].relevance | float  | Always 1.0 for Gemini responses                              |

**Confidence Levels:**

| Level  | Meaning                            | UI Suggestion                      |
| ------ | ---------------------------------- | ---------------------------------- |
| high   | Answer found with strong evidence  | Show answer normally               |
| medium | Answer found but may be incomplete | Show with "may be incomplete" note |
| low    | Answer uncertain                   | Show with warning styling          |
| none   | No relevant info found             | Show "not found" message           |

**Purpose:** Allow users to ask natural language questions about their insurance policy. Gemini reads the full document and generates accurate, context-aware answers. **Use this as the primary Q&A feature.**

---

### Research: BERT Q&A (do not integrate with Frontend)

_Academic/Research implementation using BERT-based extractive QA._

**Endpoint:** `POST /scratch-qa`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible?",
  "detailed": false
}
```

**Response (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible?",
  "answer": "$500 annual",
  "confidence": "medium",
  "confidence_score": 0.456
}
```

**Purpose:** Academic implementation using RoBERTa model trained on SQuAD 2.0 for extractive question answering. Extracts exact text spans rather than generating answers. **For production, use `/gemini-api-qa` instead.**

---

### Local AI Q&A

Ask questions using local Ollama LLM with RAG. Requires Ollama running locally.

**Endpoint:** `POST /local-qa`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible?",
  "detailed": false
}
```

**Response (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible?",
  "answer": "Your annual deductible is $500 per individual...",
  "confidence": "high",
  "relevance_score": 0.847,
  "model": "llama3.2:3b"
}
```

**Purpose:** Fallback Q&A option using local Ollama model with RAG. Use only when Gemini API is unavailable.

---

### Get Suggested Questions

Get AI-suggested questions for a document. Requires the document to have been prepared via `/local-qa` first.

**Endpoint:** `POST /local-qa/suggestions`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789..."
}
```

**Response (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "suggestions": [
    "What is covered under this policy?",
    "What are the main exclusions?",
    "What is the deductible amount?",
    "How do I file a claim?",
    "What is the coverage limit?",
    "Are pre-existing conditions covered?",
    "What is the waiting period?"
  ]
}
```

**Purpose:** Provide users with helpful starting questions they can ask about their document. Display these as clickable suggestions in the UI.

---

## Document Comparison

### Gemini API Comparison (Recommended)

Compare two insurance policies side-by-side across 20 categories using Gemini AI. Returns the same response structure as `/compare` for full frontend compatibility.

**Endpoint:** `POST /gemini-api-compare`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash1": "a1b2c3d4e5f6789...",
  "hash2": "x9y8z7w6v5u4321...",
  "include_verdict": true
}
```

| Field           | Type    | Required | Default | Description               |
| --------------- | ------- | -------- | ------- | ------------------------- |
| hash1           | string  | Yes      | -       | First document hash       |
| hash2           | string  | Yes      | -       | Second document hash      |
| include_verdict | boolean | No       | true    | Include AI recommendation |

**Response (Success - 200):**

```json
{
  "categories": [
    "Policy Type",
    "Coverage Scope",
    "Premium Amount",
    "Deductible",
    "Co-payment/Co-insurance",
    "Out-of-Pocket Maximum",
    "Coverage Limit (Per Incident)",
    "Coverage Limit (Annual/Lifetime)",
    "In-Network Benefits",
    "Out-of-Network Benefits",
    "Waiting Period",
    "Pre-existing Conditions",
    "Key Exclusions",
    "Claim Filing Process",
    "Claim Deadline",
    "Cancellation Policy",
    "Renewal Terms",
    "Grace Period",
    "Key Warnings/Red Flags",
    "Special Benefits/Riders"
  ],
  "policy1": {
    "hash": "a1b2c3d4e5f6789...",
    "values": [
      "Health Insurance",
      "Comprehensive medical coverage",
      "$500/month",
      "$1,000 annual",
      "20% co-insurance after deductible",
      "$6,000 individual / $12,000 family",
      "$100,000 per incident",
      "$1,000,000 lifetime",
      "90% coverage",
      "60% coverage",
      "30 days",
      "12-month waiting period",
      "Cosmetic surgery, experimental treatments",
      "Submit claim form within 90 days",
      "90 days from service date",
      "30-day notice required",
      "Auto-renewal annually",
      "15 days",
      "Pre-authorization required for hospital stays",
      "Dental and vision riders available"
    ]
  },
  "policy2": {
    "hash": "x9y8z7w6v5u4321...",
    "values": [
      "Health Insurance",
      "Basic medical coverage",
      "$350/month",
      "$2,500 annual",
      "30% co-insurance after deductible",
      "$8,000 individual / $16,000 family",
      "$50,000 per incident",
      "$500,000 lifetime",
      "80% coverage",
      "50% coverage",
      "60 days",
      "24-month waiting period",
      "Mental health, cosmetic surgery",
      "Online claim submission",
      "60 days from service date",
      "60-day notice required",
      "Manual renewal",
      "10 days",
      "No coverage outside network",
      "None"
    ]
  },
  "highlights": [
    {
      "category": "Premium Amount",
      "type": "cost",
      "policy1": "$500/month",
      "policy2": "$350/month",
      "note": "Policy 2 is $150/month cheaper"
    },
    {
      "category": "Deductible",
      "type": "cost",
      "policy1": "$1,000 annual",
      "policy2": "$2,500 annual",
      "note": "Policy 1 has a lower deductible — less out-of-pocket per claim"
    }
  ],
  "verdict": "Policy 1 offers more comprehensive coverage with lower deductibles and higher coverage limits, but at a higher premium. Policy 2 is more affordable but has higher out-of-pocket costs and more restrictions.",
  "model": "gemini-2.5-flash"
}
```

**Response Structure for Table Display:**

```javascript
// Example: Building a comparison table
const { categories, policy1, policy2 } = response;

categories.forEach((category, index) => {
  const row = {
    category: category,
    policy1Value: policy1.values[index],
    policy2Value: policy2.values[index],
  };
  // Render row in table
});
```

| Field          | Type   | Description                                         |
| -------------- | ------ | --------------------------------------------------- |
| categories     | array  | Array of 20 comparison category names               |
| policy1.hash   | string | Hash of first document                              |
| policy1.values | array  | Values for each category (same order as categories) |
| policy2.hash   | string | Hash of second document                             |
| policy2.values | array  | Values for each category (same order as categories) |
| highlights     | array  | Key differences with analysis (3-6 entries)         |
| verdict        | string | AI recommendation and analysis                      |
| model          | string | AI model used                                       |

**Purpose:** Compare two insurance policies side-by-side. Gemini reads both full documents and extracts comparable information. **Use this as the primary comparison feature.**

---

### Local AI Comparison

Compare two policies using local Ollama LLM. Requires Ollama running locally.

**Endpoint:** `POST /compare`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash1": "a1b2c3d4e5f6789...",
  "hash2": "x9y8z7w6v5u4321...",
  "include_verdict": true
}
```

Returns the same response structure as `/gemini-api-compare`. Use as fallback when Gemini API is unavailable.

**Response (Ollama Not Running - 503):**

```json
{
  "error": "Local LLM not available: Ollama not running. Start with: ollama serve",
  "setup_instructions": {
    "1": "Install Ollama from https://ollama.ai",
    "2": "Start Ollama: ollama serve",
    "3": "Pull a model: ollama pull llama3.2:3b"
  }
}
```

---

### Quick Comparison

Faster comparison focusing on top 10 differences. Uses local Ollama.

**Endpoint:** `POST /compare/quick`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash1": "a1b2c3d4e5f6789...",
  "hash2": "x9y8z7w6v5u4321..."
}
```

**Response (Success - 200):**

```json
{
  "differences": [
    {
      "category": "Premium",
      "policy1": "$500/month",
      "policy2": "$350/month"
    },
    {
      "category": "Deductible",
      "policy1": "$1,000 annual",
      "policy2": "$2,500 annual"
    }
  ],
  "policy1_hash": "a1b2c3d4e5f6789...",
  "policy2_hash": "x9y8z7w6v5u4321...",
  "model": "llama3.2:3b",
  "type": "quick_comparison"
}
```

**Purpose:** Quick overview of main differences. Use `/gemini-api-compare` for the full comparison.

---

## System & Configuration

### Health Check

Check if the Flask server is running.

**Endpoint:** `GET /health`

**Response (Success - 200):**

```json
{
  "status": "Flask server running"
}
```

---

### Ollama Status

Check if Ollama is running and the required model is available.

**Endpoint:** `GET /ollama/status`

**Response (Ollama Running - 200):**

```json
{
  "available": true,
  "message": "OK"
}
```

**Response (Ollama Not Running - 503):**

```json
{
  "available": false,
  "message": "Ollama not running. Start with: ollama serve",
  "setup_instructions": {
    "1": "Install Ollama from https://ollama.ai",
    "2": "Start Ollama: ollama serve",
    "3": "Pull a model: ollama pull llama3.2:3b (recommended for 4GB VRAM)",
    "alternative_models": [
      "phi3:mini (fast, good quality)",
      "qwen2.5:3b (good for structured output)",
      "mistral:7b (if you have 8GB+ VRAM)"
    ]
  }
}
```

---

### Configure Ollama

Change Ollama settings (model, URL, temperature).

**Endpoint:** `POST /ollama/configure`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "model": "llama3.2:3b",
  "ollama_url": "http://localhost:11434",
  "temperature": 0.3
}
```

| Field       | Type   | Required | Description                                         |
| ----------- | ------ | -------- | --------------------------------------------------- |
| model       | string | No       | Model name (e.g., "llama3.2:3b", "mistral:7b")      |
| ollama_url  | string | No       | Ollama server URL                                   |
| temperature | float  | No       | Response randomness (0.0-1.0, lower = more focused) |

**Response (Success - 200):**

```json
{
  "message": "Configuration updated",
  "current_settings": {
    "model": "llama3.2:3b",
    "ollama_url": "unchanged",
    "temperature": 0.3
  }
}
```

---

### Clear Caches

Clear document processing caches.

**Clear Ollama QA Cache:**

**Endpoint:** `POST /local-qa/clear-cache`

**Request Body (Optional):**

```json
{
  "hash": "specific_document_hash"
}
```

**Clear BERT QA Cache:**

**Endpoint:** `POST /qa/clear-cache`

**Request Body (Optional):**

```json
{
  "hash": "specific_document_hash"
}
```

**Response (Success - 200):**

```json
{
  "message": "Cleared all Ollama QA cache (2 documents)",
  "remaining_cached": 0
}
```

---

## Error Handling

### Standard Error Response

All errors follow this format:

```json
{
  "error": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning             | When                                      |
| ---- | ------------------- | ----------------------------------------- |
| 200  | Success             | Request completed successfully            |
| 400  | Bad Request         | Invalid or missing parameters             |
| 404  | Not Found           | Document hash not found                   |
| 500  | Server Error        | Internal processing error                 |
| 503  | Service Unavailable | Ollama not running or model not available |

### Common Errors

| Error                                 | Cause                            | Solution                             |
| ------------------------------------- | -------------------------------- | ------------------------------------ |
| "Hash required in request body"       | Missing hash parameter           | Include the document hash            |
| "Question required in request body"   | Missing question parameter       | Include the question string          |
| "File not found"                      | Invalid hash or document deleted | Re-upload the document               |
| "Ollama not running"                  | Ollama service not started       | Use Gemini endpoints or run `ollama serve` |
| "Model not found"                     | AI model not downloaded          | Run `ollama pull llama3.2:3b`        |
| "Only PDF, DOCX or DOC files allowed" | Wrong file type                  | Upload supported file type           |

---

## Integration Examples

### Complete Upload and Summary Flow

```javascript
async function uploadAndSummarize(file) {
  // 1. Upload document
  const formData = new FormData();
  formData.append("file", file);

  const uploadResponse = await fetch("http://localhost:5000/upload", {
    method: "POST",
    body: formData,
  });
  const { hash } = await uploadResponse.json();

  // 2. Generate summary using Gemini
  const summaryResponse = await fetch("http://localhost:5000/gemini-api-summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash }),
  });
  const summary = await summaryResponse.json();

  return { hash, summary };
}
```

### Q&A Flow

```javascript
async function askQuestion(hash, question, detailed = false) {
  const response = await fetch("http://localhost:5000/gemini-api-qa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash, question, detailed }),
  });

  const data = await response.json();

  if (data.error) {
    console.error("Q&A failed:", data.error);
    return null;
  }

  // data.confidence: "high" | "medium" | "low" | "none"
  // data.answer: string
  // data.sources: array (only if detailed=true)
  return data;
}
```

### Comparison Flow

```javascript
async function comparePolicies(hash1, hash2) {
  const response = await fetch("http://localhost:5000/gemini-api-compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash1, hash2, include_verdict: true }),
  });

  const data = await response.json();

  // Build comparison table
  const tableRows = data.categories.map((category, index) => ({
    category,
    policy1: data.policy1.values[index],
    policy2: data.policy2.values[index],
  }));

  return {
    tableRows,
    highlights: data.highlights,
    verdict: data.verdict,
  };
}
```

---

## Quick Reference

### Primary Endpoints (use these in the frontend)

| Feature          | Endpoint                | Method           |
| ---------------- | ----------------------- | ---------------- |
| Upload document  | `/upload`               | POST (multipart) |
| Generate summary | `/gemini-api-summary`   | POST             |
| Ask question     | `/gemini-api-qa`        | POST             |
| Compare policies | `/gemini-api-compare`   | POST             |
| Health check     | `/health`               | GET              |

### Response Times (Approximate)

| Endpoint                | Expected Time  |
| ----------------------- | -------------- |
| `/upload`               | 1-3 seconds    |
| `/gemini-api-summary`   | 10-30 seconds  |
| `/gemini-api-qa`        | 5-15 seconds   |
| `/gemini-api-compare`   | 15-40 seconds  |

---

_API Documentation version 2.0_
