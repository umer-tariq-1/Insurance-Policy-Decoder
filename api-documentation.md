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
   - [Local AI Summary (Recommended)](#local-ai-summary-recommended)
   - [Gemini API Summary](#gemini-api-summary)
   - [Research: Extractive Summary](#research-extractive-summary)
4. [Question & Answer](#question--answer)
   - [Local AI Q&A (Recommended)](#local-ai-qa-recommended)
   - [Get Suggested Questions](#get-suggested-questions)
   - [Research: BERT Q&A](#research-bert-qa)
5. [Document Comparison](#document-comparison)
   - [Full Comparison](#full-comparison)
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
| Local AI (Ollama)   | `/local-*`             | High-quality AI features using local LLM           |
| Gemini Cloud        | `/gemini-*`            | Cloud-based AI (requires API key)                  |
| Research/Academic   | `/scratch-*`           | BERT-based implementations for academic comparison |
| Comparison          | `/compare`             | Side-by-side policy comparison                     |
| System              | `/health`, `/ollama/*` | Health checks and configuration                    |

### Recommended Routes for Production

For the best user experience, use these routes:

| Feature    | Recommended Route | Fallback              |
| ---------- | ----------------- | --------------------- |
| Summary    | `/local-summary`  | `/gemini-api-summary` |
| Q&A        | `/local-qa`       | `/scratch-qa`         |
| Comparison | `/compare`        | -                     |

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

### Local AI Summary (Summary using Ollama, integrate with Frontend)

Generate a comprehensive insurance policy summary using local Ollama LLM.

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

**Mode Options:**

| Mode          | Speed             | Detail Level                    | Best For       |
| ------------- | ----------------- | ------------------------------- | -------------- |
| quick         | Fast (~30s)       | Bullet points (20 items)        | Quick overview |
| standard      | Medium (~1-2 min) | Structured sections             | Regular use    |
| comprehensive | Slow (~3-5 min)   | Very detailed with all sections | Full analysis  |

**Response - Standard/Comprehensive Mode (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "mode": "standard",
  "summary": "## POLICY OVERVIEW\n\nThis is a health insurance policy...\n\n## COVERAGE DETAILS & BENEFITS\n\n### What's Covered\n- Hospital stays...",
  "model": "llama3.2:3b",
  "sections_processed": 8,
  "sentences_analyzed": 80
}
```

**Response - Quick Mode (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "mode": "quick",
  "points": [
    {
      "category": "COVERAGE",
      "content": "Covers hospitalization up to $1,000,000 annually"
    },
    {
      "category": "DEDUCTIBLE",
      "content": "Annual deductible of $500 per individual"
    },
    {
      "category": "EXCLUSION",
      "content": "Pre-existing conditions not covered for first 12 months"
    },
    {
      "category": "WARNING",
      "content": "Claims must be filed within 90 days of treatment"
    }
  ],
  "model": "llama3.2:3b",
  "type": "quick_summary"
}
```

| Field              | Type    | Description                                                                                            |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------------ |
| summary            | string  | Markdown-formatted summary (standard/comprehensive modes)                                              |
| points             | array   | Array of categorized bullet points (quick mode)                                                        |
| points[].category  | string  | Category: COVERAGE, BENEFIT, COST, DEDUCTIBLE, EXCLUSION, LIMIT, DEADLINE, REQUIREMENT, CLAIM, WARNING |
| points[].content   | string  | The summary point content                                                                              |
| model              | string  | AI model used                                                                                          |
| sections_processed | integer | Number of document sections analyzed                                                                   |
| sentences_analyzed | integer | Total sentences processed                                                                              |

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

**Purpose:** Generate a detailed, structured summary of an insurance policy. The summary covers policy overview, coverage details, costs, exclusions, claim procedures, and important warnings. Use this as the primary summary feature.

---

### Gemini API Summary (Summary using Gemini API, donot integrate with Frontend)

Generate summary using Google's Gemini AI (cloud-based).

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
  "summary": "**Policy Overview:**\n- Policy type: Health Insurance..."
}
```

**Response (Error - 500):**

```json
{
  "error": "Failed to generate summary: API key not configured"
}
```

**Purpose:** Alternative cloud-based summary option. Requires GEMINI_API_KEY in .env file. Use as fallback when Ollama is not available, or for comparison.

---

### Research: Extractive Summary (Summary using BERT, donot integrate with Frontend)

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

**Purpose:** This is an academic implementation showing extractive summarization using BERT embeddings and semantic similarity. It extracts existing sentences rather than generating new text. Included to demonstrate research methodology. **For production, use `/local-summary` instead.**

---

## Question & Answer

### Local AI Q&A (QnA using Ollama, integrate with Frontend)

Ask questions about an insurance document and get AI-generated answers.

**Endpoint:** `POST /local-qa`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible amount?",
  "detailed": false
}
```

| Field    | Type    | Required | Default | Description                                   |
| -------- | ------- | -------- | ------- | --------------------------------------------- |
| hash     | string  | Yes      | -       | Document hash from upload                     |
| question | string  | Yes      | -       | Question to ask about the document            |
| detailed | boolean | No       | false   | If true, includes source sections in response |

**Response - Basic (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible amount?",
  "answer": "Your annual deductible is $500 per individual. For family coverage, the combined deductible is $1,000. The deductible applies to most covered services except preventive care, which is covered at 100% with no deductible.",
  "confidence": "high",
  "relevance_score": 0.847,
  "model": "llama3.2:3b"
}
```

**Response - Detailed (Success - 200):**

```json
{
  "hash": "a1b2c3d4e5f6789...",
  "question": "What is my deductible amount?",
  "answer": "Your annual deductible is $500 per individual...",
  "confidence": "high",
  "relevance_score": 0.847,
  "model": "llama3.2:3b",
  "sources": [
    {
      "text": "Section 4: Cost Sharing. The annual deductible for individual coverage is $500. Family deductible is $1,000...",
      "relevance": 0.847
    },
    {
      "text": "Preventive care services are covered at 100% and do not apply to the deductible...",
      "relevance": 0.723
    }
  ]
}
```

| Field               | Type   | Description                                                  |
| ------------------- | ------ | ------------------------------------------------------------ |
| answer              | string | AI-generated answer to the question                          |
| confidence          | string | Confidence level: `"high"`, `"medium"`, `"low"`, or `"none"` |
| relevance_score     | float  | How relevant the found content is (0-1)                      |
| sources             | array  | Source sections used to answer (only if detailed=true)       |
| sources[].text      | string | Text excerpt from document                                   |
| sources[].relevance | float  | Relevance score of this source                               |
| note                | string | Additional info (appears when confidence is low)             |

**Confidence Levels:**

| Level  | Meaning                            | UI Suggestion                      |
| ------ | ---------------------------------- | ---------------------------------- |
| high   | Answer found with strong evidence  | Show answer normally               |
| medium | Answer found but may be incomplete | Show with "may be incomplete" note |
| low    | Answer uncertain                   | Show with warning styling          |
| none   | No relevant info found             | Show "not found" message           |

**Purpose:** Allow users to ask natural language questions about their insurance policy. Uses RAG (Retrieval Augmented Generation) to find relevant sections and generate accurate answers. This is the primary Q&A feature.

---

### Get Suggested Questions

Get AI-suggested questions for a document.

**Endpoint:** `POST /local-qa/suggestions`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "hash": "a1b2c3d4e5f6789..."
}
```

**Note:** The document must have been prepared first by sending at least one question to `/local-qa`.

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

### Research: BERT Q&A (QnA using BERT, donot integrate with Frontend)

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

**Purpose:** Academic implementation using RoBERTa model trained on SQuAD 2.0 for extractive question answering. Extracts exact text spans from the document rather than generating answers. **For production, use `/local-qa` instead** as it provides more comprehensive answers.

---

## Document Comparison

### Full Comparison (Comparison using Vectorization and Ollama, integrate with Frontend)

Compare two insurance policies side-by-side across 20 categories.

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
      "note": "lower is typically better"
    },
    {
      "category": "Deductible",
      "type": "cost",
      "policy1": "$1,000 annual",
      "policy2": "$2,500 annual",
      "note": "lower deductible means less out-of-pocket per claim"
    }
  ],
  "verdict": "Policy 1 offers more comprehensive coverage with lower deductibles and higher coverage limits, but at a higher premium. Policy 2 is more affordable but has higher out-of-pocket costs and more restrictions. Policy 1 is better for those who want maximum protection and can afford higher premiums. Policy 2 is suitable for healthy individuals who want basic coverage at lower cost.",
  "model": "llama3.2:3b"
}
```

**Response Structure for Table Display:**

The response is designed for easy table rendering:

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
| highlights     | array  | Key differences with analysis                       |
| verdict        | string | AI recommendation and analysis                      |

**Purpose:** Enable users to compare two insurance policies side-by-side. Display as a table with categories as rows and policies as columns. The highlights section shows the most important differences, and the verdict provides an AI-powered recommendation.

---

### Quick Comparison

Faster comparison focusing on top 10 differences.

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

**Purpose:** Quick overview of main differences between two policies. Use when speed is important or for an initial comparison before detailed analysis.

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

**Purpose:** Basic health check for monitoring. Use to verify the backend is accessible.

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

**Purpose:** Check Ollama availability before showing AI features. If unavailable, show setup instructions or fall back to alternative features.

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

**Purpose:** Advanced configuration for power users. Allow changing the AI model or adjusting response characteristics.

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

**Purpose:** Force re-processing of documents. Use when documents are updated or to free memory.

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

| Error                                 | Cause                            | Solution                      |
| ------------------------------------- | -------------------------------- | ----------------------------- |
| "Hash required in request body"       | Missing hash parameter           | Include the document hash     |
| "File not found"                      | Invalid hash or document deleted | Re-upload the document        |
| "Ollama not running"                  | Ollama service not started       | Run `ollama serve`            |
| "Model not found"                     | AI model not downloaded          | Run `ollama pull llama3.2:3b` |
| "Only PDF, DOCX or DOC files allowed" | Wrong file type                  | Upload supported file type    |

---

## Integration Examples

### Complete Upload and Summary Flow

```javascript
// 1. Upload document
async function uploadAndSummarize(file) {
  // Upload
  const formData = new FormData();
  formData.append("file", file);

  const uploadResponse = await fetch("http://localhost:5000/upload", {
    method: "POST",
    body: formData,
  });
  const { hash } = await uploadResponse.json();

  // Generate summary
  const summaryResponse = await fetch("http://localhost:5000/local-summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash, mode: "standard" }),
  });
  const summary = await summaryResponse.json();

  return { hash, summary };
}
```

### Q&A Flow with Error Handling

```javascript
async function askQuestion(hash, question) {
  // Check Ollama status first
  const statusResponse = await fetch("http://localhost:5000/ollama/status");
  const status = await statusResponse.json();

  if (!status.available) {
    return {
      error: "AI service not available",
      instructions: status.setup_instructions,
    };
  }

  // Ask question
  const response = await fetch("http://localhost:5000/local-qa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash, question, detailed: true }),
  });

  return await response.json();
}
```

### Comparison Flow

```javascript
async function comparePolicies(hash1, hash2) {
  const response = await fetch("http://localhost:5000/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash1, hash2, include_verdict: true }),
  });

  const data = await response.json();

  // Build comparison table data
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

### Most Used Endpoints

| Feature          | Endpoint         | Method           |
| ---------------- | ---------------- | ---------------- |
| Upload document  | `/upload`        | POST (multipart) |
| Generate summary | `/local-summary` | POST             |
| Ask question     | `/local-qa`      | POST             |
| Compare policies | `/compare`       | POST             |
| Check status     | `/ollama/status` | GET              |

### Response Times (Approximate)

| Endpoint         | Mode           | Expected Time                        |
| ---------------- | -------------- | ------------------------------------ |
| `/upload`        | -              | 1-3 seconds                          |
| `/local-summary` | quick          | 30-60 seconds                        |
| `/local-summary` | standard       | 1-2 minutes                          |
| `/local-summary` | comprehensive  | 3-5 minutes                          |
| `/local-qa`      | first question | 30-60 seconds (includes preparation) |
| `/local-qa`      | subsequent     | 10-30 seconds                        |
| `/compare`       | full           | 2-4 minutes                          |
| `/compare/quick` | -              | 30-60 seconds                        |

---

_API Documentation version 1.0_
