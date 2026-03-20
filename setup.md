# Insurance Policy Decoder - Setup Guide

This guide will help you set up and run the Insurance Policy Decoder application on your Windows machine.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Install Python](#step-1-install-python)
3. [Step 2: Set Up the Project](#step-2-set-up-the-project)
4. [Step 3: Configure Gemini API Key](#step-3-configure-gemini-api-key)
5. [Step 4: Install Tesseract OCR (Optional)](#step-4-install-tesseract-ocr-optional)
6. [Step 5: Run the Application](#step-5-run-the-application)
7. [Verifying the Setup](#verifying-the-setup)
8. [Optional: Install Ollama (Local Fallback)](#optional-install-ollama-local-fallback)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, make sure you have:

- **Windows 10/11** (64-bit)
- **4GB RAM minimum**
- **Internet connection** (for Gemini API calls and initial setup)
- **Google account** (to get a free Gemini API key)

> **No GPU required.** All primary AI features run through the Google Gemini cloud API.

---

## Step 1: Install Python

### Download Python

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download **Python 3.10** or higher (3.11 or 3.12 recommended)
3. Run the installer

### Important Installation Settings

During installation, make sure to:

- [x] **Check "Add Python to PATH"** (very important!)
- [x] Select "Install for all users" (recommended)

### Verify Installation

Open **Command Prompt** (search "cmd" in Start menu) and run:

```cmd
python --version
```

You should see something like: `Python 3.11.5`

Also verify pip:

```cmd
pip --version
```

---

## Step 2: Set Up the Project

### Navigate to Project Folder

Open Command Prompt and navigate to the project folder:

```cmd
cd "path\to\Insurance Policy Decoder"
```

For example:
```cmd
cd "C:\Projects\Insurance Policy Decoder"
```

### Create Virtual Environment

```cmd
python -m venv venv
```

### Activate Virtual Environment

```cmd
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command line.

> **Important:** Always activate the virtual environment before running the application!

### Install Dependencies

```cmd
cd insurance_ai
pip install -r requirements.txt
```

This will install all required Python packages. This may take several minutes as it downloads ML libraries.

---

## Step 3: Configure Gemini API Key

The Gemini API key is **required** — it powers all primary AI features (summary, Q&A, comparison, and risk scoring).

### Get Gemini API Key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key

### Create the .env File

In the `insurance_ai` folder, there should be a file named `.env`. If not, create it:

1. Open Notepad
2. Add the following content:

```env
GEMINI_API_KEY=AIzaSyD...your_actual_key...
```

3. Save as `.env` inside the `insurance_ai` folder (make sure it's not saved as `.env.txt`)

> **Note:** Without this key, all `/gemini-*` endpoints will return an error. These are the primary endpoints used by the frontend.

---

## Step 4: Install Tesseract OCR (Optional)

Tesseract is only needed to process scanned PDF documents (PDFs that are images rather than text).

### Download Tesseract

1. Go to [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.3.3.exe`)
3. Run the installer
4. **Important:** Note the installation path (default: `C:\Program Files\Tesseract-OCR`)

### Add to System PATH (if not done automatically)

1. Search "Environment Variables" in Windows Start menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "System variables", find "Path" and click "Edit"
5. Click "New" and add: `C:\Program Files\Tesseract-OCR`
6. Click OK on all dialogs

### Verify Installation

Open a new Command Prompt and run:

```cmd
tesseract --version
```

---

## Step 5: Run the Application

### Start the Flask Server

Open Command Prompt, navigate to the project, and run:

```cmd
cd "path\to\Insurance Policy Decoder"
venv\Scripts\activate
cd insurance_ai
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**The application is now running!**

> **No need to start Ollama** unless you want to use the optional local fallback routes (`/local-summary`, `/local-qa`, `/compare`).

---

## Verifying the Setup

### Check if Server is Running

Open your browser and go to:

```
http://localhost:5000/health
```

You should see:
```json
{"status": "Flask server running"}
```

### Test a Gemini Endpoint

You can test the API using tools like:
- **Postman** (download from [postman.com](https://www.postman.com/downloads/))
- **curl** (if installed)
- **Thunder Client** (VS Code extension)

Example test — upload a document and generate a summary:

```
POST http://localhost:5000/upload       (multipart, file field)
POST http://localhost:5000/gemini-api-summary   (JSON, hash field)
```

---

## Optional: Install Ollama (Local Fallback)

Ollama is **not required** for normal use. Install it only if you want to use the local fallback routes (`/local-summary`, `/local-qa`, `/compare`) when the Gemini API is unavailable.

### Download Ollama

1. Go to [ollama.ai/download](https://ollama.ai/download)
2. Click **"Download for Windows"**
3. Run the installer (`OllamaSetup.exe`)

### Download AI Model

```cmd
ollama serve
```

In a new window:

```cmd
ollama pull llama3.2:3b
```

This downloads the Llama 3.2 3B model (~2GB). Wait for it to complete.

### Verify

```cmd
ollama list
```

You should see `llama3.2:3b` in the list. Check Ollama connectivity at:

```
http://localhost:5000/ollama/status
```

---

## Troubleshooting

### "API key not configured" error

- The `.env` file is missing or the key is not set
- Solution: Create `insurance_ai/.env` with `GEMINI_API_KEY=your_key`

### "Python is not recognized"

- Python was not added to PATH during installation
- Solution: Reinstall Python and check "Add Python to PATH"

### "Module not found" error

- The virtual environment is not activated
- Solution: Run `venv\Scripts\activate` — you should see `(venv)` in the prompt

### Installation takes too long

- The ML libraries (torch, transformers) are large
- This is normal — the first installation can take 10-20 minutes

### Port 5000 already in use

- Another application is using port 5000
- Solution: Close that application or run Flask on a different port:
  ```cmd
  python app.py --port 5001
  ```

### "Ollama not running" error

- Only affects the optional local fallback routes (`/local-*`, `/compare`)
- Solution: Use the Gemini endpoints instead, or run `ollama serve`

### "CUDA out of memory" error

- Only relevant if using Ollama or research routes
- Solution: The app will automatically fall back to CPU

---

## Quick Start Summary

After initial setup, here's all you need to do each time:

```cmd
cd "path\to\Insurance Policy Decoder"
venv\Scripts\activate
cd insurance_ai
python app.py
```

Then access the API at: `http://localhost:5000`

---

## Need Help?

If you encounter issues:

1. Make sure `GEMINI_API_KEY` is set in `insurance_ai/.env`
2. Make sure the virtual environment is activated (`(venv)` in prompt)
3. Check that all dependencies installed: `pip install -r requirements.txt`
4. Restart your computer if PATH issues persist

---

*Setup guide version 2.0*