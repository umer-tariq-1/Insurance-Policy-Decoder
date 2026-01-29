# Insurance Policy Decoder - Setup Guide

This guide will help you set up and run the Insurance Policy Decoder application on your Windows machine.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Install Python](#step-1-install-python)
3. [Step 2: Install Ollama (Local AI)](#step-2-install-ollama-local-ai)
4. [Step 3: Download AI Model](#step-3-download-ai-model)
5. [Step 4: Set Up the Project](#step-4-set-up-the-project)
6. [Step 5: Configure Environment Variables](#step-5-configure-environment-variables)
7. [Step 6: Install Tesseract OCR (Optional)](#step-6-install-tesseract-ocr-optional)
8. [Step 7: Run the Application](#step-7-run-the-application)
9. [Verifying the Setup](#verifying-the-setup)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, make sure you have:

- **Windows 10/11** (64-bit)
- **8GB RAM minimum** (16GB recommended)
- **10GB free disk space** (for AI models and dependencies)
- **Internet connection** (for initial setup)

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

## Step 2: Install Ollama (Local AI)

Ollama is the local AI engine that powers the summary, Q&A, and comparison features.

### Download Ollama

1. Go to [ollama.ai/download](https://ollama.ai/download)
2. Click **"Download for Windows"**
3. Run the installer (`OllamaSetup.exe`)
4. Follow the installation wizard (default settings are fine)

### Verify Ollama Installation

After installation, open a **new Command Prompt** and run:

```cmd
ollama --version
```

You should see the version number.

---

## Step 3: Download AI Model

Now you need to download the AI model that will analyze insurance documents.

### Start Ollama Service

First, start the Ollama service by running:

```cmd
ollama serve
```

> **Note:** Keep this window open! Ollama needs to be running for the application to work.

### Download the Model

Open a **new Command Prompt window** (keep the previous one running) and run:

```cmd
ollama pull llama3.2:3b
```

This will download the Llama 3.2 3B model (~2GB download). Wait for it to complete.

> **Tip:** If you have a powerful GPU with 8GB+ VRAM, you can use a larger model for better results:
> ```cmd
> ollama pull llama3.2:8b
> ```

### Verify Model Download

```cmd
ollama list
```

You should see `llama3.2:3b` in the list.

---

## Step 4: Set Up the Project

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

This will install all required Python packages. This may take several minutes as it downloads AI/ML libraries.

---

## Step 5: Configure Environment Variables

### Create Environment File

In the `insurance_ai` folder, there should be a file named `.env`. If not, create it:

1. Open Notepad
2. Add the following content:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Save as `.env` (make sure it's not saved as `.env.txt`)

### Get Gemini API Key (Optional but Recommended)

The Gemini API provides an additional cloud-based summary option.

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key
5. Paste it in your `.env` file:

```env
GEMINI_API_KEY=AIzaSyD...your_actual_key...
```

> **Note:** If you don't set up Gemini, the `/gemini-api-summary` route won't work, but all other routes will function normally with Ollama.

---

## Step 6: Install Tesseract OCR (Optional)

Tesseract is needed if you want to process scanned PDF documents (PDFs that are images rather than text).

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

## Step 7: Run the Application

### Step 7a: Start Ollama (if not already running)

Open a Command Prompt and run:

```cmd
ollama serve
```

Keep this window open.

### Step 7b: Start the Flask Server

Open a **new Command Prompt**, navigate to the project, and run:

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

### Check Ollama Connection

Go to:

```
http://localhost:5000/ollama/status
```

You should see:
```json
{"available": true, "message": "OK"}
```

### Test with a Sample Document

You can test the API using tools like:
- **Postman** (download from [postman.com](https://www.postman.com/downloads/))
- **curl** (if installed)
- **Thunder Client** (VS Code extension)

---

## Troubleshooting

### "Python is not recognized"

- Python was not added to PATH during installation
- Solution: Reinstall Python and check "Add Python to PATH"

### "ollama is not recognized"

- Ollama was not installed correctly or PATH not updated
- Solution: Restart your computer after installing Ollama

### "Ollama not running" error in the app

- The Ollama service is not started
- Solution: Run `ollama serve` in a separate Command Prompt

### "Model not found" error

- The AI model was not downloaded
- Solution: Run `ollama pull llama3.2:3b`

### Installation takes too long

- The ML libraries (torch, transformers) are large
- This is normal, the first installation can take 10-20 minutes

### "CUDA out of memory" error

- Your GPU doesn't have enough memory
- Solution: The app will automatically use CPU if GPU memory is insufficient

### Port 5000 already in use

- Another application is using port 5000
- Solution: Close that application or run Flask on a different port:
  ```cmd
  python app.py --port 5001
  ```

---

## Quick Start Summary

After initial setup, here's what you need to do each time:

1. **Open Command Prompt #1:**
   ```cmd
   ollama serve
   ```

2. **Open Command Prompt #2:**
   ```cmd
   cd "path\to\Insurance Policy Decoder"
   venv\Scripts\activate
   cd insurance_ai
   python app.py
   ```

3. **Access the API at:** `http://localhost:5000`

---

## Need Help?

If you encounter issues:

1. Make sure all prerequisites are installed
2. Restart your computer (helps with PATH issues)
3. Check that Ollama is running (`ollama serve`)
4. Verify the virtual environment is activated (`(venv)` should appear in prompt)

---

*Setup guide version 1.0*
