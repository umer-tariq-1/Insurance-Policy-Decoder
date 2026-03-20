# How to Run the Project

Quick reference for starting the Insurance Policy Decoder after setup is complete.

---

## Daily Startup Steps

### Step 1: Start the Flask Server

Open **Command Prompt** and run:

```cmd
cd "C:\path\to\Insurance Policy Decoder"
venv\Scripts\activate
cd insurance_ai
python app.py
```

> Replace `C:\path\to\Insurance Policy Decoder` with your actual project path.

You should see:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

That's it — the server is ready. Gemini API calls go out over the internet, so no other service needs to be running.

---

### Step 2: Verify Everything is Working

Open your browser and go to:

- **Server Status:** http://localhost:5000/health

You should see `{"status": "Flask server running"}`.

---

## Quick Commands Reference

| Action                       | Command                          |
| ---------------------------- | -------------------------------- |
| Activate virtual environment | `venv\Scripts\activate`          |
| Run Flask server             | `python app.py`                  |
| Stop server                  | Press `Ctrl + C`                 |
| Deactivate venv              | `deactivate`                     |

---

## API Base URL

Once running, the API is available at:

```
http://localhost:5000
```

### Primary Endpoints

| Feature        | Endpoint                    |
| -------------- | --------------------------- |
| Upload file    | `POST /upload`              |
| Summary        | `POST /gemini-api-summary`  |
| Q&A            | `POST /gemini-api-qa`       |
| Compare        | `POST /gemini-api-compare`  |
| Risk Score     | `POST /gemini-risk-score`   |

See **[api-documentation.md](api-documentation.md)** for the full API reference.

---

## Optional: Start Ollama (Local Fallback Only)

Ollama is **not needed** for normal use. Start it only if you want to use the `/local-summary`, `/local-qa`, or `/compare` fallback routes.

Open a separate **Command Prompt** and run:

```cmd
ollama serve
```

Keep that window open while using the local routes. Check status at: http://localhost:5000/ollama/status

---

## Shutting Down

1. Press `Ctrl + C` in the Flask server window
2. Type `deactivate` to exit the virtual environment (optional)
3. If Ollama was started, press `Ctrl + C` in that window too

---

## Troubleshooting Quick Fixes

| Problem                          | Solution                                                          |
| -------------------------------- | ----------------------------------------------------------------- |
| "API key not configured" error   | Add `GEMINI_API_KEY=your_key` to `insurance_ai/.env`             |
| "Module not found" error         | Make sure venv is activated (you should see `(venv)` in prompt)  |
| Port 5000 in use                 | Close other applications or use `python app.py --port 5001`      |
| Server won't start               | Check if you're in the `insurance_ai` folder                     |
| "Ollama not running" error       | Only affects `/local-*` routes — use `/gemini-*` instead         |

---

## Double-Click Start (Easy)

⚠️ Only follow this after you have done with setup.md.

Just double-click `start-server.bat` to start everything!

---

_Happy coding!_