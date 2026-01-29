# How to Run the Project

Quick reference for starting the Insurance Policy Decoder after setup is complete.

---

## Daily Startup Steps

### Step 1: Start Ollama (AI Engine)

Open **Command Prompt** and run:

```cmd
ollama serve
```

> Keep this window open while using the application.

---

### Step 2: Start the Flask Server

Open a **new Command Prompt** window and run these commands:

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

---

### Step 3: Verify Everything is Working

Open your browser and go to:

- **Server Status:** http://localhost:5000/health
- **AI Status:** http://localhost:5000/ollama/status

Both should return success responses.

---

## Quick Commands Reference

| Action                       | Command                 |
| ---------------------------- | ----------------------- |
| Start Ollama                 | `ollama serve`          |
| Activate virtual environment | `venv\Scripts\activate` |
| Run Flask server             | `python app.py`         |
| Stop server                  | Press `Ctrl + C`        |
| Deactivate venv              | `deactivate`            |

---

## API Base URL

Once running, the API is available at:

```
http://localhost:5000
```

---

## Shutting Down

1. Press `Ctrl + C` in the Flask server window
2. Press `Ctrl + C` in the Ollama window (or just close it)
3. Type `deactivate` to exit the virtual environment (optional)

---

## Troubleshooting Quick Fixes

| Problem                    | Solution                                                        |
| -------------------------- | --------------------------------------------------------------- |
| "Ollama not running" error | Make sure `ollama serve` is running in another window           |
| "Module not found" error   | Make sure venv is activated (you should see `(venv)` in prompt) |
| Port 5000 in use           | Close other applications or use `python app.py --port 5001`     |
| Server won't start         | Check if you're in the `insurance_ai` folder                    |

---

## Double-Click Start (Easy)

⚠️ Only follow this after you have done with setup.md.

Just double-click `start-server.bat` to start everything!

---

_Happy coding!_
