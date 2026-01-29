@echo off
title Insurance Policy Decoder - Server Startup

echo ============================================
echo   Insurance Policy Decoder - Starting...
echo ============================================
echo.

echo [1/2] Starting Ollama AI Engine...
start "Ollama Server" cmd /k "ollama serve"

echo Waiting for Ollama to initialize...
timeout /t 5 /nobreak > nul

echo.
echo [2/2] Starting Flask Server...
echo.

cd /d "%~dp0"
call venv\Scripts\activate
cd insurance_ai

echo ============================================
echo   Server Starting on http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ============================================
echo.

python app.py
