@echo off
echo Starting Insurance AI Server...

REM 
cd ../..

REM Activate virtual environment
call venv\Scripts\activate

REM Run Flask app
python insurance_ai\app.py

pause
