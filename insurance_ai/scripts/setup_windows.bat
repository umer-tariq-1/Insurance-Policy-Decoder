@echo off
echo ===========================================
echo AI Insurance Policy Decoder - Windows Setup
echo ===========================================

REM 0. Change to project root directory
cd ../../

REM 1. Create virtual environment if not exists
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM 2. Activate venv
echo Activating virtual environment...
call venv\Scripts\activate

REM 3. Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM 4. Install Python dependencies
echo Installing Python dependencies...
pip install -r insurance_ai\requirements.txt

REM 5. Install Tesseract OCR
echo Install Tesseract OCR...
powershell -Command "Start-Process https://github.com/UB-Mannheim/tesseract/wiki -Wait"

echo ================================
echo Setup completed.
echo IMPORTANT:
echo Please install Tesseract from the opened page
echo Default path should be:
echo C:\Program Files\Tesseract-OCR\
echo ================================
pause
