@echo off
echo ====================================================
echo Starting DocuMind RAG Question Answering System
echo ====================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.9 or higher and try again.
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo [INFO] Creating Python virtual environment venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
)

:: Install dependencies
echo [INFO] Checking/Installing dependencies from requirements.txt...
.\venv\Scripts\python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

echo [INFO] Starting Streamlit Application...
.\venv\Scripts\python -m streamlit run app.py
pause
