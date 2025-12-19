@echo off
REM Startup script for the Physical AI & Humanoid Robotics RAG system

REM Set the working directory
cd /d "%~dp0"

REM Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the content indexing script first
echo Indexing textbook content...
if exist "index_textbook_content.py" (
    python index_textbook_content.py
) else (
    echo Indexing script not found, using fallback...
    python ingestion/index-all-content.py
)

REM Start the RAG API server
echo Starting RAG API server...
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload