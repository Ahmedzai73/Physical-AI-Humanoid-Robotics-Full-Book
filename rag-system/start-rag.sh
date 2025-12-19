#!/bin/bash
# Startup script for the Physical AI & Humanoid Robotics RAG system

# Set the working directory
cd "$(dirname "$0")"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the content indexing script first
echo "Indexing textbook content..."
if [ -f "index_textbook_content.py" ]; then
    python index_textbook_content.py
else
    echo "Indexing script not found, using fallback..."
    python ingestion/index-all-content.py
fi

# Start the RAG API server
echo "Starting RAG API server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload