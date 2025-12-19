"""
Script to run the Physical AI & Humanoid Robotics RAG API server
"""
import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    print("Starting Physical AI & Humanoid Robotics RAG API Server...")

    # Load environment variables
    load_dotenv()

    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please make sure your .env file contains: OPENAI_API_KEY=your_actual_api_key")
        print("Create a .env file in the rag-system directory with your API key.")
        return False

    print("OpenAI API key found in environment")

    # Import and start the API directly
    try:
        from api.very_simple_main import load_textbook_content
        print("Loading textbook content...")
        load_textbook_content()
        print("Textbook content loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load textbook content: {e}")
        print("The API will still work but with limited context")

    # Run the uvicorn server
    try:
        print("Starting API server on http://localhost:8000")
        print("Press Ctrl+C to stop the server")

        # Import and run uvicorn directly
        import uvicorn
        from api.very_simple_main import app

        uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return True
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)