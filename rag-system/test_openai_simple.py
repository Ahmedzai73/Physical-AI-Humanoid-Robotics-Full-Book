"""
Simple test script to verify the OpenAI API is working with basic functionality
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test if OpenAI API key is working properly"""
    try:
        import openai
        from openai import OpenAI

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            print("Please make sure your .env file contains: OPENAI_API_KEY=your_actual_api_key")
            return False

        # Create a client and test with a simple call
        client = OpenAI(api_key=openai_api_key)

        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello, this is a test to verify the API connection. Please respond with 'API connection successful.'"}
            ],
            max_tokens=20,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        print(f"SUCCESS: OpenAI API connection successful!")
        print(f"   Test response: {result}")
        return True

    except Exception as e:
        print(f"ERROR: OpenAI API connection failed - {str(e)}")
        return False

def test_content_loading():
    """Test if textbook content can be loaded"""
    try:
        print("\nTesting textbook content loading...")

        # Import the simple main to access the content loading function
        from api.very_simple_main import load_textbook_content

        # Try to load content
        load_textbook_content()

        print("SUCCESS: Textbook content loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR loading content: {str(e)}")
        return False

def test_simple_search():
    """Test the simple search functionality"""
    try:
        print("\nTesting simple search functionality...")

        # Import the search function
        from api.very_simple_main import simple_search

        # Test search
        results = simple_search("What is ROS 2?", max_results=3)

        print(f"SUCCESS: Simple search functionality working")
        print(f"   Found {len(results)} results for 'What is ROS 2?'")
        if results:
            print(f"   First result preview: {results[0]['content'][:100]}...")

        return True
    except Exception as e:
        print(f"ERROR in search functionality: {str(e)}")
        return False

def test_full_query():
    """Test the full query functionality"""
    try:
        print("\nTesting full query functionality...")

        # Import the query function
        from api.very_simple_main import simple_search, generate_answer_with_openai

        # Test search and generation
        context = simple_search("What is Physical AI & Humanoid Robotics?", max_results=2)
        answer = generate_answer_with_openai("What is Physical AI & Humanoid Robotics?", context)

        if answer and len(answer) > 10:
            print(f"SUCCESS: Full query functionality working")
            print(f"   Generated answer preview: {answer[:200]}...")
            return True
        else:
            print("ERROR: Generated answer is empty or too short")
            return False
    except Exception as e:
        print(f"ERROR in full query functionality: {str(e)}")
        return False

def main():
    print("Testing Simple RAG API with OpenAI Integration\n")

    # Test OpenAI connection
    openai_ok = test_openai_connection()

    if not openai_ok:
        print("\nERROR: Cannot proceed without valid OpenAI API connection")
        print("Make sure your .env file contains a valid OPENAI_API_KEY")
        return False

    # Test other components
    content_ok = test_content_loading()
    search_ok = test_simple_search()
    query_ok = test_full_query()

    print(f"\nTest Results:")
    print(f"   OpenAI Connection: {'PASS' if openai_ok else 'FAIL'}")
    print(f"   Content Loading: {'PASS' if content_ok else 'FAIL'}")
    print(f"   Simple Search: {'PASS' if search_ok else 'FAIL'}")
    print(f"   Full Query: {'PASS' if query_ok else 'FAIL'}")

    all_ok = openai_ok and content_ok and search_ok and query_ok

    if all_ok:
        print(f"\nSUCCESS: All tests passed! Your simplified RAG system is properly configured.")
        print(f"   The Robotics Assistant chatbot should now provide dynamic, AI-powered responses.")
        print(f"   To run the server, use: uvicorn api.very_simple_main:app --host 0.0.0.0 --port 8000")
    else:
        print(f"\nWARNING: Some tests failed. Please check the configuration.")

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)