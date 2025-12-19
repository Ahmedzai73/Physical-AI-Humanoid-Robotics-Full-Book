"""
Simple test script to verify the RAG API is working properly with OpenAI integration
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

def test_openai_connection():
    """Test if OpenAI API key is working properly"""
    try:
        import openai
        from openai import OpenAI

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
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
        print(f"✅ OpenAI API connection successful!")
        print(f"   Test response: {result}")
        return True

    except Exception as e:
        print(f"❌ ERROR: OpenAI API connection failed - {str(e)}")
        return False

def test_content_loading():
    """Test if textbook content can be loaded"""
    try:
        print("\n📚 Testing textbook content loading...")

        # Import the simple main to access the content loading function
        from api.simple_main import load_textbook_content

        # Try to load content
        load_textbook_content()

        print("✅ Textbook content loaded successfully")
        return True
    except Exception as e:
        print(f"❌ ERROR loading content: {str(e)}")
        return False

def test_search_functionality():
    """Test the search functionality"""
    try:
        print("\n🔍 Testing search functionality...")

        # Import the search function
        from api.simple_main import search_content

        # Test search
        results = search_content("What is ROS 2?", max_results=3)

        print(f"✅ Search functionality working")
        print(f"   Found {len(results)} results for 'What is ROS 2?'")
        if results:
            print(f"   First result preview: {results[0]['content'][:100]}...")

        return True
    except Exception as e:
        print(f"❌ ERROR in search functionality: {str(e)}")
        return False

def test_answer_generation():
    """Test answer generation with OpenAI"""
    try:
        print("\n🤖 Testing answer generation...")

        # Import necessary functions
        from api.simple_main import search_content, generate_answer_with_openai

        # Get context
        context = search_content("What is Physical AI & Humanoid Robotics?", max_results=2)
        answer = generate_answer_with_openai("What is Physical AI & Humanoid Robotics?", context)

        if answer and len(answer) > 10:
            print(f"✅ Generated answer successfully")
            print(f"   Answer preview: {answer[:200]}...")
            return True
        else:
            print("❌ ERROR: Generated answer is empty or too short")
            return False
    except Exception as e:
        print(f"❌ ERROR in answer generation: {str(e)}")
        return False

def main():
    print("🧪 Testing Simple RAG API with OpenAI Integration\n")

    # Test OpenAI connection
    openai_ok = test_openai_connection()

    if not openai_ok:
        print("\n❌ Cannot proceed without valid OpenAI API connection")
        print("Make sure your .env file contains a valid OPENAI_API_KEY")
        return False

    # Test other components
    content_ok = test_content_loading()
    search_ok = test_search_functionality()
    answer_ok = test_answer_generation()

    print(f"\n📊 Test Results:")
    print(f"   OpenAI Connection: {'✅' if openai_ok else '❌'}")
    print(f"   Content Loading: {'✅' if content_ok else '❌'}")
    print(f"   Search Function: {'✅' if search_ok else '❌'}")
    print(f"   Answer Generation: {'✅' if answer_ok else '❌'}")

    all_ok = openai_ok and content_ok and search_ok and answer_ok

    if all_ok:
        print(f"\n🎉 All tests passed! Your simplified RAG system is properly configured.")
        print(f"   The Robotics Assistant chatbot should now provide dynamic, AI-powered responses.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the configuration.")

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)