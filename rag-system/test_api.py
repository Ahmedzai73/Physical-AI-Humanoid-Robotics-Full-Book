"""
Test script to verify the RAG API is working properly with OpenAI integration
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from api.main import create_embedding, retrieve_context, generate_enhanced_answer, QueryRequest

def test_openai_connection():
    """Test if OpenAI API key is working properly"""
    try:
        import openai
        from api.main import openai_api_key

        if not openai_api_key:
            print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
            print("Please make sure your .env file contains: OPENAI_API_KEY=your_actual_api_key")
            return False

        # Test embedding creation
        test_text = "This is a test to verify OpenAI API connection."
        embedding = create_embedding(test_text)

        if len(embedding) > 0:
            print("✅ OpenAI API connection successful!")
            print(f"   Embedding vector length: {len(embedding)}")
            return True
        else:
            print("❌ ERROR: OpenAI API returned empty embedding")
            return False

    except Exception as e:
        print(f"❌ ERROR: OpenAI API connection failed - {str(e)}")
        return False

def test_context_retrieval():
    """Test context retrieval (will use fallback if Qdrant not available)"""
    try:
        print("\n🔍 Testing context retrieval...")
        context = retrieve_context("What is Physical AI & Humanoid Robotics?", max_results=3)

        if context and len(context) > 0:
            print(f"✅ Retrieved {len(context)} context items")
            print(f"   First item title: {context[0]['chapter_title']}")
            print(f"   Content preview: {context[0]['content'][:100]}...")
        else:
            print("⚠️  Retrieved context (using fallback content)")

        return True
    except Exception as e:
        print(f"❌ ERROR in context retrieval: {str(e)}")
        return False

def test_answer_generation():
    """Test answer generation with OpenAI"""
    try:
        print("\n🤖 Testing answer generation...")
        context = retrieve_context("What is Physical AI & Humanoid Robotics?")
        answer = generate_enhanced_answer("What is Physical AI & Humanoid Robotics?", context)

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
    print("🧪 Testing RAG API with OpenAI Integration\n")

    # Test OpenAI connection
    openai_ok = test_openai_connection()

    if not openai_ok:
        print("\n❌ Cannot proceed without valid OpenAI API connection")
        return False

    # Test other components
    context_ok = test_context_retrieval()
    answer_ok = test_answer_generation()

    print(f"\n📊 Test Results:")
    print(f"   OpenAI Connection: {'✅' if openai_ok else '❌'}")
    print(f"   Context Retrieval: {'✅' if context_ok else '❌'}")
    print(f"   Answer Generation: {'✅' if answer_ok else '❌'}")

    all_ok = openai_ok and context_ok and answer_ok

    if all_ok:
        print(f"\n🎉 All tests passed! Your RAG system is properly configured.")
        print(f"   The Robotics Assistant chatbot should now provide dynamic, AI-powered responses.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the configuration.")

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)