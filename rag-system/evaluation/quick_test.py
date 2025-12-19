"""
Quick test script to validate RAG system meets 90%+ accuracy requirement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_accuracy_test import RAGAccuracyTester

def run_quick_accuracy_test(api_url="http://localhost:8000"):
    """
    Run a quick accuracy test to validate the RAG system
    """
    print("Running quick RAG accuracy validation test...")
    print(f"Testing API endpoint: {api_url}")

    # Initialize tester
    tester = RAGAccuracyTester(api_base_url=api_url)

    # Run all tests
    results = tester.run_all_tests()

    # Generate report
    report = tester.generate_report(results)

    # Print summary
    accuracy_rate = report["summary"]["accuracy_percentage"]
    meets_target = report["summary"]["meets_target"]

    print(f"Accuracy Rate: {accuracy_rate:.2f}%")
    print(f"Target: 90%+")
    print(f"Status: {'✓ PASS' if meets_target else '✗ FAIL'}")

    if meets_target:
        print("\n🎉 SUCCESS: RAG system meets 90%+ accuracy requirement!")
        return True
    else:
        print(f"\n⚠️  FAIL: RAG system needs improvement. Current accuracy: {accuracy_rate:.2f}%")
        print(f"   Shortfall: {90.0 - accuracy_rate:.2f} percentage points")
        return False

if __name__ == "__main__":
    # Use production API URL when deployed
    production_api = "https://physical-ai-humanoid-robotics-rag.onrender.com"

    print("Testing with production API URL...")
    success = run_quick_accuracy_test(production_api)

    if not success:
        print("\nNote: In a real deployment, you would need to:")
        print("1. Ensure your RAG backend is properly deployed and accessible")
        print("2. Verify the API endpoint returns accurate, grounded responses")
        print("3. Retrain or adjust your RAG system if accuracy targets aren't met")

    # Exit with appropriate code
    sys.exit(0 if success else 1)