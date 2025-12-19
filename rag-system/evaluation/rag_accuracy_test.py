"""
RAG Chatbot Grounding Accuracy Test
Module 4: Vision-Language-Action (VLA) - Physical AI & Humanoid Robotics Textbook

This script evaluates the RAG system's grounding accuracy by testing its ability
to provide accurate, relevant responses based on the textbook content.
"""

import asyncio
import json
import time
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin


@dataclass
class TestResult:
    query: str
    expected_sources: List[str]
    response: str
    retrieved_sources: List[str]
    is_accurate: bool
    confidence_score: float
    response_time: float
    details: str = ""


class RAGAccuracyTester:
    """
    Tests the accuracy and grounding of the RAG system
    """
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.query_endpoint = urljoin(self.api_base_url, "/query/")

        # Test questions with expected answers and sources
        self.test_questions = [
            {
                "query": "What is the main purpose of ROS 2 in robotics?",
                "expected_sources": ["module-1-ros", "ros2-architecture"],
                "expected_keywords": ["middleware", "DDS", "communication", "nodes", "topics"]
            },
            {
                "query": "How does Gazebo simulation benefit robotics development?",
                "expected_sources": ["module-2-digital-twin", "gazebo"],
                "expected_keywords": ["simulation", "physics", "testing", "safety", "development"]
            },
            {
                "query": "What is Isaac Sim and how is it used in robotics?",
                "expected_sources": ["module-3-ai-robot-brain", "isaac-sim"],
                "expected_keywords": ["photorealistic", "simulation", "synthetic", "data", "training"]
            },
            {
                "query": "Explain the Vision-Language-Action (VLA) paradigm",
                "expected_sources": ["module-4-vla", "vla-introduction"],
                "expected_keywords": ["integration", "multimodal", "perception", "action", "language"]
            },
            {
                "query": "What are the key components of a ROS 2 node?",
                "expected_sources": ["module-1-ros", "nodes"],
                "expected_keywords": ["publisher", "subscriber", "service", "client", "executor"]
            },
            {
                "query": "How does Nav2 integrate with Isaac ROS?",
                "expected_sources": ["module-3-ai-robot-brain", "nav2-integration"],
                "expected_keywords": ["navigation", "path", "planning", "localization", "mapping"]
            },
            {
                "query": "What is the difference between early and late fusion in VLA systems?",
                "expected_sources": ["module-4-vla", "multimodal-perception"],
                "expected_keywords": ["early", "late", "fusion", "integration", "modalities"]
            },
            {
                "query": "How do you create a launch file in ROS 2?",
                "expected_sources": ["module-1-ros", "launch-files"],
                "expected_keywords": ["launch", "python", "xml", "configuration", "startup"]
            },
            {
                "query": "What safety considerations are important for LLM-controlled robots?",
                "expected_sources": ["simulation", "safety-guardrails"],
                "expected_keywords": ["safety", "guardrails", "emergency", "stop", "collision"]
            },
            {
                "query": "Explain the role of URDF in robotics",
                "expected_sources": ["module-1-ros", "urdf-fundamentals"],
                "expected_keywords": ["URDF", "robot", "description", "format", "modeling"]
            }
        ]

    def call_rag_api(self, query: str) -> Dict:
        """
        Call the RAG API with the given query
        """
        try:
            payload = {
                "query": query,
                "top_k": 5,
                "return_sources": True
            }

            response = requests.post(
                self.query_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API call failed with status {response.status_code}: {response.text}")
                return {"response": "Error: API call failed", "sources": []}

        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return {"response": "Error: Request failed", "sources": []}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"response": "Error: Unexpected error", "sources": []}

    def evaluate_response_accuracy(self, query: str, response: str, expected_keywords: List[str]) -> Tuple[bool, float, str]:
        """
        Evaluate if the response contains the expected information
        """
        response_lower = response.lower()

        # Count how many expected keywords are present in the response
        found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
        keyword_score = len(found_keywords) / len(expected_keywords)

        # Additional checks for accuracy
        details = []

        if keyword_score >= 0.7:  # At least 70% of expected keywords found
            is_accurate = True
            details.append(f"Found {len(found_keywords)}/{len(expected_keywords)} expected keywords")
        else:
            is_accurate = False
            details.append(f"Only found {len(found_keywords)}/{len(expected_keywords)} expected keywords")

        # Check for hallucinations (common indicators)
        hallucination_indicators = [
            "i don't know", "i'm not sure", "possibly", "might be",
            "i think", "probably", "could be", "not mentioned in the text"
        ]

        hallucination_count = sum(1 for indicator in hallucination_indicators if indicator in response_lower)
        if hallucination_count > 0:
            is_accurate = False
            details.append(f"Detected {hallucination_count} potential hallucination indicators")

        return is_accurate, keyword_score, "; ".join(details)

    def run_single_test(self, test_case: Dict) -> TestResult:
        """
        Run a single test case
        """
        query = test_case["query"]
        expected_sources = test_case["expected_sources"]
        expected_keywords = test_case["expected_keywords"]

        start_time = time.time()

        # Call the RAG API
        api_response = self.call_rag_api(query)

        response_time = time.time() - start_time

        response_text = api_response.get("response", api_response.get("answer", ""))
        retrieved_sources = [str(src) for src in api_response.get("sources", [])]

        # Evaluate accuracy
        is_accurate, confidence_score, details = self.evaluate_response_accuracy(
            query, response_text, expected_keywords
        )

        return TestResult(
            query=query,
            expected_sources=expected_sources,
            response=response_text,
            retrieved_sources=retrieved_sources,
            is_accurate=is_accurate,
            confidence_score=confidence_score,
            response_time=response_time,
            details=details
        )

    def run_all_tests(self) -> List[TestResult]:
        """
        Run all test cases and return results
        """
        print("Starting RAG Chatbot accuracy tests...")
        print(f"Testing against API endpoint: {self.query_endpoint}")
        print(f"Total test cases: {len(self.test_questions)}")
        print("-" * 80)

        results = []
        accurate_count = 0

        for i, test_case in enumerate(self.test_questions, 1):
            print(f"Running test {i}/{len(self.test_questions)}: {test_case['query'][:50]}...")

            result = self.run_single_test(test_case)
            results.append(result)

            if result.is_accurate:
                accurate_count += 1

            print(f"  Status: {'✓ PASS' if result.is_accurate else '✗ FAIL'}")
            print(f"  Confidence: {result.confidence_score:.2f}")
            print(f"  Response time: {result.response_time:.2f}s")
            if result.details:
                print(f"  Details: {result.details}")
            print()

        return results

    def generate_report(self, results: List[TestResult]) -> Dict:
        """
        Generate a comprehensive report of the test results
        """
        total_tests = len(results)
        accurate_tests = sum(1 for r in results if r.is_accurate)
        accuracy_rate = accurate_tests / total_tests if total_tests > 0 else 0

        avg_response_time = sum(r.response_time for r in results) / total_tests if total_tests > 0 else 0
        avg_confidence = sum(r.confidence_score for r in results) / total_tests if total_tests > 0 else 0

        # Categorize failures
        keyword_failures = sum(1 for r in results if "expected keywords" in r.details and not r.is_accurate)
        hallucination_failures = sum(1 for r in results if "hallucination" in r.details and not r.is_accurate)

        report = {
            "summary": {
                "total_tests": total_tests,
                "accurate_tests": accurate_tests,
                "accuracy_rate": accuracy_rate,
                "accuracy_percentage": accuracy_rate * 100,
                "target_accuracy": 90.0,  # 90% target
                "meets_target": accuracy_rate >= 0.90,
                "avg_response_time": avg_response_time,
                "avg_confidence": avg_confidence
            },
            "breakdown": {
                "keyword_failures": keyword_failures,
                "hallucination_failures": hallucination_failures,
                "other_failures": total_tests - accurate_tests - keyword_failures - hallucination_failures
            },
            "results": [
                {
                    "query": r.query,
                    "is_accurate": r.is_accurate,
                    "confidence_score": r.confidence_score,
                    "response_time": r.response_time,
                    "details": r.details
                }
                for r in results
            ]
        }

        return report

    def print_detailed_report(self, report: Dict):
        """
        Print a detailed report to console
        """
        summary = report["summary"]

        print("=" * 80)
        print("RAG CHATBOT ACCURACY EVALUATION REPORT")
        print("=" * 80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Accurate Responses: {summary['accurate_tests']}")
        print(f"Accuracy Rate: {summary['accuracy_percentage']:.2f}%")
        print(f"Target Accuracy: {summary['target_accuracy']:.2f}%")
        print(f"Meets Target: {'✓ YES' if summary['meets_target'] else '✗ NO'}")
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"Average Confidence: {summary['avg_confidence']:.2f}")
        print()

        print("FAILURE BREAKDOWN:")
        breakdown = report["breakdown"]
        print(f"  Keyword-based failures: {breakdown['keyword_failures']}")
        print(f"  Hallucination failures: {breakdown['hallucination_failures']}")
        print(f"  Other failures: {breakdown['other_failures']}")
        print()

        print("DETAILED RESULTS:")
        for i, result in enumerate(report["results"], 1):
            status = "✓ PASS" if result["is_accurate"] else "✗ FAIL"
            print(f"{i:2d}. {status} - {result['query'][:60]}...")
            print(f"    Confidence: {result['confidence_score']:.2f}, "
                  f"Time: {result['response_time']:.2f}s")
            if result["details"]:
                print(f"    Details: {result['details']}")
            print()

        print("=" * 80)

        if summary['meets_target']:
            print("🎉 SUCCESS: RAG system meets the 90%+ accuracy target!")
        else:
            print("⚠️  IMPROVEMENT NEEDED: RAG system does not meet the 90%+ accuracy target.")
            print(f"   Shortfall: {90.0 - summary['accuracy_percentage']:.2f} percentage points")


def main():
    # Initialize tester - in a real deployment, you would use the production API URL
    # For this test, we'll use localhost, but in practice you'd use your deployed endpoint
    tester = RAGAccuracyTester(api_base_url="http://localhost:8000")  # Replace with production URL when deployed

    # Run all tests
    results = tester.run_all_tests()

    # Generate and print report
    report = tester.generate_report(results)
    tester.print_detailed_report(report)

    # Return success/failure based on target
    return report["summary"]["meets_target"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)