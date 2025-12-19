# RAG Chatbot Accuracy Evaluation

This directory contains the evaluation framework for testing the accuracy and grounding of the RAG (Retrieval-Augmented Generation) chatbot system. The evaluation ensures that the chatbot provides accurate, relevant responses based on the Physical AI & Humanoid Robotics textbook content.

## Overview

The evaluation framework tests the RAG system's ability to:
- Retrieve relevant information from the textbook content
- Generate accurate responses grounded in the source material
- Avoid hallucinations and provide factually correct answers
- Maintain high accuracy (90%+ target) across various robotics topics

## Test Structure

The evaluation includes 10 test questions covering key topics from all modules:

1. **ROS 2 fundamentals** - Testing basic robotics middleware concepts
2. **Gazebo simulation** - Testing digital twin and simulation concepts
3. **Isaac Sim and Isaac ROS** - Testing NVIDIA AI-robotics integration
4. **Vision-Language-Action systems** - Testing VLA paradigm understanding
5. **ROS 2 components** - Testing specific technical implementations
6. **Nav2 and navigation** - Testing autonomous navigation concepts
7. **Multimodal fusion** - Testing VLA integration concepts
8. **ROS 2 launch files** - Testing configuration and deployment
9. **Safety considerations** - Testing safety system understanding
10. **URDF fundamentals** - Testing robot modeling concepts

## Accuracy Metrics

The evaluation measures several key metrics:

- **Grounding Accuracy**: Percentage of responses that contain expected keywords from source material
- **Source Retrieval**: Ability to correctly identify and reference source documents
- **Response Time**: Latency for generating responses
- **Hallucination Detection**: Identification of non-factual or fabricated information
- **Confidence Scoring**: Model's confidence in its responses

## Running the Evaluation

### Prerequisites

1. RAG API server must be running
2. Python 3.8+ with required packages:
   - requests
   - asyncio

### Setup

```bash
# Install required packages
pip install requests

# Make sure your RAG API is running
# By default, the test assumes the API is at http://localhost:8000
```

### Execution

```bash
# Run the evaluation against local API
python rag_accuracy_test.py

# To run against a different API endpoint:
API_URL="https://your-deployed-rag-api.com" python rag_accuracy_test.py
```

### Custom API Endpoint

To test against a deployed API, modify the API URL in the test script:

```python
tester = RAGAccuracyTester(api_base_url="https://your-deployed-rag-api.com")
```

## Evaluation Criteria

### Pass Criteria
- **90%+ accuracy rate** across all test questions
- **Low hallucination rate** (minimal fabricated information)
- **Relevant source retrieval** for each response
- **Response time under 10 seconds** per query

### Failure Analysis
The evaluation categorizes failures into:
- **Keyword-based failures**: Responses missing expected technical terms
- **Hallucination failures**: Responses containing non-factual information
- **Other failures**: Technical issues or other problems

## Expected Results

A successful evaluation should show:
- Overall accuracy rate of 90% or higher
- Consistent performance across all modules
- Proper source attribution
- Fast response times (under 5 seconds average)
- Minimal hallucinations

## Integration with Production

The evaluation framework can be integrated into CI/CD pipelines to ensure the deployed RAG system maintains high accuracy standards. The test can be run:

1. Before deploying updates to the RAG system
2. Periodically on the production system
3. After content updates to verify continued accuracy

## Performance Optimization

If accuracy targets are not met, consider:

1. **Improving document indexing** - Better chunking and embedding strategies
2. **Fine-tuning the model** - Using robotics-specific training data
3. **Adjusting retrieval parameters** - Optimizing top_k and similarity thresholds
4. **Enhancing prompts** - Improving the instruction-following capabilities

## Maintenance

The evaluation framework should be updated when:
- New content is added to the textbook
- The RAG system architecture changes
- Accuracy requirements are modified
- New failure modes are identified