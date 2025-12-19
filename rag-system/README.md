# Physical AI & Humanoid Robotics RAG System

This RAG (Retrieval-Augmented Generation) system powers the interactive Robotics Assistant chatbot in the Physical AI & Humanoid Robotics textbook website. It combines textbook content with OpenAI's language models to provide intelligent, contextual responses.

## Features

- **Dynamic Responses**: Generates detailed answers based on textbook content
- **OpenAI Integration**: Uses your OpenAI API key for enhanced responses
- **Module Coverage**: Covers all textbook modules (ROS 2, Digital Twin, AI-Robot Brain, VLA)
- **Educational Focus**: Maintains technical accuracy and educational tone

## Prerequisites

1. **OpenAI API Key**: You need a valid OpenAI API key
2. **Python 3.8+**: Required for running the RAG system
3. **Node.js**: Required for the Docusaurus frontend (if running locally)

## Setup Instructions

### 1. Configure Environment

Create a `.env` file in the `rag-system` directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Dependencies

```bash
cd rag-system
pip install -r requirements.txt
```

### 3. Run the System

#### Windows:
```bash
start-rag.bat
```

#### Linux/Mac:
```bash
chmod +x start-rag.sh
./start-rag.sh
```

This will:
1. Create and activate a virtual environment
2. Install dependencies
3. Index the textbook content
4. Start the RAG API server on `http://localhost:8000`

## API Endpoints

- `POST /query/` - Query the RAG system (used by the chatbot)
- `POST /rag-query` - Alternative query endpoint with more options
- `POST /chat` - Chat endpoint with conversation history
- `POST /ingest` - Ingest new content into the system
- `GET /health` - Health check endpoint

## Troubleshooting

### Common Issues:

1. **API Key Not Working**:
   - Verify your OpenAI API key is correct in the `.env` file
   - Check that your OpenAI account has sufficient credits

2. **Chatbot Shows Fallback Messages**:
   - Ensure the RAG API server is running
   - Verify the API endpoint in the Docusaurus chatbot component

3. **Content Not Indexed**:
   - Make sure the docs directory contains the textbook content
   - Check that the indexing script completed successfully

### Testing the API:

You can test the API directly:

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is ROS 2?",
    "max_results": 5,
    "grounding_threshold": 0.7
  }'
```

## Docusaurus Integration

The RAG system is integrated with the Docusaurus site through the `RAGChatbot.jsx` component. The chatbot will automatically connect to:

- Production: `https://physical-ai-humanoid-robotics-rag.onrender.com/query/`
- Development: `http://localhost:8000/query/`

## Architecture

```
User Question → RAG System → OpenAI Integration → Contextual Response
     ↓              ↓               ↓                   ↓
  Docusaurus ←  Content DB ←  Vector Store ←  Textbook Content
```

The system retrieves relevant textbook content, combines it with OpenAI's language understanding, and generates educational responses specific to robotics and AI concepts.

## Deployment

For production deployment, the RAG API should be deployed to a cloud service (like Render, Heroku, or AWS) and the Docusaurus site should be configured to point to the deployed API endpoint.