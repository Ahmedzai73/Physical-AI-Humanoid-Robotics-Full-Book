"""
Simplified Physical AI & Humanoid Robotics Book RAG API
FastAPI backend for the RAG system that powers the book's chatbot
This version uses a simple in-memory search instead of Qdrant for easier setup
"""
import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine
from contextlib import asynccontextmanager
import glob
import markdown
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai.api_key = openai_api_key

# Database setup
DATABASE_URL = os.getenv("NEON_DATABASE_URL", "sqlite:///./book_rag.db")
engine = create_engine(DATABASE_URL, echo=True)

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5
    grounding_threshold: Optional[float] = 0.3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    grounding_score: float


class IngestRequest(BaseModel):
    content: str
    chapter_id: str
    chapter_title: str
    metadata: Optional[Dict[str, Any]] = {}


class IngestResponse(BaseModel):
    success: bool
    chunk_count: int
    chunk_ids: List[str]

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_results: Optional[int] = 5
    grounding_threshold: Optional[float] = 0.3


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    grounding_score: float

# Global variables for the in-memory search
textbook_content = []
vectorizer = None
tfidf_matrix = None

def load_textbook_content():
    """Load textbook content from the docs directory"""
    global textbook_content, vectorizer, tfidf_matrix

    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')

    if not os.path.exists(docs_dir):
        logger.warning(f"Docs directory not found: {docs_dir}")
        return

    # Find all markdown files
    md_files = glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)

    textbook_content = []
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract text from markdown
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            plain_text = soup.get_text(separator=' ', strip=True)

            # Create chunks of the content
            chunks = create_chunks(plain_text, file_path)

            for i, chunk in enumerate(chunks):
                textbook_content.append({
                    'id': f"{file_path.replace(os.sep, '_').replace('.md', '')}_chunk_{i}",
                    'content': chunk,
                    'file_path': file_path,
                    'title': os.path.basename(file_path).replace('.md', '').replace('_', ' ')
                })
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Loaded {len(textbook_content)} content chunks from textbook")

    # Create TF-IDF vectorizer
    if textbook_content:
        documents = [item['content'] for item in textbook_content]
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        logger.info("TF-IDF matrix created for content search")


def create_chunks(text: str, file_path: str, chunk_size: int = 500) -> List[str]:
    """Create chunks of text for processing"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks


def search_content(query: str, max_results: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Search for relevant content using TF-IDF and cosine similarity"""
    global vectorizer, tfidf_matrix, textbook_content

    if not vectorizer or not tfidf_matrix or not textbook_content:
        # Return default content if no textbook content loaded
        return [{
            "chapter_id": "default",
            "chapter_title": "Physical AI & Humanoid Robotics Overview",
            "content": "Physical AI & Humanoid Robotics is a comprehensive field combining robotics, artificial intelligence, and human-like physical systems. The textbook covers: Module 1: The Robotic Nervous System (ROS 2), Module 2: The Digital Twin (Gazebo & Unity), Module 3: The AI-Robot Brain (NVIDIA Isaac), Module 4: Vision-Language-Action (VLA) systems.",
            "relevance_score": 0.8
        }]

    try:
        # Transform the query using the fitted vectorizer
        query_vector = vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top results
        top_indices = similarities.argsort()[-max_results:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append({
                    'chapter_id': textbook_content[idx]['id'],
                    'chapter_title': textbook_content[idx]['title'],
                    'content': textbook_content[idx]['content'],
                    'relevance_score': float(similarities[idx])
                })

        return results
    except Exception as e:
        logger.error(f"Error in search_content: {e}")
        # Return default content on error
        return [{
            "chapter_id": "default",
            "chapter_title": "Physical AI & Humanoid Robotics Overview",
            "content": "Physical AI & Humanoid Robotics is a comprehensive field combining robotics, artificial intelligence, and human-like physical systems. The textbook covers: Module 1: The Robotic Nervous System (ROS 2), Module 2: The Digital Twin (Gazebo & Unity), Module 3: The AI-Robot Brain (NVIDIA Isaac), Module 4: Vision-Language-Action (VLA) systems.",
            "relevance_score": 0.8
        }]


def generate_answer_with_openai(question: str, context: List[Dict[str, Any]]) -> str:
    """Generate an answer using OpenAI with the provided context"""
    try:
        # Format context for the LLM
        context_str = "\n\n".join([f"Source: {item['chapter_title']}\nContent: {item['content'][:1000]}..." for item in context])

        # Create a detailed prompt for OpenAI
        prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics textbook.
        Answer the following question based on the provided context from the textbook.

        Context:
        {context_str}

        Question: {question}

        Please provide a comprehensive, technically accurate answer based on the textbook content that:
        1. Directly addresses the question with specific information from the context
        2. Includes relevant technical terminology and concepts from the textbook
        3. References specific modules when applicable (Module 1: ROS 2, Module 2: Digital Twin,
           Module 3: AI-Robot Brain, Module 4: VLA)
        4. Maintains an educational tone appropriate for students learning robotics
        5. If the context doesn't contain sufficient information, acknowledge this limitation
           but suggest where the user might find the information in the textbook modules
        6. Provide practical examples or code snippets when relevant to the question

        Answer:
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Using a more cost-effective model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. Provide accurate, detailed responses based on textbook content. Maintain technical accuracy and an educational tone."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1200,
            temperature=0.3,
            top_p=0.9
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer with OpenAI: {e}")
        return "I found some information about that topic in the textbook. For detailed answers, please check the relevant module sections (Module 1: ROS 2, Module 2: Digital Twin, Module 3: AI-Robot Brain, Module 4: VLA)."


# Initialize the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing RAG system...")

    # Load textbook content
    load_textbook_content()

    # Create database tables
    SQLModel.metadata.create_all(engine)

    yield

    # Shutdown
    logger.info("Shutting down RAG system...")


app = FastAPI(
    title="Physical AI & Humanoid Robotics Book RAG API",
    description="API for the RAG (Retrieval-Augmented Generation) system that powers the book's chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, configure this properly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-12-16"}


@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system with a question - simplified version for chatbot"""
    try:
        # Retrieve relevant context
        context = search_content(request.question, request.max_results, request.grounding_threshold)

        if not context:
            return QueryResponse(
                answer="I couldn't find any relevant information in the book to answer your question. Please check the relevant module sections for more details.",
                sources=[],
                grounding_score=0.0
            )

        # Generate answer using OpenAI
        answer = generate_answer_with_openai(request.question, context)

        # Calculate grounding score (average of relevance scores)
        grounding_score = sum([item["relevance_score"] for item in context]) / len(context) if context else 0.0

        return QueryResponse(
            answer=answer,
            sources=context,
            grounding_score=min(grounding_score, 1.0)  # Ensure score is between 0 and 1
        )
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/rag-query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """Query the RAG system with a question"""
    try:
        # Retrieve relevant context
        context = search_content(request.question, request.max_results, request.grounding_threshold)

        if not context:
            return QueryResponse(
                answer="I couldn't find any relevant information in the book to answer your question. Please check the relevant module sections for more details.",
                sources=[],
                grounding_score=0.0
            )

        # Generate answer using OpenAI
        answer = generate_answer_with_openai(request.question, context)

        # Calculate grounding score (average of relevance scores)
        grounding_score = sum([item["relevance_score"] for item in context]) / len(context) if context else 0.0

        return QueryResponse(
            answer=answer,
            sources=context,
            grounding_score=min(grounding_score, 1.0)  # Ensure score is between 0 and 1
        )
    except Exception as e:
        logger.error(f"Error in rag-query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with conversation history"""
    try:
        # For simplicity, we'll just use the last message as the question
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1].content
        context = search_content(last_message, request.max_results, request.grounding_threshold)

        if not context:
            return ChatResponse(
                response="I couldn't find any relevant information in the book to answer your question. Please check the relevant module sections for more details.",
                sources=[],
                grounding_score=0.0
            )

        # Generate answer based on context and conversation history
        context_str = "\n\n".join([f"Source: {item['chapter_title']}\nContent: {item['content'][:1000]}..." for item in context])

        # Create a prompt that includes conversation history
        conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[:-1]])

        prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics book.
        Answer the following question based on the provided context from the book and conversation history.

        Conversation History:
        {conversation_history}

        Context:
        {context_str}

        User's latest question: {last_message}

        Please provide a comprehensive, technically accurate answer based on the textbook content that:
        1. Directly addresses the question with specific information from the context
        2. Includes relevant technical terminology and concepts from the textbook
        3. References specific modules when applicable (Module 1: ROS 2, Module 2: Digital Twin,
           Module 3: AI-Robot Brain, Module 4: VLA)
        4. Maintains an educational tone appropriate for students learning robotics
        5. If the context doesn't contain sufficient information, acknowledge this limitation
           but suggest where the user might find the information in the textbook modules
        6. Provide practical examples or code snippets when relevant to the question

        Answer:
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. Provide accurate answers based on the book content and maintain conversation context. Maintain technical accuracy and an educational tone."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1200,
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
        grounding_score = sum([item["relevance_score"] for item in context]) / len(context) if context else 0.0

        return ChatResponse(
            response=answer,
            sources=context,
            grounding_score=min(grounding_score, 1.0)
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Physical AI & Humanoid Robotics Book RAG API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)