"""
Physical AI & Humanoid Robotics Book RAG API
FastAPI backend for the RAG system that powers the book's chatbot
"""
import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import instructor
from sqlmodel import Session, SQLModel, create_engine, select
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Database setup
DATABASE_URL = os.getenv("NEON_DATABASE_URL", "sqlite:///./book_rag.db")
engine = create_engine(DATABASE_URL, echo=True)

# Qdrant client setup
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Collection name for book content
COLLECTION_NAME = "book_content"

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None
    max_results: Optional[int] = 5
    grounding_threshold: Optional[float] = 0.7


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
    grounding_threshold: Optional[float] = 0.7


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    grounding_score: float


# Initialize the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing RAG system...")

    # Create Qdrant collection if it doesn't exist
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' already exists")
    except:
        logger.info(f"Creating collection '{COLLECTION_NAME}'")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )

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


def create_embedding(text: str) -> List[float]:
    """Create embedding for text using OpenAI API"""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


def retrieve_context(question: str, max_results: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Retrieve relevant context from the vector store"""
    try:
        query_embedding = create_embedding(question)

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=max_results,
            score_threshold=threshold
        )

        results = []
        for result in search_results:
            results.append({
                "chapter_id": result.payload.get("chapter_id"),
                "chapter_title": result.payload.get("chapter_title"),
                "content": result.payload.get("content"),
                "relevance_score": result.score
            })

        return results
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []


def generate_answer(question: str, context: List[Dict[str, Any]]) -> str:
    """Generate answer using OpenAI with retrieved context"""
    try:
        # Format context for the LLM
        context_str = "\n\n".join([f"Chapter: {item['chapter_title']}\nContent: {item['content']}" for item in context])

        # Create a prompt for the LLM
        prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics book.
        Answer the following question based on the provided context from the book.
        If the answer is not available in the context, say "I don't have enough information from the book to answer this question."

        Context:
        {context_str}

        Question: {question}

        Answer:
        """

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert assistant for the Physical AI & Humanoid Robotics book. Provide accurate answers based on the book content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-12-16"}


@app.post("/embed", response_model=Dict[str, Any])
async def embed_text(text: Dict[str, str]):
    """Create embedding for a given text"""
    try:
        embedding = create_embedding(text["text"])
        return {"embedding": embedding, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(e)}")


@app.post("/search", response_model=List[Dict[str, Any]])
async def search(query: Dict[str, Any]):
    """Search for relevant content in the vector store"""
    try:
        question = query.get("question", "")
        max_results = query.get("max_results", 5)
        threshold = query.get("threshold", 0.7)

        results = retrieve_context(question, max_results, threshold)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.post("/rag-query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """Query the RAG system with a question"""
    try:
        # Retrieve relevant context
        context = retrieve_context(request.question, request.max_results, request.grounding_threshold)

        if not context:
            return QueryResponse(
                answer="I couldn't find any relevant information in the book to answer your question.",
                sources=[],
                grounding_score=0.0
            )

        # Generate answer
        answer = generate_answer(request.question, context)

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
        # In a real implementation, you would process the entire conversation history
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1].content
        context = retrieve_context(last_message, request.max_results, request.grounding_threshold)

        if not context:
            return ChatResponse(
                response="I couldn't find any relevant information in the book to answer your question.",
                sources=[],
                grounding_score=0.0
            )

        # Generate answer based on context and conversation history
        context_str = "\n\n".join([f"Chapter: {item['chapter_title']}\nContent: {item['content']}" for item in context])

        # Create a prompt that includes conversation history
        conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[:-1]])

        prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics book.
        Answer the following question based on the provided context from the book and conversation history.
        If the answer is not available in the context, say "I don't have enough information from the book to answer this question."

        Conversation History:
        {conversation_history}

        Context:
        {context_str}

        User's latest question: {last_message}

        Answer:
        """

        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert assistant for the Physical AI & Humanoid Robotics book. Provide accurate answers based on the book content and maintain conversation context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
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


@app.post("/ingest", response_model=IngestResponse)
async def ingest_content(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest new content into the RAG system"""
    try:
        # In a real implementation, we would chunk the content properly
        # For now, we'll treat the entire content as one chunk
        content = request.content
        chapter_id = request.chapter_id
        chapter_title = request.chapter_title
        metadata = request.metadata or {}

        # Create embedding for the content
        embedding = create_embedding(content)

        # Store in Qdrant
        point_id = f"{chapter_id}_{hash(content) % 1000000}"  # Simple ID generation

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "chapter_id": chapter_id,
                        "chapter_title": chapter_title,
                        "metadata": metadata
                    }
                )
            ]
        )

        logger.info(f"Ingested content for chapter: {chapter_id}")

        return IngestResponse(
            success=True,
            chunk_count=1,
            chunk_ids=[point_id]
        )
    except Exception as e:
        logger.error(f"Error ingesting content: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting content: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Physical AI & Humanoid Robotics Book RAG API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)