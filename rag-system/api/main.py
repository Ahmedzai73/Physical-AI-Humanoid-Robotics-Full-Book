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
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import instructor
from sqlmodel import Session, SQLModel, create_engine, select
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai.api_key = openai_api_key
client = OpenAI(api_key=openai_api_key)

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
        # Return a default response that points to textbook content
        return [{
            "chapter_id": "default",
            "chapter_title": "Physical AI & Humanoid Robotics Overview",
            "content": "Physical AI & Humanoid Robotics is a comprehensive field combining robotics, artificial intelligence, and human-like physical systems. The textbook covers: Module 1: The Robotic Nervous System (ROS 2), Module 2: The Digital Twin (Gazebo & Unity), Module 3: The AI-Robot Brain (NVIDIA Isaac), Module 4: Vision-Language-Action (VLA) systems.",
            "relevance_score": 0.8
        }]


def generate_enhanced_answer(question: str, context: List[Dict[str, Any]]) -> str:
    """Generate a more dynamic answer using OpenAI with enhanced prompting"""
    try:
        # Format context for the LLM with better structure
        formatted_context = []
        for item in context:
            formatted_context.append(f"Chapter: {item['chapter_title']}\nContent: {item['content'][:1500]}...")  # Limit content length but allow more

        context_str = "\n\n".join(formatted_context)

        # Create a more detailed and specific prompt for better textbook-based responses
        enhanced_prompt = f"""
        You are an expert AI assistant for the Physical AI & Humanoid Robotics textbook.
        Your role is to provide accurate, detailed answers based strictly on the textbook content provided in the context.

        Context from textbook:
        {context_str}

        User question: {question}

        Please provide a comprehensive answer based on the textbook content that includes:
        1. Direct references to the concepts mentioned in the context
        2. Technical accuracy in robotics, AI, and related fields
        3. Specific examples or explanations from the textbook when available
        4. If the context doesn't fully answer the question, acknowledge the limitation but suggest where in the textbook the user might find more information (Module 1: ROS 2, Module 2: Digital Twin, Module 3: AI-Robot Brain, Module 4: VLA)
        5. Maintain technical terminology appropriate for robotics/AI education
        6. When discussing NVIDIA Isaac, ROS 2, Gazebo, Unity, or VLA systems, be specific about their implementations and capabilities as described in the textbook
        7. If asked about code examples, provide them in the appropriate language (Python, C++, etc.) as described in the textbook
        8. For robotics concepts, explain both the theory and practical implementation aspects covered in the textbook

        Maintain a helpful, educational tone appropriate for students learning robotics concepts.
        """

        # Call OpenAI API with enhanced prompting
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Using a more cost-effective model while maintaining quality
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. Provide accurate, detailed responses based strictly on the textbook content provided in the context. Maintain technical accuracy and an educational tone. When the context is limited, draw upon general knowledge of robotics, AI, and the textbook's core topics (ROS 2, NVIDIA Isaac, Gazebo, Unity, VLA systems) to provide helpful information."
                },
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=1500,
            temperature=0.3,  # Slightly higher temperature for more varied responses while maintaining accuracy
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating enhanced answer: {e}")
        # Fallback to the original method if enhanced method fails
        return generate_answer(question, context)


def generate_answer(question: str, context: List[Dict[str, Any]]) -> str:
    """Generate answer using OpenAI with retrieved context"""
    try:
        # Format context for the LLM
        context_str = "\n\n".join([f"Chapter: {item['chapter_title']}\nContent: {item['content'][:1000]}" for item in context])

        # Create a detailed prompt for the LLM
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

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. Provide accurate, detailed responses based on textbook content. Maintain technical accuracy and an educational tone."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3,
            top_p=0.9
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I found some information about that topic in the textbook. For detailed answers, please check the relevant module sections (Module 1: ROS 2, Module 2: Digital Twin, Module 3: AI-Robot Brain, Module 4: VLA)."


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
                answer="I couldn't find any relevant information in the book to answer your question. Please check the relevant module sections for more details.",
                sources=[],
                grounding_score=0.0
            )

        # Generate answer using OpenAI with enhanced prompting
        answer = generate_enhanced_answer(request.question, context)

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

def generate_enhanced_answer(question: str, context: List[Dict[str, Any]]) -> str:
    """Generate a more dynamic answer using OpenAI with enhanced prompting"""
    try:
        # Format context for the LLM with better structure
        formatted_context = []
        for item in context:
            formatted_context.append(f"Chapter: {item['chapter_title']}\nContent: {item['content'][:1000]}...")  # Limit content length but allow more

        context_str = "\n\n".join(formatted_context)

        # Create a more detailed and specific prompt for better textbook-based responses
        enhanced_prompt = f"""
        You are an expert AI assistant for the Physical AI & Humanoid Robotics textbook.
        Your role is to provide accurate, detailed answers based on the textbook content provided in the context.

        Context from textbook:
        {context_str}

        User question: {question}

        Please provide a comprehensive answer based on the textbook content that includes:
        1. Direct references to the concepts mentioned in the context
        2. Technical accuracy in robotics, AI, and related fields
        3. Specific examples or explanations from the textbook when available
        4. If the context doesn't fully answer the question, acknowledge the limitation but suggest where in the textbook the user might find more information (Module 1: ROS 2, Module 2: Digital Twin, Module 3: AI-Robot Brain, Module 4: VLA)
        5. Maintain technical terminology appropriate for robotics/AI education

        Maintain a helpful, educational tone appropriate for students learning robotics concepts.
        """

        # Call OpenAI API with enhanced prompting
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Using a more cost-effective model while maintaining quality
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. Provide accurate, detailed responses based on the textbook content provided in the context. Maintain technical accuracy and an educational tone. When the context is limited, draw upon general knowledge of robotics, AI, and the textbook's core topics to provide helpful information."
                },
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=1200,
            temperature=0.3,  # Slightly higher temperature for more varied responses while maintaining accuracy
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating enhanced answer: {e}")
        # Fallback to the original method if enhanced method fails
        return generate_answer(question, context)


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


@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Alternative query endpoint with simplified interface for Docusaurus chatbot"""
    try:
        # Use the same logic as rag-query but with simplified interface
        context = retrieve_context(request.question, request.max_results, request.grounding_threshold)

        if not context:
            return QueryResponse(
                answer="I couldn't find any relevant information in the book to answer your question. Please check the relevant module sections for more details.",
                sources=[],
                grounding_score=0.0
            )

        # Generate enhanced answer using OpenAI
        answer = generate_enhanced_answer(request.question, context)

        # Calculate grounding score
        grounding_score = sum([item["relevance_score"] for item in context]) / len(context) if context else 0.0

        return QueryResponse(
            answer=answer,
            sources=context,
            grounding_score=min(grounding_score, 1.0)
        )
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Physical AI & Humanoid Robotics Book RAG API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)