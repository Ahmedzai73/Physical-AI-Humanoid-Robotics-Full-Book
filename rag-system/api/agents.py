"""
OpenAI Agent / ChatKit SDK Configuration for the Physical AI & Humanoid Robotics RAG System

This module provides the agent implementation for the RAG system that powers
the Physical AI & Humanoid Robotics textbook Q&A functionality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from .main import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for the OpenAI Agent"""
    api_key: str = Field(..., description="OpenAI API Key")
    model: str = Field(default="gpt-4-turbo", description="OpenAI model to use")
    temperature: float = Field(default=0.3, description="Temperature for response generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for response")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class OpenAIAgent:
    """OpenAI Agent for the Physical AI & Humanoid Robotics RAG System"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout
        )
        self.system_prompt = """
        You are an expert assistant for the Physical AI & Humanoid Robotics textbook.
        You help students and researchers understand concepts related to:
        - ROS 2 (Robot Operating System 2) and its ecosystem
        - Digital Twin systems using Gazebo and Unity
        - NVIDIA Isaac Sim and Isaac ROS for AI-powered robotics
        - Vision-Language-Action (VLA) systems
        - Robot navigation, perception, and control
        - Simulation environments and real-world robot deployment

        Provide accurate, detailed responses based on the textbook content.
        When referencing specific modules or chapters, clearly indicate:
        - Module 1: The Robotic Nervous System (ROS 2)
        - Module 2: The Digital Twin (Gazebo & Unity)
        - Module 3: The AI-Robot Brain (NVIDIA Isaac™)
        - Module 4: Vision-Language-Action (VLA)

        Use technical terminology appropriately but explain complex concepts clearly.
        If asked about code examples, provide them in the appropriate language (Python, C++, etc.).
        """

    async def query(self, query_request: QueryRequest, context_documents: List[Dict]) -> QueryResponse:
        """
        Process a query using the OpenAI agent with RAG context

        Args:
            query_request: The user's query request (from main.py)
            context_documents: Retrieved documents to provide context

        Returns:
            QueryResponse with the agent's answer and metadata (from main.py)
        """
        try:
            # Format context from retrieved documents
            context_text = self._format_context(context_documents)

            # For compatibility with main.py QueryRequest, use question instead of query
            user_question = getattr(query_request, 'question', getattr(query_request, 'query', 'No question provided'))

            # Construct the full prompt with enhanced context and instructions
            user_message = f"""
            Context from the Physical AI & Humanoid Robotics textbook:
            {context_text}

            User Query: {user_question}

            Please provide a comprehensive, technically accurate answer based on the textbook content that:
            1. Directly addresses the question with specific information from the context
            2. Includes relevant technical terminology and concepts from the textbook
            3. References specific modules when applicable (Module 1: ROS 2, Module 2: Digital Twin,
               Module 3: AI-Robot Brain, Module 4: VLA)
            4. Maintains an educational tone appropriate for students learning robotics
            5. If the context doesn't contain sufficient information, acknowledge this limitation
               but suggest where the user might find the information in the textbook modules
            6. Provide practical examples or code snippets when relevant to the question
            """

            # Make the API call to OpenAI with enhanced parameters
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )

            # Extract the response
            answer = response.choices[0].message.content

            # Extract document references with enhanced metadata
            referenced_docs = []
            for doc in context_documents:
                metadata = doc.get('metadata', {})
                # Include content snippets for better context
                content_snippet = doc.get('content', '')[:300] + "..." if doc.get('content') else ""
                referenced_docs.append({
                    'metadata': metadata,
                    'content_snippet': content_snippet,
                    'relevance_score': doc.get('relevance_score', 0.0)
                })

            # For compatibility with main.py QueryResponse, return answer and sources
            # Note: main.py QueryResponse has answer, sources, grounding_score
            return QueryResponse(
                answer=answer,
                sources=referenced_docs,
                grounding_score=min(0.9, len(referenced_docs) * 0.2)  # Calculate a basic grounding score
            )

        except Exception as e:
            logger.error(f"Error in OpenAI agent query: {str(e)}")
            raise

    def _format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents into context text

        Args:
            documents: List of retrieved documents with content and metadata

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found in the textbook."

        formatted_docs = []
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            doc_text = f"Document {i+1}:\n"
            doc_text += f"Content: {content}\n"

            if metadata:
                doc_text += f"Metadata: {metadata}\n"

            doc_text += "---\n"
            formatted_docs.append(doc_text)

        return "\n".join(formatted_docs)

    async def validate_connection(self) -> bool:
        """
        Validate the connection to the OpenAI API

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Make a simple test call to validate the API key
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Test connection"}
                ],
                temperature=0.0,
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False


class ChatKitAgent:
    """
    ChatKit-style agent implementation for the Physical AI & Humanoid Robotics textbook

    This provides additional chat capabilities beyond the basic OpenAI agent
    """

    def __init__(self, openai_agent: OpenAIAgent):
        self.openai_agent = openai_agent
        self.conversation_history = []

    async def chat(self, query_request: QueryRequest, context_documents: List[Dict]) -> QueryResponse:
        """
        Process a chat query with conversation history context

        Args:
            query_request: The user's query request (from main.py)
            context_documents: Retrieved documents to provide context

        Returns:
            QueryResponse with the agent's answer and metadata (from main.py)
        """
        # For compatibility with main.py QueryRequest, use question instead of query
        user_question = getattr(query_request, 'question', getattr(query_request, 'query', 'No question provided'))

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_question,
            "timestamp": asyncio.get_event_loop().time()
        })

        # Limit conversation history to prevent token overflow
        recent_history = self.conversation_history[-10:]  # Keep last 10 exchanges

        # Create enhanced prompt with conversation history
        history_context = "\n".join([
            f"{item['role'].title()}: {item['content']}"
            for item in recent_history
        ])

        # Include the history in the query
        # Create a new request with conversation history
        enhanced_query_text = f"Previous conversation:\n{history_context}\n\nCurrent query: {user_question}"

        # Since we can't create QueryRequest with the new structure, we'll pass the enhanced text
        # Create a temporary request object with the enhanced query
        class TempQueryRequest:
            def __init__(self, question, **kwargs):
                self.question = question
                # Add any other attributes that might be needed
                for key, value in kwargs.items():
                    setattr(self, key, value)

        enhanced_request = TempQueryRequest(question=enhanced_query_text)

        # Use the OpenAI agent to process the query
        response = await self.openai_agent.query(enhanced_request, context_documents)

        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.answer,
            "timestamp": asyncio.get_event_loop().time()
        })

        return response

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []


# Global agent instance - to be initialized with config
agent_instance: Optional[ChatKitAgent] = None


def initialize_agent(config: AgentConfig) -> ChatKitAgent:
    """
    Initialize the global agent instance with the provided configuration

    Args:
        config: Agent configuration

    Returns:
        Initialized ChatKitAgent instance
    """
    global agent_instance
    openai_agent = OpenAIAgent(config)
    agent_instance = ChatKitAgent(openai_agent)
    return agent_instance


def get_agent() -> Optional[ChatKitAgent]:
    """
    Get the global agent instance

    Returns:
        ChatKitAgent instance if initialized, None otherwise
    """
    return agent_instance