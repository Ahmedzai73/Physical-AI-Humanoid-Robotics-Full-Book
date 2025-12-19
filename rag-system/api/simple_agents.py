"""
Simple OpenAI Agent for the Physical AI & Humanoid Robotics RAG System
This module provides a simplified agent implementation without complex dependencies
"""
import os
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for the OpenAI Agent"""
    api_key: str = Field(..., description="OpenAI API Key")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.3, description="Temperature for response generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens for response")


class SimpleOpenAIAgent:
    """Simple OpenAI Agent for the Physical AI & Humanoid Robotics RAG System"""

    def __init__(self, config: AgentConfig):
        self.config = config
        openai.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

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

    def query(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query using the OpenAI agent with optional context

        Args:
            question: The user's question
            context: Optional context to provide to the agent

        Returns:
            Dictionary with the agent's answer and metadata
        """
        try:
            # Construct the full prompt with context
            if context:
                user_message = f"""
                Context from the Physical AI & Humanoid Robotics textbook:
                {context}

                User Query: {question}

                Please provide a comprehensive answer based on the textbook content.
                If the context doesn't contain sufficient information to answer the query,
                acknowledge this limitation and suggest where the user might find the information
                in the textbook modules.
                """
            else:
                user_message = f"""
                User Query: {question}

                Please provide a comprehensive answer about Physical AI & Humanoid Robotics.
                Focus on textbook-relevant topics such as ROS 2, Digital Twins, NVIDIA Isaac,
                Vision-Language-Action systems, and related robotics concepts.
                """

            # Make the API call to OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Extract the response
            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": [],
                "model_used": self.model,
                "confidence": 0.9
            }

        except Exception as e:
            logger.error(f"Error in OpenAI agent query: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "model_used": self.model,
                "confidence": 0.0
            }

# Global agent instance
simple_agent_instance: Optional[SimpleOpenAIAgent] = None


def initialize_agent(config: AgentConfig) -> SimpleOpenAIAgent:
    """
    Initialize the global agent instance with the provided configuration

    Args:
        config: Agent configuration

    Returns:
        Initialized SimpleOpenAIAgent instance
    """
    global simple_agent_instance
    simple_agent_instance = SimpleOpenAIAgent(config)
    return simple_agent_instance


def get_agent() -> Optional[SimpleOpenAIAgent]:
    """
    Get the global agent instance

    Returns:
        SimpleOpenAIAgent instance if initialized, None otherwise
    """
    return simple_agent_instance