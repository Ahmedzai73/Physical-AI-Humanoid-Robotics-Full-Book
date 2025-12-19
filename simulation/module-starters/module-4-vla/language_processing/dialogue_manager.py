#!/usr/bin/env python3

"""
Dialogue Manager for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module manages conversational interactions between human and robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class DialogueState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_FOR_INPUT = "waiting_for_input"
    CONFUSED = "confused"


class DialogueManagerNode(Node):
    def __init__(self):
        super().__init__('dialogue_manager_node')

        # Subscribers
        self.speech_input_sub = self.create_subscription(
            String,
            '/speech/to_text',
            self.speech_input_callback,
            10
        )

        self.command_response_sub = self.create_subscription(
            String,
            '/robot/command_response',
            self.command_response_callback,
            10
        )

        self.perception_response_sub = self.create_subscription(
            String,
            '/perception/query_response',
            self.perception_response_callback,
            10
        )

        # Publishers
        self.text_output_pub = self.create_publisher(
            String,
            '/dialogue/text_output',
            10
        )

        self.speech_output_pub = self.create_publisher(
            String,
            '/dialogue/speech_output',
            10
        )

        self.dialogue_state_pub = self.create_publisher(
            String,
            '/dialogue/state',
            10
        )

        self.conversation_history_pub = self.create_publisher(
            String,
            '/dialogue/conversation_history',
            10
        )

        self.get_logger().info('Dialogue Manager Node initialized')

        # Dialogue state
        self.current_state = DialogueState.IDLE
        self.conversation_history = []
        self.pending_questions = []
        self.user_context = {}
        self.robot_name = "RoboAssistant"

        # Dialogue patterns and responses
        self.greeting_patterns = [
            r'hello', r'hi', r'hey', r'good morning', r'good afternoon', r'good evening'
        ]

        self.farewell_patterns = [
            r'goodbye', r'bye', r'good bye', r'see you', r'see ya', r'farewell'
        ]

        self.question_patterns = [
            r'what', r'how', r'where', r'when', r'who', r'why'
        ]

        # Response templates
        self.greeting_responses = [
            f"Hello! I'm {self.robot_name}, your robotic assistant. How can I help you today?",
            f"Hi there! I'm {self.robot_name}. What would you like me to do?",
            f"Good to see you! I'm ready to assist. What can I help with?"
        ]

        self.farewell_responses = [
            "Goodbye! Have a great day!",
            "See you later! Feel free to call me if you need anything.",
            "Farewell! I'll be here if you need assistance."
        ]

        self.confused_responses = [
            "I'm sorry, I didn't understand that. Could you please rephrase?",
            "I'm not sure I follow. Can you say that again in a different way?",
            "I don't quite understand. Could you be more specific?",
            "I missed that. Can you repeat it please?"
        ]

        self.acknowledgment_responses = [
            "I understand.",
            "Got it.",
            "I see.",
            "Understood.",
            "OK, I'll do that."
        ]

    def speech_input_callback(self, msg):
        """Process incoming speech-to-text input"""
        user_input = msg.data.strip()
        self.get_logger().info(f'User said: {user_input}')

        # Add to conversation history
        self.add_to_conversation("User", user_input)

        # Update dialogue state
        self.current_state = DialogueState.PROCESSING
        self.publish_state()

        # Process the input
        response = self.process_input(user_input)

        # Publish response
        self.publish_response(response)

        # Update state to responding
        self.current_state = DialogueState.RESPONDING
        self.publish_state()

        # After response, go back to listening
        def reset_state():
            self.current_state = DialogueState.LISTENING
            self.publish_state()

        # Use a timer to reset the state after the response is processed
        self.create_timer(2.0, reset_state)

    def command_response_callback(self, msg):
        """Process responses from robot command execution"""
        command_response = msg.data
        self.get_logger().info(f'Command response: {command_response}')

        # Add to conversation history
        self.add_to_conversation("Robot", f"Command executed: {command_response}")

    def perception_response_callback(self, msg):
        """Process responses from perception system"""
        perception_response = msg.data
        self.get_logger().info(f'Perception response: {perception_response}')

        # Add to conversation history
        self.add_to_conversation("Robot", f"Perception: {perception_response}")

    def process_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        user_input_lower = user_input.lower()

        # Check for greetings
        if any(re.search(pattern, user_input_lower) for pattern in self.greeting_patterns):
            return random.choice(self.greeting_responses)

        # Check for farewells
        if any(re.search(pattern, user_input_lower) for pattern in self.farewell_patterns):
            return random.choice(self.farewell_responses)

        # Check if it's a question
        if any(re.search(r'\b' + pattern + r'\b', user_input_lower) for pattern in self.question_patterns):
            return self.handle_question(user_input_lower)

        # Check for command-like phrases
        if self.is_command_like(user_input_lower):
            return self.handle_command_request(user_input_lower)

        # Default response
        return random.choice(self.acknowledgment_responses)

    def handle_question(self, user_input: str) -> str:
        """Handle questions from the user"""
        # Identify question type
        if any(word in user_input for word in ['where', 'location', 'find', 'see']):
            return self.handle_location_question(user_input)
        elif any(word in user_input for word in ['what', 'describe', 'tell me about']):
            return self.handle_description_question(user_input)
        elif any(word in user_input for word in ['time', 'date', 'day']):
            return self.handle_time_question()
        elif any(word in user_input for word in ['name', 'who are you', 'what are you']):
            return self.handle_identity_question()
        else:
            # For other questions, delegate to perception or other systems
            return self.delegate_question(user_input)

    def handle_location_question(self, user_input: str) -> str:
        """Handle questions about object locations"""
        # Extract object from question
        object_match = re.search(r'where is (?:the )?(\w+)', user_input)
        if object_match:
            target_object = object_match.group(1)
            return f"I'll check where the {target_object} is. Let me look around and report back."
        else:
            return "I can help you find objects. Can you specify what you're looking for?"

    def handle_description_question(self, user_input: str) -> str:
        """Handle questions asking for scene description"""
        return "I'm analyzing the scene around me. Let me describe what I see."

    def handle_time_question(self) -> str:
        """Handle questions about time"""
        now = datetime.now()
        return f"The current time is {now.strftime('%H:%M')} on {now.strftime('%B %d, %Y')}."

    def handle_identity_question(self) -> str:
        """Handle questions about the robot's identity"""
        return f"I'm {self.robot_name}, an AI-powered robot assistant designed to help with everyday tasks. I can navigate, recognize objects, and follow your commands."

    def handle_command_request(self, user_input: str) -> str:
        """Handle requests that seem like commands"""
        # Acknowledge the command and pass to command interpretation
        return f"I understand you'd like me to perform a task. I'm processing your request: '{user_input}'"

    def delegate_question(self, user_input: str) -> str:
        """Delegate complex questions to appropriate systems"""
        # For this example, we'll just acknowledge
        # In a real system, this would route to perception, navigation, etc.
        return f"That's an interesting question: '{user_input}'. Let me see what I can find out."

    def is_command_like(self, text: str) -> bool:
        """Check if text seems like a command"""
        command_indicators = [
            'go', 'move', 'navigate', 'find', 'look', 'search', 'pick', 'grasp', 'take',
            'bring', 'get', 'follow', 'stop', 'wait', 'come', 'show', 'tell'
        ]

        return any(indicator in text for indicator in command_indicators)

    def publish_response(self, response: str):
        """Publish response to both text and speech outputs"""
        # Publish text response
        text_msg = String()
        text_msg.data = response
        self.text_output_pub.publish(text_msg)

        # Publish speech response (same as text for this example)
        speech_msg = String()
        speech_msg.data = response
        self.speech_output_pub.publish(speech_msg)

        # Add response to conversation history
        self.add_to_conversation("Robot", response)

        self.get_logger().info(f'Robot response: {response}')

    def add_to_conversation(self, speaker: str, message: str):
        """Add message to conversation history"""
        timestamp = self.get_clock().now().to_msg().sec
        entry = {
            'timestamp': timestamp,
            'speaker': speaker,
            'message': message
        }
        self.conversation_history.append(entry)

        # Keep only recent history (last 20 entries)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        # Publish updated conversation history
        history_msg = String()
        history_msg.data = json.dumps(self.conversation_history)
        self.conversation_history_pub.publish(history_msg)

    def publish_state(self):
        """Publish current dialogue state"""
        state_msg = String()
        state_msg.data = self.current_state.value
        self.dialogue_state_pub.publish(state_msg)

    def get_recent_context(self, n: int = 3) -> List[Dict]:
        """Get recent conversation context"""
        return self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.pending_questions = []
        self.user_context = {}

        # Publish empty history
        history_msg = String()
        history_msg.data = json.dumps(self.conversation_history)
        self.conversation_history_pub.publish(history_msg)

        self.get_logger().info('Conversation history reset')


def main(args=None):
    rclpy.init(args=args)

    dialogue_node = DialogueManagerNode()

    try:
        # Example: Simulate a greeting after startup
        def send_example_greeting():
            greeting_msg = String()
            greeting_msg.data = "Hello robot"
            dialogue_node.speech_input_callback(greeting_msg)

        # Send example after 2 seconds
        dialogue_node.create_timer(2.0, send_example_greeting)

        rclpy.spin(dialogue_node)
    except KeyboardInterrupt:
        pass
    finally:
        dialogue_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()