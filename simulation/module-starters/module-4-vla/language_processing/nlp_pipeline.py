#!/usr/bin/env python3

"""
NLP Pipeline for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module demonstrates natural language processing for the VLA system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import json
import re
from typing import Dict, List, Tuple
import spacy
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    FOLLOW = "follow"
    FIND = "find"
    GREET = "greet"
    STOP = "stop"
    EXPLORE = "explore"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    command_type: CommandType
    target_object: str = None
    target_location: str = None
    confidence: float = 0.0
    raw_command: str = ""


class NLPPipelineNode(Node):
    def __init__(self):
        super().__init__('nlp_pipeline_node')

        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/nlp/raw_command',
            self.command_callback,
            10
        )

        # Publishers
        self.parsed_command_pub = self.create_publisher(
            String,
            '/nlp/parsed_command',
            10
        )

        self.intent_pub = self.create_publisher(
            String,
            '/nlp/intent',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/nlp/status',
            10
        )

        self.get_logger().info('NLP Pipeline Node initialized')

        # Command patterns and keywords
        self.command_keywords = {
            CommandType.NAVIGATE: [
                'go to', 'navigate to', 'move to', 'travel to', 'walk to', 'head to',
                'go', 'move', 'navigate', 'travel', 'walk', 'head', 'approach'
            ],
            CommandType.GRASP: [
                'pick up', 'grasp', 'grab', 'take', 'lift', 'hold', 'get',
                'pick', 'collect', 'fetch', 'retrieve'
            ],
            CommandType.FOLLOW: [
                'follow', 'accompany', 'go with', 'accompany me', 'follow me'
            ],
            CommandType.FIND: [
                'find', 'look for', 'search for', 'locate', 'where is', 'find me',
                'show me', 'point to', 'identify'
            ],
            CommandType.GREET: [
                'hello', 'hi', 'greet', 'say hello', 'wave to', 'introduce yourself'
            ],
            CommandType.STOP: [
                'stop', 'halt', 'pause', 'freeze', 'cease', 'quit'
            ],
            CommandType.EXPLORE: [
                'explore', 'look around', 'scan', 'investigate', 'examine', 'wander'
            ]
        }

        # Location keywords
        self.location_keywords = [
            'kitchen', 'bedroom', 'living room', 'office', 'bathroom', 'hallway',
            'entrance', 'exit', 'door', 'window', 'table', 'chair', 'couch',
            'here', 'there', 'nearby', 'front', 'back', 'left', 'right'
        ]

        # Object keywords (colors and basic objects)
        self.object_keywords = [
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'white', 'black',
            'cup', 'bottle', 'book', 'phone', 'keys', 'ball', 'box', 'object',
            'person', 'human', 'robot', 'toy', 'fruit', 'apple', 'banana'
        ]

        # Initialize spacy (if available) or use simple methods
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except OSError:
            self.get_logger().warn("spaCy model not available, using simple parsing")
            self.spacy_available = False

    def command_callback(self, msg):
        """Process incoming natural language command"""
        raw_command = msg.data.strip().lower()
        self.get_logger().info(f'Received command: {raw_command}')

        # Parse the command
        parsed_command = self.parse_command(raw_command)

        # Publish parsed command
        parsed_msg = String()
        parsed_msg.data = json.dumps({
            'command_type': parsed_command.command_type.value,
            'target_object': parsed_command.target_object,
            'target_location': parsed_command.target_location,
            'confidence': parsed_command.confidence,
            'raw_command': parsed_command.raw_command
        })
        self.parsed_command_pub.publish(parsed_msg)

        # Publish intent
        intent_msg = String()
        intent_msg.data = f"{parsed_command.command_type.value}: {parsed_command.target_object or 'none'} -> {parsed_command.target_location or 'none'}"
        self.intent_pub.publish(intent_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Parsed command: {parsed_command.command_type.value} with confidence {parsed_command.confidence:.2f}"
        self.status_pub.publish(status_msg)

        self.get_logger().debug(f'Command parsed: {parsed_command.command_type.value}')

    def parse_command(self, command: str) -> ParsedCommand:
        """Parse a natural language command into structured format"""
        # Identify command type
        command_type = self.identify_command_type(command)

        # Extract target object
        target_object = self.extract_target_object(command)

        # Extract target location
        target_location = self.extract_target_location(command)

        # Calculate confidence based on keyword matches
        confidence = self.calculate_confidence(command, command_type, target_object, target_location)

        return ParsedCommand(
            command_type=command_type,
            target_object=target_object,
            target_location=target_location,
            confidence=confidence,
            raw_command=command
        )

    def identify_command_type(self, command: str) -> CommandType:
        """Identify the type of command from the text"""
        # Count matches for each command type
        type_scores = {}

        for cmd_type, keywords in self.command_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of the keyword
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', command))
                score += matches

            type_scores[cmd_type] = score

        # Find the command type with the highest score
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]

        # If no keywords matched, return unknown
        if best_score == 0:
            return CommandType.UNKNOWN

        return best_type

    def extract_target_object(self, command: str) -> str:
        """Extract target object from command"""
        # Look for object keywords in the command
        found_objects = []

        for obj in self.object_keywords:
            if re.search(r'\b' + re.escape(obj) + r'\b', command):
                found_objects.append(obj)

        # Return the most relevant object (for now, just the first one found)
        if found_objects:
            # Prioritize colored objects
            colored_objects = [obj for obj in found_objects if obj in ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'white', 'black']]
            if colored_objects:
                # Look for colored objects followed by object types
                for color in colored_objects:
                    # Find the next word after the color
                    pattern = r'\b' + color + r'\s+(\w+)'
                    match = re.search(pattern, command)
                    if match:
                        return f"{color} {match.group(1)}"
                return colored_objects[0]  # Just return the color if no object follows

            # Return the first non-color object found
            non_color_objects = [obj for obj in found_objects if obj not in ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'white', 'black']]
            if non_color_objects:
                return non_color_objects[0]

        # If no keywords matched, try to extract any noun phrases (simple approach)
        # Look for patterns like "the <adjective> <noun>" or "<adjective> <noun>"
        pattern = r'(?:the\s+|a\s+|an\s+)?(\w+\s+\w+|\w+)'
        matches = re.findall(pattern, command)

        for match in matches:
            # Filter out common non-objects
            if match not in ['the', 'a', 'an', 'to', 'me', 'it', 'you', 'we', 'us', 'here', 'there', 'now', 'then']:
                return match

        return None

    def extract_target_location(self, command: str) -> str:
        """Extract target location from command"""
        found_locations = []

        for loc in self.location_keywords:
            if re.search(r'\b' + re.escape(loc) + r'\b', command):
                found_locations.append(loc)

        # Return the most relevant location
        if found_locations:
            # Prioritize room names over furniture
            room_names = ['kitchen', 'bedroom', 'living room', 'office', 'bathroom', 'hallway', 'entrance', 'exit']
            room_matches = [loc for loc in found_locations if loc in room_names]

            if room_matches:
                return room_matches[0]

            return found_locations[0]

        # Look for directional words
        directionals = ['front', 'back', 'left', 'right', 'nearby', 'here', 'there']
        for directional in directionals:
            if re.search(r'\b' + re.escape(directional) + r'\b', command):
                return directional

        return None

    def calculate_confidence(self, command: str, cmd_type: CommandType, target_obj: str, target_loc: str) -> float:
        """Calculate confidence in the parsing result"""
        confidence = 0.0

        # Base confidence on command type identification
        if cmd_type != CommandType.UNKNOWN:
            confidence += 0.4

        # Boost confidence if we found a target object
        if target_obj:
            confidence += 0.3

        # Boost confidence if we found a target location
        if target_loc:
            confidence += 0.2

        # Boost confidence if both object and location are found
        if target_obj and target_loc:
            confidence += 0.1

        # Ensure confidence is between 0 and 1
        return min(1.0, confidence)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove punctuation (optional, depending on the use case)
        # text = re.sub(r'[^\w\s]', ' ', text)

        return text


def main(args=None):
    rclpy.init(args=args)

    nlp_node = NLPPipelineNode()

    try:
        rclpy.spin(nlp_node)
    except KeyboardInterrupt:
        pass
    finally:
        nlp_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()