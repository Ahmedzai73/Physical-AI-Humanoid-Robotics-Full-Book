#!/usr/bin/env python3

"""
Complete VLA System for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module integrates all VLA components into a complete system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import json
import threading
import time
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class VLAState:
    """Data class to hold VLA system state"""
    vision_data: Dict[str, Any] = None
    language_input: str = ""
    action_plan: Dict[str, Any] = None
    system_status: str = "idle"
    confidence: float = 0.0


class VLASystemNode(Node):
    def __init__(self):
        super().__init__('vla_system_node')

        # Initialize VLA state
        self.vla_state = VLAState()

        # Subscribers for all VLA components
        self.speech_sub = self.create_subscription(
            String,
            '/speech/to_text',
            self.speech_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.nlp_intent_sub = self.create_subscription(
            String,
            '/nlp/intent',
            self.nlp_intent_callback,
            10
        )

        self.perception_sub = self.create_subscription(
            String,
            '/perception/scene_description',
            self.perception_callback,
            10
        )

        self.scene_understanding_sub = self.create_subscription(
            String,
            '/scene_understanding/spatial_relationships',
            self.scene_understanding_callback,
            10
        )

        self.skill_feedback_sub = self.create_subscription(
            String,
            '/skill/feedback',
            self.skill_feedback_callback,
            10
        )

        # Publishers for system outputs
        self.command_pub = self.create_publisher(
            String,
            '/vla/command',
            10
        )

        self.text_output_pub = self.create_publisher(
            String,
            '/vla/text_output',
            10
        )

        self.speech_output_pub = self.create_publisher(
            String,
            '/vla/speech_output',
            10
        )

        self.system_status_pub = self.create_publisher(
            String,
            '/vla/system_status',
            10
        )

        self.action_plan_pub = self.create_publisher(
            String,
            '/vla/action_plan',
            10
        )

        self.get_logger().info('VLA System Node initialized')

        # System parameters
        self.system_active = True
        self.processing_lock = threading.Lock()

        # Component readiness flags
        self.vision_ready = False
        self.language_ready = False
        self.action_ready = False

        # Timer for system coordination
        self.coordination_timer = self.create_timer(0.1, self.coordinate_components)

    def speech_callback(self, msg):
        """Handle speech input from user"""
        with self.processing_lock:
            self.vla_state.language_input = msg.data
            self.vla_state.system_status = "processing_language"
            self.get_logger().info(f'VLA received speech: {msg.data}')
            self.process_language_input(msg.data)

    def image_callback(self, msg):
        """Handle image input from camera"""
        with self.processing_lock:
            # Convert image message to simple representation
            self.vla_state.vision_data = {
                'timestamp': msg.header.stamp.sec,
                'encoding': msg.encoding,
                'height': msg.height,
                'width': msg.width
            }
            self.vision_ready = True
            self.vla_state.system_status = "processing_vision"
            self.get_logger().debug('VLA received image data')

    def scan_callback(self, msg):
        """Handle LIDAR scan input"""
        with self.processing_lock:
            # Process scan data if needed
            pass

    def odom_callback(self, msg):
        """Handle odometry input"""
        with self.processing_lock:
            # Process odometry if needed for action planning
            pass

    def nlp_intent_callback(self, msg):
        """Handle NLP intent from language processing"""
        with self.processing_lock:
            try:
                intent_data = json.loads(msg.data)
                self.get_logger().info(f'NLP intent received: {intent_data}')
                # Process the intent and generate action plan
                self.generate_action_plan(intent_data)
            except json.JSONDecodeError:
                self.get_logger().error(f'Invalid NLP intent JSON: {msg.data}')

    def perception_callback(self, msg):
        """Handle perception output"""
        with self.processing_lock:
            self.get_logger().info(f'Perception output: {msg.data}')
            # Update vision data with perception results
            if self.vla_state.vision_data is None:
                self.vla_state.vision_data = {}
            self.vla_state.vision_data['perception'] = msg.data

    def scene_understanding_callback(self, msg):
        """Handle scene understanding output"""
        with self.processing_lock:
            self.get_logger().info(f'Scene understanding: {msg.data}')
            # Update vision data with scene understanding
            if self.vla_state.vision_data is None:
                self.vla_state.vision_data = {}
            self.vla_state.vision_data['scene_understanding'] = msg.data

    def skill_feedback_callback(self, msg):
        """Handle skill execution feedback"""
        with self.processing_lock:
            try:
                feedback_data = json.loads(msg.data)
                self.get_logger().info(f'Skill feedback: {feedback_data}')

                # Update system status based on skill feedback
                if feedback_data.get('status') == 'completed':
                    self.vla_state.system_status = "action_completed"
                    self.publish_system_status(f"Action completed: {feedback_data.get('message', '')}")
                elif feedback_data.get('status') == 'failed':
                    self.vla_state.system_status = "action_failed"
                    self.publish_system_status(f"Action failed: {feedback_data.get('message', '')}")
            except json.JSONDecodeError:
                self.get_logger().error(f'Invalid skill feedback JSON: {msg.data}')

    def process_language_input(self, text: str):
        """Process natural language input through NLP pipeline"""
        # Publish to NLP pipeline
        nlp_msg = String()
        nlp_msg.data = text
        # In a real system, this would go to the NLP pipeline
        # For this example, we'll simulate the NLP processing

    def generate_action_plan(self, intent_data: Dict):
        """Generate action plan based on intent and perception"""
        with self.processing_lock:
            action_plan = {
                'intent': intent_data,
                'vision_context': self.vla_state.vision_data,
                'steps': [],
                'confidence': intent_data.get('confidence', 0.8)
            }

            # Determine action steps based on intent
            command_type = intent_data.get('command_type', 'unknown')

            if command_type == 'navigate':
                action_plan['steps'] = [
                    {'action': 'localize', 'description': 'Determine current position'},
                    {'action': 'plan_path', 'description': 'Plan path to destination'},
                    {'action': 'execute_navigation', 'description': 'Navigate to destination'}
                ]
            elif command_type == 'find':
                action_plan['steps'] = [
                    {'action': 'analyze_scene', 'description': 'Analyze current scene'},
                    {'action': 'search', 'description': 'Search for target object'},
                    {'action': 'report_location', 'description': 'Report object location'}
                ]
            elif command_type == 'grasp':
                action_plan['steps'] = [
                    {'action': 'locate_object', 'description': 'Locate target object'},
                    {'action': 'plan_approach', 'description': 'Plan approach trajectory'},
                    {'action': 'execute_grasp', 'description': 'Execute grasping motion'}
                ]
            else:
                action_plan['steps'] = [
                    {'action': 'interpret_command', 'description': f'Interpret {command_type} command'},
                    {'action': 'plan_execution', 'description': 'Plan execution steps'},
                    {'action': 'execute_action', 'description': f'Execute {command_type} action'}
                ]

            self.vla_state.action_plan = action_plan
            self.vla_state.system_status = "planning_action"

            # Publish action plan
            plan_msg = String()
            plan_msg.data = json.dumps(action_plan)
            self.action_plan_pub.publish(plan_msg)

            # Execute the plan
            self.execute_action_plan(action_plan)

    def execute_action_plan(self, action_plan: Dict):
        """Execute the generated action plan"""
        with self.processing_lock:
            self.vla_state.system_status = "executing_plan"
            self.publish_system_status(f"Executing plan with {len(action_plan['steps'])} steps")

            # For this example, we'll execute steps sequentially
            for step in action_plan['steps']:
                self.get_logger().info(f'Executing step: {step["description"]}')

                # Publish step command
                step_cmd = {
                    'action': step['action'],
                    'description': step['description'],
                    'plan': action_plan
                }

                cmd_msg = String()
                cmd_msg.data = json.dumps(step_cmd)
                self.command_pub.publish(cmd_msg)

                # Simulate step execution time
                time.sleep(0.5)  # In real system, this would be asynchronous

            self.vla_state.system_status = "plan_executed"
            self.publish_system_status("Action plan completed")

    def coordinate_components(self):
        """Main coordination loop for VLA system"""
        with self.processing_lock:
            # Check if all components are ready
            all_ready = self.vision_ready and self.language_ready and self.action_ready

            if all_ready and self.vla_state.system_status == "idle":
                self.vla_state.system_status = "waiting_for_input"
                self.publish_system_status("VLA system ready - waiting for input")

            # Publish current system status
            self.publish_system_status(self.vla_state.system_status)

    def publish_system_status(self, status: str):
        """Publish system status"""
        status_msg = String()
        status_msg.data = status
        self.system_status_pub.publish(status_msg)

        # Log status
        self.get_logger().info(f'VLA System Status: {status}')

    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            'vision_data': self.vla_state.vision_data,
            'language_input': self.vla_state.language_input,
            'action_plan': self.vla_state.action_plan,
            'system_status': self.vla_state.system_status,
            'confidence': self.vla_state.confidence,
            'vision_ready': self.vision_ready,
            'language_ready': self.language_ready,
            'action_ready': self.action_ready
        }

    def reset_system(self):
        """Reset the VLA system to initial state"""
        with self.processing_lock:
            self.vla_state = VLAState()
            self.vision_ready = False
            self.language_ready = False
            self.action_ready = False
            self.vla_state.system_status = "idle"

            self.publish_system_status("VLA system reset")


def main(args=None):
    rclpy.init(args=args)

    vla_system = VLASystemNode()

    try:
        # Example: Simulate a command after startup
        def simulate_command():
            # Simulate user saying "find the red object"
            speech_msg = String()
            speech_msg.data = "find the red object"
            vla_system.speech_callback(speech_msg)

        # Send example command after 3 seconds
        vla_system.create_timer(3.0, simulate_command)

        # Run the VLA system
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()