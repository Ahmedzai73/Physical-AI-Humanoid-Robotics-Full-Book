#!/usr/bin/env python3

"""
Command Interpretation Module for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module interprets natural language commands and translates them into robot actions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, Point
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image, LaserScan
import json
import math
from typing import Dict, List, Optional
from enum import Enum


class InterpretationState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_PERCEPTION = "waiting_for_perception"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


class CommandInterpretationNode(Node):
    def __init__(self):
        super().__init__('command_interpretation_node')

        # Subscribers
        self.parsed_command_sub = self.create_subscription(
            String,
            '/nlp/parsed_command',
            self.parsed_command_callback,
            10
        )

        self.scene_description_sub = self.create_subscription(
            String,
            '/perception/scene_description',
            self.scene_description_callback,
            10
        )

        self.spatial_relationships_sub = self.create_subscription(
            String,
            '/scene_understanding/spatial_relationships',
            self.spatial_relationships_callback,
            10
        )

        # Publishers
        self.robot_command_pub = self.create_publisher(
            String,
            '/robot/command',
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.interpretation_status_pub = self.create_publisher(
            String,
            '/command_interpretation/status',
            10
        )

        self.action_plan_pub = self.create_publisher(
            String,
            '/command_interpretation/action_plan',
            10
        )

        # Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('Command Interpretation Node initialized')

        # Internal state
        self.current_state = InterpretationState.IDLE
        self.current_parsed_command = None
        self.scene_description = ""
        self.spatial_relationships = []
        self.object_locations = {}

        # Navigation parameters
        self.navigation_timeout = 30.0  # seconds

    def parsed_command_callback(self, msg):
        """Process parsed command from NLP pipeline"""
        try:
            parsed_data = json.loads(msg.data)
            self.current_parsed_command = {
                'command_type': parsed_data['command_type'],
                'target_object': parsed_data['target_object'],
                'target_location': parsed_data['target_location'],
                'confidence': parsed_data['confidence'],
                'raw_command': parsed_data['raw_command']
            }

            self.get_logger().info(f'Received parsed command: {self.current_parsed_command["command_type"]}')

            # Process the command
            self.process_command()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to decode parsed command: {e}')

    def scene_description_callback(self, msg):
        """Update with scene description from perception"""
        self.scene_description = msg.data

    def spatial_relationships_callback(self, msg):
        """Update with spatial relationships from scene understanding"""
        try:
            self.spatial_relationships = json.loads(msg.data)
            # Update object locations based on spatial relationships
            self.update_object_locations()
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to decode spatial relationships: {e}')

    def update_object_locations(self):
        """Update object locations based on spatial relationships"""
        # This would typically come from visual SLAM or other localization
        # For now, we'll maintain a simple mapping
        for relation in self.spatial_relationships:
            if 'object' in relation and 'position' in relation:
                self.object_locations[relation['object']] = relation['position']

    def process_command(self):
        """Process the current parsed command"""
        if not self.current_parsed_command:
            return

        cmd_type = self.current_parsed_command['command_type']
        target_obj = self.current_parsed_command['target_object']
        target_loc = self.current_parsed_command['target_location']
        confidence = self.current_parsed_command['confidence']

        # Check confidence threshold
        if confidence < 0.5:
            self.get_logger().warn(f'Command confidence too low: {confidence}')
            self.publish_status(f'Command confidence too low: {confidence:.2f}')
            return

        self.current_state = InterpretationState.PROCESSING
        self.publish_status(f'Processing {cmd_type} command')

        # Interpret and execute command based on type
        if cmd_type == 'navigate':
            self.handle_navigate_command(target_loc)
        elif cmd_type == 'find':
            self.handle_find_command(target_obj)
        elif cmd_type == 'grasp':
            self.handle_grasp_command(target_obj)
        elif cmd_type == 'follow':
            self.handle_follow_command()
        elif cmd_type == 'explore':
            self.handle_explore_command()
        else:
            self.handle_generic_command(cmd_type, target_obj, target_loc)

    def handle_navigate_command(self, target_location):
        """Handle navigation commands"""
        if not target_location:
            self.get_logger().error('No target location specified for navigation')
            self.publish_status('Error: No target location specified')
            return

        # For this example, we'll use simple location keywords
        # In practice, this would interface with a map or spatial memory
        location_poses = {
            'kitchen': (2.0, 1.0, 0.0),
            'bedroom': (-2.0, 2.0, 0.0),
            'living room': (0.0, -2.0, 0.0),
            'office': (3.0, -1.0, 0.0),
            'bathroom': (-1.0, -3.0, 0.0),
        }

        if target_location in location_poses:
            x, y, theta = location_poses[target_location]
            self.navigate_to_pose(x, y, theta)
            self.publish_status(f'Navigating to {target_location}')
        else:
            # Try to find the location in scene description
            if target_location in self.scene_description.lower():
                # This is a simplified approach - in reality, you'd need spatial mapping
                self.get_logger().info(f'Location {target_location} recognized in scene, navigating to approximate area')
                # Navigate to a default position for unknown locations
                self.navigate_to_pose(1.0, 1.0, 0.0)
            else:
                self.get_logger().error(f'Unknown location: {target_location}')
                self.publish_status(f'Unknown location: {target_location}')

    def handle_find_command(self, target_object):
        """Handle find commands"""
        if not target_object:
            self.get_logger().error('No target object specified for find command')
            self.publish_status('Error: No target object specified')
            return

        # Check if object is in scene description
        if target_object.lower() in self.scene_description.lower():
            self.get_logger().info(f'Found {target_object} in scene description: {self.scene_description}')
            self.publish_status(f'Found {target_object} in the scene')

            # If object location is known, navigate to it
            if target_object in self.object_locations:
                pos = self.object_locations[target_object]
                self.navigate_to_pose(pos['x'], pos['y'], 0.0)
        else:
            self.get_logger().info(f'{target_object} not found in current scene, beginning search')
            self.publish_status(f'Searching for {target_object}')
            # Begin search behavior
            self.begin_search_for_object(target_object)

    def handle_grasp_command(self, target_object):
        """Handle grasp commands"""
        if not target_object:
            self.get_logger().error('No target object specified for grasp command')
            self.publish_status('Error: No target object specified')
            return

        # First, find the object
        if target_object.lower() in self.scene_description.lower():
            self.get_logger().info(f'Attempting to grasp {target_object}')
            self.publish_status(f'Moving to grasp {target_object}')

            # This would trigger manipulation planning in a real system
            grasp_cmd = {
                'action': 'grasp',
                'target': target_object,
                'status': 'initiated'
            }
            cmd_msg = String()
            cmd_msg.data = json.dumps(grasp_cmd)
            self.robot_command_pub.publish(cmd_msg)
        else:
            self.get_logger().info(f'Need to find {target_object} before grasping')
            self.publish_status(f'Searching for {target_object} before grasping')
            self.begin_search_for_object(target_object)

    def handle_follow_command(self):
        """Handle follow commands"""
        self.get_logger().info('Following command received')
        self.publish_status('Following mode activated')

        follow_cmd = {
            'action': 'follow',
            'target': 'human',
            'status': 'initiated'
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(follow_cmd)
        self.robot_command_pub.publish(cmd_msg)

    def handle_explore_command(self):
        """Handle explore commands"""
        self.get_logger().info('Exploring command received')
        self.publish_status('Exploring environment')

        explore_cmd = {
            'action': 'explore',
            'status': 'initiated'
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(explore_cmd)
        self.robot_command_pub.publish(cmd_msg)

    def handle_generic_command(self, cmd_type, target_obj, target_loc):
        """Handle other command types"""
        self.get_logger().info(f'Handling generic command: {cmd_type}')
        self.publish_status(f'Executing {cmd_type} command')

        cmd = {
            'action': cmd_type,
            'target_object': target_obj,
            'target_location': target_loc,
            'status': 'initiated'
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(cmd)
        self.robot_command_pub.publish(cmd_msg)

    def begin_search_for_object(self, target_object):
        """Begin search behavior for a specific object"""
        # This would implement a search pattern
        # For now, we'll just navigate to a few waypoints
        search_cmd = {
            'action': 'search',
            'target': target_object,
            'waypoints': [
                {'x': 1.0, 'y': 1.0, 'theta': 0.0},
                {'x': -1.0, 'y': 1.0, 'theta': 0.0},
                {'x': -1.0, 'y': -1.0, 'theta': 0.0},
                {'x': 1.0, 'y': -1.0, 'theta': 0.0}
            ],
            'status': 'initiated'
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(search_cmd)
        self.robot_command_pub.publish(cmd_msg)

    def navigate_to_pose(self, x, y, theta):
        """Navigate to a specific pose using Nav2"""
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Navigation action server not available')
            self.publish_status('Navigation server not available')
            return

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta (yaw) to quaternion
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # Send goal
        self.get_logger().info(f'Sending navigation goal to ({x}, {y}, {theta})')
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Navigation goal accepted')
                self.current_state = InterpretationState.EXECUTING
                self.publish_status('Navigation in progress')
            else:
                self.get_logger().error('Navigation goal rejected')
                self.current_state = InterpretationState.ERROR
                self.publish_status('Navigation goal rejected')
        except Exception as e:
            self.get_logger().error(f'Exception in navigation result: {e}')
            self.current_state = InterpretationState.ERROR
            self.publish_status(f'Navigation error: {str(e)}')

    def publish_status(self, status_msg):
        """Publish status message"""
        msg = String()
        msg.data = status_msg
        self.interpretation_status_pub.publish(msg)

    def create_action_plan(self, command_type, target_obj=None, target_loc=None):
        """Create an action plan for the given command"""
        plan = {
            'command': command_type,
            'target_object': target_obj,
            'target_location': target_loc,
            'steps': [],
            'status': 'created'
        }

        if command_type == 'navigate':
            plan['steps'] = [
                {'action': 'localize', 'description': 'Determine current position'},
                {'action': 'plan_path', 'description': f'Plan path to {target_loc}'},
                {'action': 'execute_navigation', 'description': f'Navigate to {target_loc}'}
            ]
        elif command_type == 'find':
            plan['steps'] = [
                {'action': 'analyze_scene', 'description': f'Analyze scene for {target_obj}'},
                {'action': 'search', 'description': f'Search for {target_obj} if not found'},
                {'action': 'report', 'description': f'Report location of {target_obj}'}
            ]
        elif command_type == 'grasp':
            plan['steps'] = [
                {'action': 'locate_object', 'description': f'Locate {target_obj}'},
                {'action': 'approach_object', 'description': f'Approach {target_obj}'},
                {'action': 'grasp_object', 'description': f'Grasp {target_obj}'},
                {'action': 'verify_grasp', 'description': 'Verify successful grasp'}
            ]

        # Publish action plan
        plan_msg = String()
        plan_msg.data = json.dumps(plan)
        self.action_plan_pub.publish(plan_msg)

        return plan


def main(args=None):
    rclpy.init(args=args)

    interpretation_node = CommandInterpretationNode()

    try:
        rclpy.spin(interpretation_node)
    except KeyboardInterrupt:
        pass
    finally:
        interpretation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()