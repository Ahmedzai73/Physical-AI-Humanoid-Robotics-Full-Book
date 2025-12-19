#!/usr/bin/env python3

"""
Skill Execution Module for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module demonstrates skill execution for the VLA system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Twist
from sensor_msgs.msg import JointState, Image, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import time
import math
from typing import Dict, List, Optional
from enum import Enum


class SkillType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    INTERACTION = "interaction"
    LOCOMOTION = "locomotion"


class SkillState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SkillExecutorNode(Node):
    def __init__(self):
        super().__init__('skill_executor_node')

        # Subscribers
        self.skill_command_sub = self.create_subscription(
            String,
            '/skill/command',
            self.skill_command_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.skill_status_pub = self.create_publisher(
            String,
            '/skill/status',
            10
        )

        self.skill_feedback_pub = self.create_publisher(
            String,
            '/skill/feedback',
            10
        )

        self.get_logger().info('Skill Executor Node initialized')

        # Skill execution state
        self.current_state = SkillState.IDLE
        self.current_skill = None
        self.current_skill_params = {}
        self.current_joint_state = None
        self.current_scan = None

        # Robot configuration
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.default_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.max_velocities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # Skill execution parameters
        self.skill_timeout = 30.0  # seconds
        self.skill_execution_start_time = None

        # Active skills
        self.active_skills = {}

    def skill_command_callback(self, msg):
        """Process incoming skill commands"""
        try:
            command_data = json.loads(msg.data)
            skill_name = command_data.get('skill', '')
            skill_params = command_data.get('params', {})
            command = command_data.get('command', 'execute')

            self.get_logger().info(f'Received skill command: {command} for {skill_name}')

            if command == 'execute':
                self.execute_skill(skill_name, skill_params)
            elif command == 'cancel':
                self.cancel_skill(skill_name)
            elif command == 'pause':
                self.pause_skill(skill_name)
            elif command == 'resume':
                self.resume_skill(skill_name)

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid skill command JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing skill command: {e}')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        self.current_joint_state = msg

    def scan_callback(self, msg):
        """Update scan information"""
        self.current_scan = msg

    def execute_skill(self, skill_name: str, params: Dict):
        """Execute a named skill with given parameters"""
        self.current_state = SkillState.INITIALIZING
        self.publish_status(f'Initializing skill: {skill_name}')

        # Store skill parameters
        self.current_skill = skill_name
        self.current_skill_params = params
        self.skill_execution_start_time = time.time()

        # Execute the appropriate skill based on name
        if skill_name == 'move_to_position':
            self.execute_move_to_position(params)
        elif skill_name == 'grasp_object':
            self.execute_grasp_object(params)
        elif skill_name == 'navigate_to':
            self.execute_navigate_to(params)
        elif skill_name == 'wave':
            self.execute_wave(params)
        elif skill_name == 'point_to':
            self.execute_point_to(params)
        elif skill_name == 'turn_around':
            self.execute_turn_around(params)
        else:
            self.get_logger().error(f'Unknown skill: {skill_name}')
            self.current_state = SkillState.FAILED
            self.publish_status(f'Unknown skill: {skill_name}')

    def execute_move_to_position(self, params: Dict):
        """Execute move to position skill"""
        target_x = params.get('x', 0.0)
        target_y = params.get('y', 0.0)
        target_z = params.get('z', 0.0)

        self.get_logger().info(f'Moving to position ({target_x}, {target_y}, {target_z})')

        # This would typically interface with a motion planner
        # For this example, we'll just publish a simple velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2  # Move forward slowly
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

        # Set state to executing
        self.current_state = SkillState.EXECUTING
        self.publish_status(f'Moving to position ({target_x}, {target_y}, {target_z})')

        # In a real system, you'd monitor progress and update state accordingly
        # For this example, we'll complete after a short time
        def complete_move():
            self.current_state = SkillState.COMPLETED
            self.publish_status('Move to position completed')

        self.create_timer(3.0, complete_move)

    def execute_grasp_object(self, params: Dict):
        """Execute grasp object skill"""
        object_name = params.get('object', 'unknown')
        approach_distance = params.get('approach_distance', 0.3)

        self.get_logger().info(f'Attempting to grasp {object_name}')

        # For this example, we'll just simulate a grasping motion
        # using joint trajectories

        # First, move to approach position
        self.move_to_approach_position(approach_distance)

        # Then, execute grasp motion
        self.execute_grasp_motion()

        self.current_state = SkillState.EXECUTING
        self.publish_status(f'Grasping {object_name}')

        # Simulate completion
        def complete_grasp():
            self.current_state = SkillState.COMPLETED
            self.publish_status(f'Grasped {object_name} successfully')

        self.create_timer(5.0, complete_grasp)

    def execute_navigate_to(self, params: Dict):
        """Execute navigate to location skill"""
        target_location = params.get('location', 'unknown')
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)

        self.get_logger().info(f'Navigating to {target_location} at ({x}, {y})')

        # This would interface with the navigation system
        # For this example, we'll publish a simple navigation command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.3  # Move forward
        cmd_vel.angular.z = 0.0  # No rotation initially

        self.cmd_vel_pub.publish(cmd_vel)

        self.current_state = SkillState.EXECUTING
        self.publish_status(f'Navigating to {target_location}')

        # Simulate navigation completion
        def complete_navigation():
            self.current_state = SkillState.COMPLETED
            self.publish_status(f'Arrived at {target_location}')

        self.create_timer(4.0, complete_navigation)

    def execute_wave(self, params: Dict):
        """Execute waving gesture"""
        num_waves = params.get('count', 3)
        wave_speed = params.get('speed', 1.0)

        self.get_logger().info(f'Waving {num_waves} times')

        # Create a waving motion trajectory
        trajectory = self.create_waving_trajectory(num_waves, wave_speed)
        self.joint_trajectory_pub.publish(trajectory)

        self.current_state = SkillState.EXECUTING
        self.publish_status(f'Waving {num_waves} times')

        # Calculate duration based on number of waves
        duration = num_waves * 2.0 / wave_speed  # 2 seconds per wave cycle
        def complete_wave():
            self.current_state = SkillState.COMPLETED
            self.publish_status('Waving completed')

        self.create_timer(duration, complete_wave)

    def execute_point_to(self, params: Dict):
        """Execute pointing gesture"""
        target_x = params.get('x', 0.0)
        target_y = params.get('y', 0.0)
        target_z = params.get('z', 0.0)

        self.get_logger().info(f'Pointing to ({target_x}, {target_y}, {target_z})')

        # Calculate joint angles to point to target
        # This is a simplified calculation
        trajectory = self.calculate_pointing_trajectory(target_x, target_y, target_z)
        self.joint_trajectory_pub.publish(trajectory)

        self.current_state = SkillState.EXECUTING
        self.publish_status(f'Pointing to ({target_x}, {target_y}, {target_z})')

        def complete_point():
            self.current_state = SkillState.COMPLETED
            self.publish_status('Pointing completed')

        self.create_timer(2.0, complete_point)

    def execute_turn_around(self, params: Dict):
        """Execute turn around skill"""
        turn_direction = params.get('direction', 'left')
        angle = params.get('angle', 180)  # degrees

        self.get_logger().info(f'Turning {turn_direction} by {angle} degrees')

        cmd_vel = Twist()
        if turn_direction == 'right':
            cmd_vel.angular.z = -0.5  # Turn right
        else:
            cmd_vel.angular.z = 0.5   # Turn left

        # Calculate time to turn specified angle
        angle_rad = math.radians(angle)
        turn_time = angle_rad / 0.5  # 0.5 rad/s turn rate

        self.cmd_vel_pub.publish(cmd_vel)

        self.current_state = SkillState.EXECUTING
        self.publish_status(f'Turning {turn_direction} by {angle} degrees')

        def stop_turn():
            # Stop the turn
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            self.current_state = SkillState.COMPLETED
            self.publish_status('Turn completed')

        self.create_timer(turn_time, stop_turn)

    def move_to_approach_position(self, distance: float):
        """Move robot to approach position for grasping"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.1  # Slow approach
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

    def execute_grasp_motion(self):
        """Execute the actual grasping motion"""
        # Create a simple grasping trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        # Create trajectory points for grasping
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]  # Approach position
        point1.velocities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        point1.time_from_start = Duration(sec=1, nanosec=0)

        point2 = JointTrajectoryPoint()
        point2.positions = [0.0, -1.0, 1.0, 0.0, 0.5, 0.0]  # Grasp position
        point2.velocities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        point2.time_from_start = Duration(sec=2, nanosec=0)

        trajectory.points = [point1, point2]
        trajectory.header.stamp = self.get_clock().now().to_msg()

        self.joint_trajectory_pub.publish(trajectory)

    def create_waving_trajectory(self, num_waves: int, speed: float) -> JointTrajectory:
        """Create a waving motion trajectory"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        points = []
        time_step = 0.5 / speed  # Adjust for speed

        for i in range(num_waves * 4):  # 4 points per wave cycle
            t = i * time_step
            point = JointTrajectoryPoint()

            # Create waving motion in one joint
            wave_pos = math.sin(i * math.pi / 2) * 0.5  # Oscillate between -0.5 and 0.5

            positions = self.default_positions.copy()
            positions[1] = wave_pos  # Use joint 2 for waving
            positions[2] = -wave_pos  # Counter-movement in joint 3

            point.positions = positions
            point.velocities = [0.0] * len(positions)
            point.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))

            points.append(point)

        trajectory.points = points
        trajectory.header.stamp = self.get_clock().now().to_msg()

        return trajectory

    def calculate_pointing_trajectory(self, target_x: float, target_y: float, target_z: float) -> JointTrajectory:
        """Calculate joint angles to point to a specific location"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        # This is a simplified calculation
        # In reality, you'd use inverse kinematics
        point = JointTrajectoryPoint()

        # Calculate approximate angles to point to target
        # This is just a placeholder - real IK would be needed
        dx = target_x
        dy = target_y
        dz = target_z

        # Simple proportional angles
        joint_angles = [
            math.atan2(dy, dx),  # Base rotation
            math.atan2(dz, math.sqrt(dx*dx + dy*dy)),  # Elevation
            0.0,  # Elbow
            0.0,  # Wrist 1
            0.0,  # Wrist 2
            0.0   # Wrist 3
        ]

        point.positions = joint_angles
        point.velocities = [0.0] * len(joint_angles)
        point.time_from_start = Duration(sec=1, nanosec=0)

        trajectory.points = [point]
        trajectory.header.stamp = self.get_clock().now().to_msg()

        return trajectory

    def cancel_skill(self, skill_name: str):
        """Cancel the currently executing skill"""
        self.current_state = SkillState.CANCELLED
        self.publish_status(f'Skill {skill_name} cancelled')

        # Stop any ongoing motion
        self.stop_all_motion()

    def pause_skill(self, skill_name: str):
        """Pause the currently executing skill"""
        if self.current_state == SkillState.EXECUTING:
            self.current_state = SkillState.PAUSED
            self.publish_status(f'Skill {skill_name} paused')

            # Stop motion while paused
            self.stop_all_motion()

    def resume_skill(self, skill_name: str):
        """Resume a paused skill"""
        if self.current_state == SkillState.PAUSED:
            self.current_state = SkillState.EXECUTING
            self.publish_status(f'Skill {skill_name} resumed')

            # In a real system, you'd resume the skill execution
            # For this example, we'll just continue with the same motion

    def stop_all_motion(self):
        """Stop all robot motion"""
        # Stop linear and angular motion
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

        # Cancel any joint trajectories
        # In a real system, you might send a stop trajectory

    def check_skill_timeout(self):
        """Check if current skill has timed out"""
        if (self.current_state == SkillState.EXECUTING and
            self.skill_execution_start_time and
            time.time() - self.skill_execution_start_time > self.skill_timeout):
            self.get_logger().warn('Skill execution timed out')
            self.current_state = SkillState.FAILED
            self.publish_status('Skill execution timed out')
            self.stop_all_motion()

    def publish_status(self, status_msg: str):
        """Publish status message"""
        msg = String()
        msg.data = status_msg
        self.skill_status_pub.publish(msg)

        # Also publish to feedback
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            'skill': self.current_skill,
            'status': self.current_state.value,
            'message': status_msg
        })
        self.skill_feedback_pub.publish(feedback_msg)

    def get_current_skill_info(self) -> Dict:
        """Get information about the current skill"""
        return {
            'current_skill': self.current_skill,
            'state': self.current_state.value,
            'params': self.current_skill_params,
            'execution_time': time.time() - self.skill_execution_start_time if self.skill_execution_start_time else 0
        }


def main(args=None):
    rclpy.init(args=args)

    skill_node = SkillExecutorNode()

    try:
        # Example: Execute a skill after a delay
        def execute_example_skill():
            skill_cmd = {
                'skill': 'wave',
                'params': {'count': 2, 'speed': 1.0},
                'command': 'execute'
            }
            cmd_msg = String()
            cmd_msg.data = json.dumps(skill_cmd)
            skill_node.skill_command_callback(cmd_msg)

        # Execute example skill after 2 seconds
        skill_node.create_timer(2.0, execute_example_skill)

        # Add a periodic timeout check
        def check_timeout():
            skill_node.check_skill_timeout()

        skill_node.create_timer(1.0, check_timeout)

        rclpy.spin(skill_node)
    except KeyboardInterrupt:
        pass
    finally:
        skill_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()