#!/usr/bin/env python3

"""
Safety Guardrails for LLM-Controlled Robots
Module 4: Vision-Language-Action (VLA) - Physical AI & Humanoid Robotics Textbook

This module implements safety guardrails for LLM-controlled robots to ensure safe operation.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Duration
import numpy as np
import math
from typing import List, Tuple, Dict, Any
import json


class SafetyGuardrailsNode(Node):
    """
    Node that implements safety guardrails for LLM-controlled robots
    """
    def __init__(self):
        super().__init__('safety_guardrails_node')

        # Publishers
        self.safe_cmd_vel_pub = self.create_publisher(
            Twist,
            '/safe_cmd_vel',
            10
        )

        self.emergency_stop_pub = self.create_publisher(
            Bool,
            '/emergency_stop',
            10
        )

        self.safety_status_pub = self.create_publisher(
            String,
            '/safety/status',
            10
        )

        # Subscribers
        self.unsafe_cmd_vel_sub = self.create_subscription(
            Twist,
            '/unsafe_cmd_vel',  # LLM-generated commands go here first
            self.unsafe_cmd_vel_callback,
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

        # Internal state
        self.current_scan = None
        self.current_pose = None
        self.emergency_stop_active = False
        self.last_safe_command = Twist()

        # Safety parameters
        self.min_obstacle_distance = 0.5  # meters
        self.max_linear_speed = 0.5       # m/s
        self.max_angular_speed = 1.0      # rad/s
        self.collision_threshold = 0.3    # meters

        # Safety statistics
        self.safety_interventions = 0
        self.total_commands_processed = 0

        self.get_logger().info('Safety Guardrails Node initialized')

    def scan_callback(self, msg):
        """Handle incoming LIDAR scan data"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Handle incoming odometry data"""
        self.current_pose = msg.pose.pose

    def unsafe_cmd_vel_callback(self, msg):
        """Process potentially unsafe commands from LLM"""
        if self.emergency_stop_active:
            # If emergency stop is active, only allow zero velocity
            safe_cmd = Twist()
            self.safe_cmd_vel_pub.publish(safe_cmd)
            return

        self.total_commands_processed += 1

        # Check if the proposed command is safe
        if self.is_command_safe(msg):
            # Command is safe, pass it through
            self.safe_cmd_vel_pub.publish(msg)
            self.last_safe_command = msg

            status_msg = String()
            status_msg.data = f"SAFE: Command passed through. Linear: ({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), Angular: ({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})"
            self.safety_status_pub.publish(status_msg)

            self.get_logger().debug(f'Safe command passed: v={msg.linear.x:.2f}, ω={msg.angular.z:.2f}')
        else:
            # Command is unsafe, apply corrections or stop
            safe_cmd = self.apply_safety_correction(msg)
            self.safe_cmd_vel_pub.publish(safe_cmd)

            self.safety_interventions += 1

            status_msg = String()
            status_msg.data = f"SAFETY INTERVENTION: Unsafe command corrected. Original: v={msg.linear.x:.2f}, ω={msg.angular.z:.2f}, Corrected: v={safe_cmd.linear.x:.2f}, ω={safe_cmd.angular.z:.2f}"
            self.safety_status_pub.publish(status_msg)

            self.get_logger().warn(f'Safety intervention: unsafe command corrected')

    def is_command_safe(self, cmd_vel: Twist) -> bool:
        """Check if a command is safe to execute"""
        if not self.current_scan:
            # If no scan data, be conservative and allow only safe commands
            return self.is_safe_default_behavior(cmd_vel)

        # Check speed limits
        if abs(cmd_vel.linear.x) > self.max_linear_speed:
            self.get_logger().debug(f'Unsafe: Linear speed too high: {cmd_vel.linear.x:.2f} > {self.max_linear_speed:.2f}')
            return False

        if abs(cmd_vel.angular.z) > self.max_angular_speed:
            self.get_logger().debug(f'Unsafe: Angular speed too high: {cmd_vel.angular.z:.2f} > {self.max_angular_speed:.2f}')
            return False

        # Check for obstacles in the direction of movement
        if cmd_vel.linear.x > 0:  # Moving forward
            front_distance = self.get_front_distance()
            if front_distance < self.min_obstacle_distance:
                self.get_logger().debug(f'Unsafe: Obstacle in front: {front_distance:.2f}m < {self.min_obstacle_distance:.2f}m')
                return False

        if cmd_vel.linear.x < 0:  # Moving backward
            rear_distance = self.get_rear_distance()
            if rear_distance < self.min_obstacle_distance:
                self.get_logger().debug(f'Unsafe: Obstacle behind: {rear_distance:.2f}m < {self.min_obstacle_distance:.2f}m')
                return False

        # Check if turning into an obstacle
        if abs(cmd_vel.angular.z) > 0.1:  # Significant turning
            turn_direction = 1 if cmd_vel.angular.z > 0 else -1
            side_distance = self.get_side_distance(turn_direction)
            if side_distance < self.min_obstacle_distance:
                self.get_logger().debug(f'Unsafe: Obstacle during turn: {side_distance:.2f}m < {self.min_obstacle_distance:.2f}m')
                return False

        return True

    def is_safe_default_behavior(self, cmd_vel: Twist) -> bool:
        """Check if command is safe when no sensor data is available"""
        # When no sensor data is available, only allow very cautious movements
        if abs(cmd_vel.linear.x) > 0.1:  # Very slow movement allowed
            return False
        if abs(cmd_vel.angular.z) > 0.1:  # Very slow turning allowed
            return False
        return True

    def apply_safety_correction(self, unsafe_cmd: Twist) -> Twist:
        """
        Apply safety corrections to an unsafe command
        Returns a safe command that is as close as possible to the original
        """
        safe_cmd = Twist()

        # If no scan data, stop the robot
        if not self.current_scan:
            return safe_cmd

        # Check for immediate collision risk
        front_distance = self.get_front_distance()
        rear_distance = self.get_rear_distance()

        # If there's an imminent collision risk, stop
        if front_distance < self.collision_threshold and unsafe_cmd.linear.x > 0:
            # Stop forward motion but allow other movements if safe
            safe_cmd.linear.x = 0.0
        elif rear_distance < self.collision_threshold and unsafe_cmd.linear.x < 0:
            # Stop backward motion but allow other movements if safe
            safe_cmd.linear.x = 0.0
        else:
            # Scale down the command based on obstacle proximity
            speed_scale = self.calculate_speed_scale(front_distance)
            safe_cmd.linear.x = unsafe_cmd.linear.x * speed_scale
            safe_cmd.linear.y = unsafe_cmd.linear.y * speed_scale
            safe_cmd.linear.z = unsafe_cmd.linear.z * speed_scale

        # Limit angular velocity based on forward speed to prevent sliding
        max_angular_for_speed = min(self.max_angular_speed, abs(unsafe_cmd.linear.x) * 2.0 + 0.2)
        safe_cmd.angular.z = max(-max_angular_for_speed, min(max_angular_for_speed, unsafe_cmd.angular.z))
        safe_cmd.angular.x = unsafe_cmd.angular.x
        safe_cmd.angular.y = unsafe_cmd.angular.y

        # Ensure limits are respected
        safe_cmd.linear.x = max(-self.max_linear_speed, min(self.max_linear_speed, safe_cmd.linear.x))
        safe_cmd.linear.y = max(-self.max_linear_speed, min(self.max_linear_speed, safe_cmd.linear.y))
        safe_cmd.linear.z = max(-self.max_linear_speed, min(self.max_linear_speed, safe_cmd.linear.z))

        return safe_cmd

    def get_front_distance(self) -> float:
        """Get the minimum distance to obstacles in the front 60-degree sector"""
        if not self.current_scan:
            return float('inf')

        # Calculate indices for front sector (±30 degrees from center)
        total_angles = len(self.current_scan.ranges)
        center_idx = total_angles // 2
        sector_width = int((60.0 / 360.0) * total_angles)  # 60-degree sector
        start_idx = max(0, center_idx - sector_width // 2)
        end_idx = min(total_angles, center_idx + sector_width // 2)

        front_ranges = self.current_scan.ranges[start_idx:end_idx]
        valid_ranges = [r for r in front_ranges
                       if self.current_scan.range_min < r < self.current_scan.range_max]

        return min(valid_ranges) if valid_ranges else float('inf')

    def get_rear_distance(self) -> float:
        """Get the minimum distance to obstacles in the rear 60-degree sector"""
        if not self.current_scan:
            return float('inf')

        # Calculate indices for rear sector (180 degrees from center)
        total_angles = len(self.current_scan.ranges)
        center_idx = total_angles // 2
        opposite_idx = (center_idx + total_angles // 2) % total_angles
        sector_width = int((60.0 / 360.0) * total_angles)  # 60-degree sector
        start_idx = max(0, opposite_idx - sector_width // 2)
        end_idx = min(total_angles, opposite_idx + sector_width // 2)

        rear_ranges = self.current_scan.ranges[start_idx:end_idx]
        valid_ranges = [r for r in rear_ranges
                       if self.current_scan.range_min < r < self.current_scan.range_max]

        return min(valid_ranges) if valid_ranges else float('inf')

    def get_side_distance(self, direction: int) -> float:
        """
        Get the minimum distance to obstacles on the side
        direction: 1 for right, -1 for left
        """
        if not self.current_scan:
            return float('inf')

        total_angles = len(self.current_scan.ranges)

        if direction > 0:  # Right side (90 degrees)
            side_idx = (total_angles * 3) // 4  # 270 degrees (right in ROS convention)
        else:  # Left side (-90 degrees)
            side_idx = total_angles // 4  # 90 degrees (left in ROS convention)

        sector_width = int((30.0 / 360.0) * total_angles)  # 30-degree sector
        start_idx = max(0, side_idx - sector_width // 2)
        end_idx = min(total_angles, side_idx + sector_width // 2)

        side_ranges = self.current_scan.ranges[start_idx:end_idx]
        valid_ranges = [r for r in side_ranges
                       if self.current_scan.range_min < r < self.current_scan.range_max]

        return min(valid_ranges) if valid_ranges else float('inf')

    def calculate_speed_scale(self, distance: float) -> float:
        """Calculate speed scaling factor based on distance to obstacles"""
        if distance >= 2.0:  # Far away, full speed allowed
            return 1.0
        elif distance <= self.min_obstacle_distance:  # Too close, stop
            return 0.0
        else:  # Scale linearly between min distance and 2m
            scale = (distance - self.min_obstacle_distance) / (2.0 - self.min_obstacle_distance)
            return max(0.0, min(1.0, scale))

    def emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        stop_cmd = Twist()
        self.safe_cmd_vel_pub.publish(stop_cmd)

        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)

        self.get_logger().error('EMERGENCY STOP ACTIVATED!')

    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop_active = False
        self.get_logger().info('Emergency stop reset')

    def get_safety_report(self) -> Dict[str, Any]:
        """Get a report of safety metrics"""
        return {
            'total_commands_processed': self.total_commands_processed,
            'safety_interventions': self.safety_interventions,
            'intervention_rate': self.safety_interventions / max(1, self.total_commands_processed),
            'last_safe_command': {
                'linear': {
                    'x': self.last_safe_command.linear.x,
                    'y': self.last_safe_command.linear.y,
                    'z': self.last_safe_command.linear.z
                },
                'angular': {
                    'x': self.last_safe_command.angular.x,
                    'y': self.last_safe_command.angular.y,
                    'z': self.last_safe_command.angular.z
                }
            },
            'current_scan_available': self.current_scan is not None,
            'current_pose_available': self.current_pose is not None
        }


def main(args=None):
    rclpy.init(args=args)

    safety_node = SafetyGuardrailsNode()

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        safety_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()