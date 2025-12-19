#!/usr/bin/env python3

"""
Decision Making Module for Physical AI & Humanoid Robotics Textbook
Module 3: The AI-Robot Brain (NVIDIA Isaac™)

This module implements high-level decision making for the AI robot brain.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from enum import Enum
from collections import deque
import json


class RobotState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    REACHED_GOAL = "reached_goal"
    EMERGENCY_STOP = "emergency_stop"
    EXPLORING = "exploring"


class DecisionMakingNode(Node):
    def __init__(self):
        super().__init__('decision_making_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/ai_brain/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ai_brain/markers', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.emergency_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_callback,
            10
        )

        # Internal state
        self.current_state = RobotState.IDLE
        self.current_pose = None
        self.current_scan = None
        self.current_goal = None
        self.emergency_stop = False
        self.scan_buffer = deque(maxlen=5)

        # Parameters
        self.safety_distance = 0.5  # meters
        self.goal_tolerance = 0.3   # meters
        self.linear_speed = 0.3     # m/s
        self.angular_speed = 0.5    # rad/s

        # Timer for decision making
        self.decision_timer = self.create_timer(0.1, self.make_decision)  # 10 Hz

        self.get_logger().info('Decision Making Node initialized')

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Handle LIDAR scan updates"""
        self.current_scan = msg
        self.scan_buffer.append(msg)

    def emergency_callback(self, msg):
        """Handle emergency stop commands"""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.current_state = RobotState.EMERGENCY_STOP
            self.publish_stop_command()

    def make_decision(self):
        """Main decision making function"""
        if self.emergency_stop:
            self.publish_stop_command()
            self.publish_status("EMERGENCY STOP")
            return

        # Update state based on conditions
        self.update_state()

        # Execute behavior based on current state
        if self.current_state == RobotState.IDLE:
            self.handle_idle_state()
        elif self.current_state == RobotState.NAVIGATING:
            self.handle_navigation_state()
        elif self.current_state == RobotState.AVOIDING_OBSTACLE:
            self.handle_obstacle_avoidance_state()
        elif self.current_state == RobotState.REACHED_GOAL:
            self.handle_reached_goal_state()
        elif self.current_state == RobotState.EXPLORING:
            self.handle_exploration_state()
        elif self.current_state == RobotState.EMERGENCY_STOP:
            self.publish_stop_command()

        # Publish status
        self.publish_status(f"State: {self.current_state.value}")

    def update_state(self):
        """Update robot state based on current conditions"""
        if self.current_scan is None or self.current_pose is None:
            return

        # Check for obstacles in front
        front_distance = self.get_front_distance(self.current_scan)

        # Check if goal is set and reached
        if self.current_goal is not None:
            distance_to_goal = self.get_distance_to_goal()

            if distance_to_goal < self.goal_tolerance:
                self.current_state = RobotState.REACHED_GOAL
            elif front_distance < self.safety_distance:
                self.current_state = RobotState.AVOIDING_OBSTACLE
            else:
                self.current_state = RobotState.NAVIGATING
        else:
            # If no goal, consider exploring or staying idle
            if front_distance > 2.0:  # Clear path ahead
                self.current_state = RobotState.EXPLORING
            else:
                self.current_state = RobotState.IDLE

    def handle_idle_state(self):
        """Handle idle state behavior"""
        self.publish_stop_command()
        # Could implement some random exploration or waiting behavior
        pass

    def handle_navigation_state(self):
        """Handle navigation state behavior"""
        if self.current_goal is not None:
            # Calculate direction to goal
            cmd_vel = self.calculate_navigation_command()
            self.cmd_vel_pub.publish(cmd_vel)

    def handle_obstacle_avoidance_state(self):
        """Handle obstacle avoidance state behavior"""
        cmd_vel = Twist()

        # Simple obstacle avoidance: turn away from obstacles
        if self.current_scan:
            left_clear = self.get_sector_distance(self.current_scan, -60, -30) > self.safety_distance
            right_clear = self.get_sector_distance(self.current_scan, 30, 60) > self.safety_distance

            if left_clear and right_clear:
                # Both sides clear, go straight but slow down
                cmd_vel.linear.x = self.linear_speed * 0.3
                cmd_vel.angular.z = 0.0
            elif left_clear:
                # Left side clear, turn left
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed
            elif right_clear:
                # Right side clear, turn right
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = -self.angular_speed
            else:
                # Neither side clear, turn randomly
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed if np.random.random() > 0.5 else -self.angular_speed

        self.cmd_vel_pub.publish(cmd_vel)

    def handle_reached_goal_state(self):
        """Handle reached goal state behavior"""
        self.publish_stop_command()
        # Could implement goal completion logic or set new goal

    def handle_exploration_state(self):
        """Handle exploration state behavior"""
        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear_speed * 0.7  # Move forward slowly
        cmd_vel.angular.z = 0.0

        # Add some random turning for exploration
        if np.random.random() < 0.02:  # 2% chance per cycle to turn
            cmd_vel.angular.z = self.angular_speed if np.random.random() > 0.5 else -self.angular_speed

        self.cmd_vel_pub.publish(cmd_vel)

    def calculate_navigation_command(self):
        """Calculate navigation command to reach goal"""
        cmd_vel = Twist()

        if self.current_pose is None or self.current_goal is None:
            return cmd_vel

        # Calculate direction to goal
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y

        # Calculate angle to goal
        angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)

        # Calculate robot's current angle (from quaternion)
        robot_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Calculate angle difference
        angle_diff = angle_to_goal - robot_yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Set angular velocity proportional to angle error
        cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 1.5))

        # If roughly facing goal, move forward
        if abs(angle_diff) < 0.2:
            cmd_vel.linear.x = self.linear_speed

        return cmd_vel

    def get_distance_to_goal(self):
        """Get distance from current pose to goal"""
        if self.current_pose is None or self.current_goal is None:
            return float('inf')

        dx = self.current_pose.position.x - self.current_goal.pose.position.x
        dy = self.current_pose.position.y - self.current_goal.pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def get_front_distance(self, scan_msg):
        """Get distance to obstacle directly in front"""
        if scan_msg is None:
            return float('inf')

        # Get points in front (±15 degrees)
        front_indices = range(
            len(scan_msg.ranges) // 2 - len(scan_msg.ranges) // 24,
            len(scan_msg.ranges) // 2 + len(scan_msg.ranges) // 24
        )

        front_ranges = [scan_msg.ranges[i] for i in front_indices
                       if 0 <= i < len(scan_msg.ranges)]

        valid_ranges = [r for r in front_ranges if scan_msg.range_min < r < scan_msg.range_max]

        return min(valid_ranges) if valid_ranges else float('inf')

    def get_sector_distance(self, scan_msg, start_angle_deg, end_angle_deg):
        """Get minimum distance in a specific angular sector"""
        if scan_msg is None:
            return float('inf')

        # Convert angles to indices
        angle_increment = scan_msg.angle_increment
        start_idx = int((math.radians(start_angle_deg) - scan_msg.angle_min) / angle_increment)
        end_idx = int((math.radians(end_angle_deg) - scan_msg.angle_min) / angle_increment)

        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(scan_msg.ranges) - 1))
        end_idx = max(0, min(end_idx, len(scan_msg.ranges) - 1))

        sector_ranges = scan_msg.ranges[start_idx:end_idx] if start_idx <= end_idx else scan_msg.ranges[end_idx:start_idx]

        valid_ranges = [r for r in sector_ranges if scan_msg.range_min < r < scan_msg.range_max]

        return min(valid_ranges) if valid_ranges else float('inf')

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_stop_command(self):
        """Publish zero velocity command to stop robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_status(self, status_msg):
        """Publish status message"""
        msg = String()
        msg.data = status_msg
        self.status_pub.publish(msg)

    def set_goal(self, x, y, z=0.0):
        """Set a new navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z
        goal_msg.pose.orientation.w = 1.0  # No rotation

        self.current_goal = goal_msg
        self.goal_pub.publish(goal_msg)

        self.get_logger().info(f"New goal set: ({x}, {y})")


def main(args=None):
    rclpy.init(args=args)

    decision_node = DecisionMakingNode()

    try:
        # Example: Set a goal after a delay
        def set_example_goal():
            decision_node.set_goal(2.0, 2.0)

        # Set goal after 5 seconds
        decision_node.create_timer(5.0, set_example_goal)

        rclpy.spin(decision_node)
    except KeyboardInterrupt:
        pass
    finally:
        decision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()