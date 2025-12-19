#!/usr/bin/env python3

"""
Motion Planning Module for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module demonstrates motion planning for the VLA system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import json
from typing import List, Tuple, Optional
from enum import Enum


class MotionState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    ADAPTING = "adapting"
    COMPLETED = "completed"
    FAILED = "failed"


class MotionPlannerNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.motion_command_sub = self.create_subscription(
            String,
            '/motion/command',
            self.motion_command_callback,
            10
        )

        # Publishers
        self.path_pub = self.create_publisher(
            Path,
            '/motion/plan',
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.motion_status_pub = self.create_publisher(
            String,
            '/motion/status',
            10
        )

        self.motion_plan_pub = self.create_publisher(
            String,
            '/motion/plan_details',
            10
        )

        self.get_logger().info('Motion Planner Node initialized')

        # Robot state
        self.current_state = MotionState.IDLE
        self.current_pose = None
        self.current_goal = None
        self.current_map = None
        self.current_scan = None
        self.motion_plan = []

        # Motion planning parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.goal_tolerance = 0.3  # meters
        self.obstacle_threshold = 0.5  # meters

        # Grid map representation
        self.grid_map = None

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg.pose
        self.get_logger().info(f'Received goal: ({msg.pose.position.x}, {msg.pose.position.y})')

        # Plan motion to goal
        if self.current_pose:
            self.plan_motion_to_goal()

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.current_pose = msg.pose.pose

        # If we have a goal and are in executing state, check progress
        if self.current_goal and self.current_state == MotionState.EXECUTING:
            self.check_progress_to_goal()

    def map_callback(self, msg):
        """Process map data"""
        self.current_map = msg
        self.build_grid_map(msg)

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.current_scan = msg

        # If executing and need to adapt to obstacles
        if self.current_state == MotionState.EXECUTING:
            self.check_for_obstacles()

    def motion_command_callback(self, msg):
        """Process motion commands"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '')

            if command == 'stop':
                self.stop_motion()
            elif command == 'pause':
                self.pause_motion()
            elif command == 'resume':
                self.resume_motion()
            elif command == 'plan_to':
                target = command_data.get('target', {})
                if 'x' in target and 'y' in target:
                    self.plan_to_position(target['x'], target['y'])
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid motion command: {msg.data}')

    def plan_motion_to_goal(self):
        """Plan motion from current pose to goal"""
        if not self.current_pose or not self.current_goal:
            return

        self.current_state = MotionState.PLANNING
        self.publish_status('Planning motion to goal')

        # For this example, we'll use a simple potential field approach
        # In practice, you'd use A*, RRT, or other advanced planners
        self.motion_plan = self.generate_simple_plan(
            self.current_pose,
            self.current_goal
        )

        # Publish the plan
        self.publish_motion_plan()
        self.publish_path()

        # Start executing the plan
        self.start_execution()

    def generate_simple_plan(self, start_pose, goal_pose):
        """Generate a simple motion plan (straight line with obstacle avoidance)"""
        plan = []

        # Calculate the straight-line path
        start_x = start_pose.position.x
        start_y = start_pose.position.y
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y

        # Calculate direction vector
        dx = goal_x - start_x
        dy = goal_y - start_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Create waypoints along the path
        num_waypoints = max(10, int(distance / 0.5))  # 0.5m between waypoints
        for i in range(num_waypoints + 1):
            t = i / num_waypoints if num_waypoints > 0 else 0
            wp_x = start_x + t * dx
            wp_y = start_y + t * dy

            waypoint = Pose()
            waypoint.position.x = wp_x
            waypoint.position.y = wp_y
            waypoint.position.z = start_pose.position.z  # Maintain height

            # Calculate orientation towards next point
            if i < num_waypoints:
                next_t = (i + 1) / num_waypoints
                next_x = start_x + next_t * dx
                next_y = start_y + next_t * dy
                yaw = math.atan2(next_y - wp_y, next_x - wp_x)

                waypoint.orientation.z = math.sin(yaw / 2)
                waypoint.orientation.w = math.cos(yaw / 2)
            else:
                # Use goal orientation
                waypoint.orientation = goal_pose.orientation

            plan.append(waypoint)

        return plan

    def start_execution(self):
        """Start executing the motion plan"""
        if not self.motion_plan:
            self.get_logger().error('No motion plan to execute')
            return

        self.current_state = MotionState.EXECUTING
        self.publish_status('Executing motion plan')

        # Start following the plan
        self.follow_plan()

    def follow_plan(self):
        """Follow the current motion plan"""
        if not self.motion_plan:
            return

        # For this example, we'll just move toward the first waypoint in the plan
        if self.motion_plan:
            next_waypoint = self.motion_plan[0]
            self.move_to_waypoint(next_waypoint)

    def move_to_waypoint(self, waypoint):
        """Move robot to a specific waypoint"""
        if not self.current_pose:
            return

        # Calculate direction to waypoint
        dx = waypoint.position.x - self.current_pose.position.x
        dy = waypoint.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate required rotation
        target_yaw = math.atan2(dy, dx)
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Calculate angle difference
        angle_diff = target_yaw - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Create velocity command
        cmd_vel = Twist()

        # If we need to rotate significantly, rotate first
        if abs(angle_diff) > 0.2:  # 0.2 rad = ~11 degrees
            cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 1.5))
        else:
            # Move forward toward the waypoint
            cmd_vel.linear.x = min(self.linear_speed, distance * 0.5)  # Proportional to distance
            cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2.0))

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def check_progress_to_goal(self):
        """Check if we've reached the goal"""
        if not self.current_pose or not self.current_goal:
            return

        # Calculate distance to goal
        dx = self.current_goal.position.x - self.current_pose.position.x
        dy = self.current_goal.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance < self.goal_tolerance:
            self.get_logger().info('Reached goal position')
            self.current_state = MotionState.COMPLETED
            self.publish_status('Goal reached')
            self.stop_motion()

    def check_for_obstacles(self):
        """Check for obstacles in the path and adapt"""
        if not self.current_scan:
            return

        # Check if there are obstacles directly in front
        front_distances = self.get_front_distances(self.current_scan)
        min_front_dist = min(front_distances) if front_distances else float('inf')

        if min_front_dist < self.obstacle_threshold:
            self.get_logger().info('Obstacle detected, adapting motion')
            self.current_state = MotionState.ADAPTING
            self.publish_status('Adapting to obstacle')

            # Stop and replan
            self.stop_motion()
            # In a real system, you would replan here
            # For this example, just wait a moment then continue
            self.create_timer(1.0, self.resume_after_adaptation)

    def get_front_distances(self, scan_msg):
        """Get distances in the front sector of the scan"""
        if not scan_msg.ranges:
            return []

        # Get points in front (±30 degrees)
        front_start = len(scan_msg.ranges) // 2 - len(scan_msg.ranges) // 12
        front_end = len(scan_msg.ranges) // 2 + len(scan_msg.ranges) // 12

        front_ranges = []
        for i in range(front_start, front_end):
            idx = i % len(scan_msg.ranges)
            if scan_msg.range_min < scan_msg.ranges[idx] < scan_msg.range_max:
                front_ranges.append(scan_msg.ranges[idx])

        return front_ranges

    def resume_after_adaptation(self):
        """Resume motion after obstacle adaptation"""
        self.current_state = MotionState.EXECUTING
        self.publish_status('Resuming motion after obstacle adaptation')
        if self.motion_plan:
            self.follow_plan()

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def build_grid_map(self, map_msg):
        """Build a grid map representation from occupancy grid"""
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution

        # Convert the 1D map data to 2D grid
        grid = np.array(map_msg.data).reshape((height, width))
        self.grid_map = {
            'grid': grid,
            'resolution': resolution,
            'origin': (map_msg.info.origin.position.x, map_msg.info.origin.position.y),
            'width': width,
            'height': height
        }

    def stop_motion(self):
        """Stop all motion"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
        self.publish_status('Motion stopped')

    def pause_motion(self):
        """Pause motion"""
        self.current_state = MotionState.IDLE
        self.stop_motion()
        self.publish_status('Motion paused')

    def resume_motion(self):
        """Resume motion"""
        if self.motion_plan:
            self.current_state = MotionState.EXECUTING
            self.publish_status('Motion resumed')
            self.follow_plan()

    def plan_to_position(self, x, y):
        """Plan motion to a specific position"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0  # No rotation

        self.current_goal = goal_msg.pose
        self.plan_motion_to_goal()

    def publish_status(self, status_msg):
        """Publish status message"""
        msg = String()
        msg.data = status_msg
        self.motion_status_pub.publish(msg)

    def publish_motion_plan(self):
        """Publish detailed motion plan information"""
        plan_info = {
            'state': self.current_state.value,
            'num_waypoints': len(self.motion_plan),
            'total_distance': self.calculate_plan_distance(),
            'status': 'published'
        }

        plan_msg = String()
        plan_msg.data = json.dumps(plan_info)
        self.motion_plan_pub.publish(plan_msg)

    def publish_path(self):
        """Publish the planned path"""
        if not self.motion_plan:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = [PoseStamped(pose=wp) for wp in self.motion_plan]

        self.path_pub.publish(path_msg)

    def calculate_plan_distance(self):
        """Calculate total distance of the motion plan"""
        if len(self.motion_plan) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(self.motion_plan)):
            prev_wp = self.motion_plan[i-1]
            curr_wp = self.motion_plan[i]

            dx = curr_wp.position.x - prev_wp.position.x
            dy = curr_wp.position.y - prev_wp.position.y
            dz = curr_wp.position.z - prev_wp.position.z

            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_distance += distance

        return total_distance


def main(args=None):
    rclpy.init(args=args)

    motion_node = MotionPlannerNode()

    try:
        # Example: Set a goal after a delay
        def set_example_goal():
            goal_msg = PoseStamped()
            goal_msg.header.stamp = motion_node.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = 2.0
            goal_msg.pose.position.y = 2.0
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0
            motion_node.goal_callback(goal_msg)

        # Set goal after 3 seconds
        motion_node.create_timer(3.0, set_example_goal)

        rclpy.spin(motion_node)
    except KeyboardInterrupt:
        pass
    finally:
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()