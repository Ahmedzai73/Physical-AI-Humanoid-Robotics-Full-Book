#!/usr/bin/env python3

"""
Nav2 Integration for LLM-Generated Navigation Routes
Module 4: Vision-Language-Action (VLA) - Physical AI & Humanoid Robotics Textbook

This module integrates Nav2 with LLM-generated navigation routes, allowing the AI to plan
complex navigation tasks that are executed safely by the robot.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
import json
import math
from rclpy.action import ActionClient
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from typing import List, Dict, Any, Optional


class Nav2IntegrationNode(Node):
    """
    Node that integrates Nav2 with LLM-generated navigation routes
    """
    def __init__(self):
        super().__init__('nav2_integration_node')

        # Action client for NavigateToPose
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers
        self.navigation_status_pub = self.create_publisher(
            String,
            '/nav2_integration/status',
            10
        )

        self.navigation_feedback_pub = self.create_publisher(
            String,
            '/nav2_integration/feedback',
            10
        )

        # Subscribers
        self.llm_route_sub = self.create_subscription(
            String,
            '/vla/llm_navigation_route',
            self.llm_route_callback,
            10
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            10
        )

        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal state
        self.current_goal = None
        self.navigation_active = False
        self.emergency_stop_active = False

        # Navigation parameters
        self.min_waypoint_distance = 0.5  # meters
        self.max_route_length = 20  # maximum waypoints in a route

        self.get_logger().info('Nav2 Integration Node initialized')

    def llm_route_callback(self, msg):
        """
        Process LLM-generated navigation route
        """
        try:
            route_data = json.loads(msg.data)
            route_type = route_data.get('type', 'waypoints')

            if route_type == 'waypoints':
                waypoints = route_data.get('waypoints', [])
                self.execute_waypoint_route(waypoints)
            elif route_type == 'destination':
                destination = route_data.get('destination', {})
                self.navigate_to_destination(destination)
            elif route_type == 'explore':
                self.explore_area()
            else:
                self.get_logger().warn(f'Unknown route type: {route_type}')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in navigation route: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing LLM route: {str(e)}')

    def execute_waypoint_route(self, waypoints: List[Dict[str, float]]):
        """
        Execute a route with multiple waypoints
        """
        if not waypoints:
            self.get_logger().warn('Empty waypoint route received')
            return

        if len(waypoints) > self.max_route_length:
            self.get_logger().warn(f'Too many waypoints: {len(waypoints)}, limiting to {self.max_route_length}')
            waypoints = waypoints[:self.max_route_length]

        self.get_logger().info(f'Executing route with {len(waypoints)} waypoints')

        # Process each waypoint in sequence
        for i, waypoint in enumerate(waypoints):
            if self.emergency_stop_active:
                self.get_logger().warn('Emergency stop active, aborting navigation')
                break

            try:
                # Convert waypoint to PoseStamped
                pose_stamped = self.waypoint_to_pose(waypoint)

                # Navigate to waypoint
                success = self.navigate_to_pose_blocking(pose_stamped)

                if success:
                    status_msg = String()
                    status_msg.data = f'Waypoint {i+1}/{len(waypoints)} reached: ({waypoint["x"]:.2f}, {waypoint["y"]:.2f})'
                    self.navigation_status_pub.publish(status_msg)

                    self.get_logger().info(f'Waypoint {i+1}/{len(waypoints)} reached')
                else:
                    status_msg = String()
                    status_msg.data = f'Failed to reach waypoint {i+1}: ({waypoint["x"]:.2f}, {waypoint["y"]:.2f})'
                    self.navigation_status_pub.publish(status_msg)

                    self.get_logger().error(f'Failed to reach waypoint {i+1}')
                    break  # Stop route execution if waypoint fails

            except Exception as e:
                self.get_logger().error(f'Error navigating to waypoint {i+1}: {str(e)}')
                break

        # Publish completion status
        completion_msg = String()
        completion_msg.data = f'Route execution completed. Reached {i+1}/{len(waypoints)} waypoints.'
        self.navigation_status_pub.publish(completion_msg)

    def navigate_to_destination(self, destination: Dict[str, Any]):
        """
        Navigate to a single destination
        """
        try:
            # Convert destination to PoseStamped
            pose_stamped = self.destination_to_pose(destination)

            # Navigate to destination
            success = self.navigate_to_pose_blocking(pose_stamped)

            status_msg = String()
            if success:
                status_msg.data = f'Destination reached: {destination.get("name", "unnamed")}'
                self.get_logger().info(f'Destination reached: {destination.get("name", "unnamed")}')
            else:
                status_msg.data = f'Failed to reach destination: {destination.get("name", "unnamed")}'
                self.get_logger().error(f'Failed to reach destination: {destination.get("name", "unnamed")}')

            self.navigation_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error navigating to destination: {str(e)}')

    def explore_area(self):
        """
        Execute exploration behavior
        """
        self.get_logger().info('Starting area exploration')

        # For now, implement a simple spiral exploration pattern
        # In a real implementation, this would use frontier-based exploration
        exploration_pattern = self.generate_spiral_pattern()

        for pose in exploration_pattern:
            if self.emergency_stop_active:
                break

            success = self.navigate_to_pose_blocking(pose)
            if success:
                self.get_logger().info('Exploration waypoint reached')
            else:
                self.get_logger().warn('Could not reach exploration waypoint, continuing')
                continue

        status_msg = String()
        status_msg.data = 'Area exploration completed'
        self.navigation_status_pub.publish(status_msg)

    def generate_spiral_pattern(self) -> List[PoseStamped]:
        """
        Generate a spiral exploration pattern centered at current position
        """
        try:
            # Get current robot position
            current_pose = self.get_current_pose()
            if not current_pose:
                self.get_logger().warn('Could not get current pose for exploration')
                return []

            # Generate spiral pattern
            pattern = []
            center_x = current_pose.position.x
            center_y = current_pose.position.y
            radius_increment = 0.5  # meters
            angle_increment = math.pi / 8  # radians

            max_radius = 5.0  # meters
            current_radius = 0.5
            current_angle = 0.0

            while current_radius < max_radius:
                x = center_x + current_radius * math.cos(current_angle)
                y = center_y + current_radius * math.sin(current_angle)

                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                # Point in direction of movement
                pose.pose.orientation.z = math.sin(current_angle / 2)
                pose.pose.orientation.w = math.cos(current_angle / 2)

                pattern.append(pose)

                current_angle += angle_increment
                current_radius += radius_increment * angle_increment / (2 * math.pi)

            return pattern

        except Exception as e:
            self.get_logger().error(f'Error generating exploration pattern: {str(e)}')
            return []

    def navigate_to_pose_blocking(self, pose_stamped: PoseStamped) -> bool:
        """
        Navigate to pose using action client with blocking behavior
        """
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        # Send goal
        self.get_logger().info(f'Navigating to pose: ({pose_stamped.pose.position.x:.2f}, {pose_stamped.pose.position.y:.2f})')

        future = self.nav_to_pose_client.send_goal_async(goal_msg)

        # Wait for result
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            return False

        # Get result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        status = result.status

        # Check status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded')
            return True
        else:
            self.get_logger().warn(f'Navigation failed with status: {status}')
            return False

    def waypoint_to_pose(self, waypoint: Dict[str, float]) -> PoseStamped:
        """
        Convert waypoint dictionary to PoseStamped message
        """
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'

        pose.pose.position.x = float(waypoint.get('x', 0.0))
        pose.pose.position.y = float(waypoint.get('y', 0.0))
        pose.pose.position.z = float(waypoint.get('z', 0.0))

        # Set orientation (default to facing forward)
        pose.pose.orientation.w = 1.0

        return pose

    def destination_to_pose(self, destination: Dict[str, Any]) -> PoseStamped:
        """
        Convert destination dictionary to PoseStamped message
        """
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'

        pose.pose.position.x = float(destination.get('x', 0.0))
        pose.pose.position.y = float(destination.get('y', 0.0))
        pose.pose.position.z = float(destination.get('z', 0.0))

        # If orientation is specified, use it; otherwise face forward
        orientation = destination.get('orientation', {})
        if orientation:
            pose.pose.orientation.x = float(orientation.get('x', 0.0))
            pose.pose.orientation.y = float(orientation.get('y', 0.0))
            pose.pose.orientation.z = float(orientation.get('z', 0.0))
            pose.pose.orientation.w = float(orientation.get('w', 1.0))
        else:
            pose.pose.orientation.w = 1.0  # Default orientation

        return pose

    def get_current_pose(self) -> Optional[Point]:
        """
        Get the current robot pose from TF
        """
        try:
            # Look up transform from base_link to map
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            pose = Point()
            pose.x = transform.transform.translation.x
            pose.y = transform.transform.translation.y
            pose.z = transform.transform.translation.z

            return pose
        except Exception as e:
            self.get_logger().warn(f'Could not get current pose: {str(e)}')
            return None

    def emergency_stop_callback(self, msg):
        """
        Handle emergency stop commands
        """
        self.emergency_stop_active = msg.data
        if self.emergency_stop_active:
            self.get_logger().warn('Emergency stop activated, aborting navigation')
            # Cancel any active navigation goals
            self.cancel_current_navigation()

    def cancel_current_navigation(self):
        """
        Cancel any currently active navigation goal
        """
        try:
            # This would cancel the current navigation goal
            # Implementation depends on Nav2 version and capabilities
            self.get_logger().info('Attempting to cancel current navigation')
        except Exception as e:
            self.get_logger().error(f'Error canceling navigation: {str(e)}')

    def validate_route(self, waypoints: List[Dict[str, float]]) -> bool:
        """
        Validate that the route is reasonable (not too long, not too dense, etc.)
        """
        if len(waypoints) == 0:
            return False

        if len(waypoints) > self.max_route_length:
            self.get_logger().warn(f'Route too long: {len(waypoints)} waypoints, max: {self.max_route_length}')
            return False

        # Check waypoint density (minimum distance between consecutive waypoints)
        for i in range(1, len(waypoints)):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]

            dist = math.sqrt(
                (curr_wp['x'] - prev_wp['x'])**2 +
                (curr_wp['y'] - prev_wp['y'])**2
            )

            if dist < self.min_waypoint_distance:
                self.get_logger().warn(f'Waypoints too close together: {dist:.2f}m < {self.min_waypoint_distance:.2f}m')
                return False

        return True


def main(args=None):
    rclpy.init(args=args)

    nav2_integration_node = Nav2IntegrationNode()

    try:
        rclpy.spin(nav2_integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav2_integration_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()