# Copyright 2025 Physical AI & Humanoid Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Navigation node for Physical AI & Humanoid Robotics examples.
This node demonstrates navigation concepts from the textbook using Nav2.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool
import time


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Action client for NavigateToPose
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # Publisher for navigation status
        self.nav_status_publisher = self.create_publisher(
            Bool, '/navigation_active', 10
        )

        # Timer to periodically check navigation status
        self.timer = self.create_timer(1.0, self.check_navigation_status)

        self.navigation_active = False

        self.get_logger().info('Navigation Node initialized')

    def send_goal_pose(self, x: float, y: float, theta: float):
        """Send a goal pose to the Nav2 stack"""
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta (yaw) to quaternion
        from math import sin, cos
        sin_theta = sin(theta / 2.0)
        cos_theta = cos(theta / 2.0)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = sin_theta
        goal_msg.pose.pose.orientation.w = cos_theta

        self.get_logger().info(f'Sending navigation goal: ({x}, {y}, {theta})')

        # Send goal and handle response
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.navigation_active = True
        self.nav_status_publisher.publish(Bool(data=True))

        # Get result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        self.navigation_active = False
        self.nav_status_publisher.publish(Bool(data=False))

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Navigation feedback: {feedback.current_pose}')

    def check_navigation_status(self):
        """Check navigation status and publish"""
        status_msg = Bool()
        status_msg.data = self.navigation_active
        self.nav_status_publisher.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    navigation_node = NavigationNode()

    # Example: Send a simple navigation goal after a delay
    def send_example_goal():
        time.sleep(2)  # Wait for systems to initialize
        navigation_node.send_goal_pose(1.0, 1.0, 0.0)  # Go to (1,1) with 0 rotation

    # Schedule the example goal
    navigation_node.create_timer(0.1, send_example_goal)

    rclpy.spin(navigation_node)
    navigation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()