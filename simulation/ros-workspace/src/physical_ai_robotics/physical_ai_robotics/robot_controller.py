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
Robot controller node for Physical AI & Humanoid Robotics examples.
This node demonstrates robot control concepts from the textbook.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import math


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for robot velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publishers for joint positions (for simulation)
        self.joint_publishers = {}
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        for joint_name in joint_names:
            self.joint_publishers[joint_name] = self.create_publisher(
                Float64, f'/{joint_name}_position_controller/commands', 10
            )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Robot Controller node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state messages"""
        self.get_logger().info(f'Received joint states: {len(msg.name)} joints')

    def control_loop(self):
        """Main control loop"""
        # Example: Send a simple velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd_vel.angular.z = 0.2  # Rotate at 0.2 rad/s

        self.cmd_vel_publisher.publish(cmd_vel)

        # Example: Send joint position commands
        for i, (joint_name, publisher) in enumerate(self.joint_publishers.items()):
            joint_cmd = Float64()
            # Create a sinusoidal motion pattern
            joint_cmd.data = math.sin(self.get_clock().now().nanoseconds * 0.000000001 + i) * 0.5
            publisher.publish(joint_cmd)


def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()