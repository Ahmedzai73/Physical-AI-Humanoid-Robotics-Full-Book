#!/usr/bin/env python3

"""
Humanoid Robot Controller Node

This node demonstrates basic humanoid robot control concepts
for the Physical AI & Humanoid Robotics textbook.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publisher for robot status
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Humanoid Controller node initialized')

        # Robot state
        self.current_twist = Twist()
        self.joint_positions = {}

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.current_twist = msg
        self.get_logger().info(f'Received velocity command: linear={msg.linear.x}, angular={msg.angular.z}')

    def control_loop(self):
        """Main control loop"""
        # Create joint state message
        joint_msg = JointState()
        joint_msg.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        # Calculate joint positions based on desired movement
        # This is a simplified example - real controllers would use inverse kinematics
        joint_msg.position = [0.0] * len(joint_msg.name)

        # Apply some basic movement logic based on cmd_vel
        if abs(self.current_twist.linear.x) > 0.01:
            # Moving forward/backward - adjust leg joints
            joint_msg.position[0] = self.current_twist.linear.x * 0.1  # left hip
            joint_msg.position[3] = self.current_twist.linear.x * 0.1  # right hip

        if abs(self.current_twist.angular.z) > 0.01:
            # Turning - adjust hip joints differentially
            joint_msg.position[0] = self.current_twist.angular.z * 0.05  # left hip
            joint_msg.position[3] = -self.current_twist.angular.z * 0.05  # right hip

        joint_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(joint_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Moving: linear=({self.current_twist.linear.x:.2f}, {self.current_twist.linear.y:.2f}, {self.current_twist.linear.z:.2f}), " \
                         f"angular=({self.current_twist.angular.x:.2f}, {self.current_twist.angular.y:.2f}, {self.current_twist.angular.z:.2f})"
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)

    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()