#!/usr/bin/env python3

"""
Basic Publisher Node for Physical AI & Humanoid Robotics Textbook
Module 1: The Robotic Nervous System (ROS 2)

This node demonstrates basic topic publishing in ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import random


class BasicPublisher(Node):
    def __init__(self):
        super().__init__('basic_publisher')

        # Create publishers
        self.string_publisher = self.create_publisher(String, 'robot_status', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create timer for publishing messages
        self.timer = self.create_timer(0.5, self.publish_data)

        self.get_logger().info('Basic Publisher node initialized')

    def publish_data(self):
        # Publish a status message
        status_msg = String()
        status_msg.data = f'Robot operational at {self.get_clock().now().seconds_as_nanoseconds()}'
        self.string_publisher.publish(status_msg)

        # Publish a simple velocity command
        twist_msg = Twist()
        twist_msg.linear.x = 0.2  # Move forward at 0.2 m/s
        twist_msg.angular.z = random.uniform(-0.2, 0.2)  # Random turn
        self.cmd_vel_publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)

    publisher = BasicPublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()