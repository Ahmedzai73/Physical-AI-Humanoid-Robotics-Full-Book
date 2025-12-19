#!/usr/bin/env python3

"""
Basic Subscriber Node for Physical AI & Humanoid Robotics Textbook
Module 1: The Robotic Nervous System (ROS 2)

This node demonstrates basic topic subscription in ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class BasicSubscriber(Node):
    def __init__(self):
        super().__init__('basic_subscriber')

        # Create subscribers
        self.status_subscriber = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10
        )

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.get_logger().info('Basic Subscriber node initialized')

    def status_callback(self, msg):
        self.get_logger().info(f'Received status: {msg.data}')

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f'Received velocity command - Linear: {msg.linear.x}, Angular: {msg.angular.z}')


def main(args=None):
    rclpy.init(args=args)

    subscriber = BasicSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()