#!/usr/bin/env python3

"""
Humanoid Robot Sensor Processor Node

This node demonstrates basic sensor processing for humanoid robots
for the Physical AI & Humanoid Robotics textbook.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32


class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Subscribers for various sensor data
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers for processed sensor data
        self.balance_pub = self.create_publisher(Float32, '/balance_state', 10)
        self.obstacle_distance_pub = self.create_publisher(Float32, '/obstacle_distance', 10)
        self.foot_contact_pub = self.create_publisher(PointStamped, '/foot_contact', 10)

        # Timer for processing loop
        self.timer = self.create_timer(0.05, self.process_sensors)

        self.get_logger().info('Sensor Processor node initialized')

        # Store sensor data
        self.joint_states = JointState()
        self.imu_data = Imu()
        self.scan_data = LaserScan()
        self.last_joint_update = self.get_clock().now()
        self.last_imu_update = self.get_clock().now()
        self.last_scan_update = self.get_clock().now()

    def joint_state_callback(self, msg):
        """Handle joint state messages"""
        self.joint_states = msg
        self.last_joint_update = self.get_clock().now()
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg
        self.last_imu_update = self.get_clock().now()
        self.get_logger().debug(f'Received IMU data - orientation: ({msg.orientation.x:.3f}, {msg.orientation.y:.3f}, {msg.orientation.z:.3f}, {msg.orientation.w:.3f})')

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg
        self.last_scan_update = self.get_clock().now()
        self.get_logger().debug(f'Received laser scan with {len(msg.ranges)} ranges')

    def process_sensors(self):
        """Process sensor data to extract meaningful information"""
        # Calculate balance state from IMU data
        balance_msg = Float32()

        # Simplified balance calculation using roll and pitch from IMU
        roll = self.imu_data.orientation.x  # This is simplified - real calculation would use quaternion
        pitch = self.imu_data.orientation.y

        # Calculate a balance metric (0.0 = perfectly balanced, 1.0 = completely unbalanced)
        balance_metric = min(1.0, abs(roll) + abs(pitch))
        balance_msg.data = balance_metric

        self.balance_pub.publish(balance_msg)

        # Calculate closest obstacle distance from laser scan
        obstacle_msg = Float32()
        if len(self.scan_data.ranges) > 0:
            valid_ranges = [r for r in self.scan_data.ranges if r > 0 and r < float('inf')]
            if valid_ranges:
                obstacle_msg.data = min(valid_ranges)
            else:
                obstacle_msg.data = float('inf')
        else:
            obstacle_msg.data = float('inf')

        self.obstacle_distance_pub.publish(obstacle_msg)

        # Simulate foot contact detection
        foot_contact_msg = PointStamped()
        foot_contact_msg.header.stamp = self.get_clock().now().to_msg()
        foot_contact_msg.header.frame_id = 'base_link'
        # For this example, we'll just publish a point at the robot's center
        foot_contact_msg.point.x = 0.0
        foot_contact_msg.point.y = 0.0
        foot_contact_msg.point.z = 0.0

        self.foot_contact_pub.publish(foot_contact_msg)

        # Log sensor status
        now = self.get_clock().now()
        joint_age = (now - self.last_joint_update).nanoseconds / 1e9
        imu_age = (now - self.last_imu_update).nanoseconds / 1e9
        scan_age = (now - self.last_scan_update).nanoseconds / 1e9

        if joint_age > 1.0 or imu_age > 1.0 or scan_age > 1.0:
            self.get_logger().warn(f'Sensor data may be stale - joint: {joint_age:.1f}s, imu: {imu_age:.1f}s, scan: {scan_age:.1f}s')


def main(args=None):
    rclpy.init(args=args)

    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()