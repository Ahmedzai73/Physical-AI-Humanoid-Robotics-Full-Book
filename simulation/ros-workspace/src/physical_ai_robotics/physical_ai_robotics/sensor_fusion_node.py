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
Sensor fusion node for Physical AI & Humanoid Robotics examples.
This node demonstrates sensor fusion concepts from the textbook.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
from typing import List, Optional


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscribers for different sensor data
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.imu_subscriber = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Publisher for fused sensor data
        self.fused_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10
        )

        self.get_logger().info('Sensor Fusion Node initialized')

        # Initialize sensor data storage
        self.latest_scan: Optional[LaserScan] = None
        self.latest_image: Optional[Image] = None
        self.latest_imu: Optional[Imu] = None
        self.latest_odom: Optional[Odometry] = None

    def scan_callback(self, msg: LaserScan):
        """Callback for laser scan data"""
        self.latest_scan = msg
        self.get_logger().debug('Received laser scan data')

    def image_callback(self, msg: Image):
        """Callback for camera image data"""
        self.latest_image = msg
        self.get_logger().debug('Received image data')

    def imu_callback(self, msg: Imu):
        """Callback for IMU data"""
        self.latest_imu = msg
        self.get_logger().debug('Received IMU data')

    def odom_callback(self, msg: Odometry):
        """Callback for odometry data"""
        self.latest_odom = msg
        self.get_logger().debug('Received odometry data')

    def fuse_sensor_data(self) -> PoseWithCovarianceStamped:
        """Fuse sensor data to estimate robot pose"""
        fused_pose = PoseWithCovarianceStamped()
        fused_pose.header.stamp = self.get_clock().now().to_msg()
        fused_pose.header.frame_id = 'map'

        # Simple sensor fusion - in practice, this would use more sophisticated
        # algorithms like Kalman filters or particle filters
        if self.latest_odom:
            fused_pose.pose.pose = self.latest_odom.pose.pose
            # Set covariance based on sensor confidence
            fused_pose.pose.covariance = list(np.eye(6).flatten())  # Identity matrix

        return fused_pose

    def publish_fused_data(self):
        """Publish fused sensor data"""
        fused_pose = self.fuse_sensor_data()
        self.fused_pose_publisher.publish(fused_pose)


def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()

    # Timer to periodically publish fused data
    timer = sensor_fusion_node.create_timer(0.1, sensor_fusion_node.publish_fused_data)

    rclpy.spin(sensor_fusion_node)
    sensor_fusion_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()