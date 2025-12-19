#!/usr/bin/env python3

"""
Sensor Fusion Node for Physical AI & Humanoid Robotics Textbook
Module 3: The AI-Robot Brain (NVIDIA Isaac™)

This node demonstrates sensor fusion combining vision, LIDAR, and IMU data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String, Float32
from vision_msgs.msg import Detection2DArray
import numpy as np
from collections import deque
import threading
import time


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscribers for different sensors
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/vision/detections',
            self.detection_callback,
            10
        )

        # Publishers for fused data
        self.environment_map_pub = self.create_publisher(
            String,
            '/sensor_fusion/environment_map',
            10
        )

        self.object_tracking_pub = self.create_publisher(
            String,
            '/sensor_fusion/object_tracking',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/sensor_fusion/status',
            10
        )

        self.get_logger().info('Sensor Fusion Node initialized')

        # Data storage
        self.latest_image = None
        self.latest_scan = None
        self.latest_imu = None
        self.latest_detections = None

        # Lock for thread safety
        self.data_lock = threading.Lock()

        # Timers for processing
        self.fusion_timer = self.create_timer(0.1, self.perform_sensor_fusion)  # 10 Hz

        # Statistics
        self.fusion_count = 0

    def image_callback(self, msg):
        """Handle incoming image data"""
        with self.data_lock:
            self.latest_image = msg

    def scan_callback(self, msg):
        """Handle incoming LIDAR scan data"""
        with self.data_lock:
            self.latest_scan = msg

    def imu_callback(self, msg):
        """Handle incoming IMU data"""
        with self.data_lock:
            self.latest_imu = msg

    def detection_callback(self, msg):
        """Handle incoming vision detections"""
        with self.data_lock:
            self.latest_detections = msg

    def perform_sensor_fusion(self):
        """Perform sensor fusion on available data"""
        with self.data_lock:
            # Get current data
            current_image = self.latest_image
            current_scan = self.latest_scan
            current_imu = self.latest_imu
            current_detections = self.latest_detections

        # Perform fusion logic
        environment_map = self._create_environment_map(current_scan, current_detections)
        object_status = self._track_objects(current_detections, current_scan)
        robot_state = self._integrate_robot_state(current_imu)

        # Publish fused results
        env_map_msg = String()
        env_map_msg.data = environment_map
        self.environment_map_pub.publish(env_map_msg)

        obj_track_msg = String()
        obj_track_msg.data = object_status
        self.object_tracking_pub.publish(obj_track_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f'Sensor fusion completed. Objects: {len(current_detections.detections) if current_detections else 0}, Fusion count: {self.fusion_count}'
        self.status_pub.publish(status_msg)

        self.fusion_count += 1

        self.get_logger().debug(f'Sensor fusion completed - {object_status}')

    def _create_environment_map(self, scan, detections):
        """Create a combined environment map from LIDAR and vision data"""
        if scan is None:
            return "No LIDAR data available"

        # Basic environment map from LIDAR
        valid_ranges = [r for r in scan.ranges if r > scan.range_min and r < scan.range_max]
        if not valid_ranges:
            return "Environment: Clear (no obstacles detected)"

        # Find closest obstacle
        closest_obstacle = min(valid_ranges) if valid_ranges else float('inf')

        # Count obstacles in different ranges
        close_obstacles = len([r for r in valid_ranges if r < 1.0])
        medium_obstacles = len([r for r in valid_ranges if 1.0 <= r < 3.0])
        far_obstacles = len([r for r in valid_ranges if r >= 3.0])

        # Add vision information if available
        vision_info = ""
        if detections is not None and len(detections.detections) > 0:
            object_classes = [det.results[0].hypothesis.class_id if det.results else "unknown" for det in detections.detections]
            vision_info = f", Vision objects: {', '.join(set(object_classes))}"

        return f"Environment: Close {close_obstacles}, Medium {medium_obstacles}, Far {far_obstacles}{vision_info}"

    def _track_objects(self, detections, scan):
        """Track objects using vision and LIDAR data"""
        if detections is None or len(detections.detections) == 0:
            return "Object tracking: No objects detected"

        # For simplicity, just return the detected object types
        object_types = []
        if detections:
            for detection in detections.detections:
                if detection.results:
                    object_types.append(detection.results[0].hypothesis.class_id)

        unique_objects = list(set(object_types))
        return f"Objects tracked: {', '.join(unique_objects)} (count: {len(unique_objects)})"

    def _integrate_robot_state(self, imu):
        """Integrate robot state from IMU data"""
        if imu is None:
            return "Robot state: IMU data unavailable"

        # Extract orientation (simplified)
        orientation = imu.orientation
        # Calculate approximate roll and pitch
        sinr_cosp = 2 * (orientation.w * orientation.x + orientation.y * orientation.z)
        cosr_cosp = 1 - 2 * (orientation.x * orientation.x + orientation.y * orientation.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (orientation.w * orientation.y - orientation.z * orientation.x)
        pitch = np.arcsin(sinp)

        return f"Robot state: Roll {np.degrees(roll):.2f}°, Pitch {np.degrees(pitch):.2f}°"

    def get_fused_environment_state(self):
        """Get the current fused environment state"""
        with self.data_lock:
            return {
                'image': self.latest_image,
                'scan': self.latest_scan,
                'imu': self.latest_imu,
                'detections': self.latest_detections
            }


def main(args=None):
    rclpy.init(args=args)

    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()