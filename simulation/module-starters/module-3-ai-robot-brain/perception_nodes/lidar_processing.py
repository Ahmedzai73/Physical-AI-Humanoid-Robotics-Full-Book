#!/usr/bin/env python3

"""
LIDAR Processing Node for Physical AI & Humanoid Robotics Textbook
Module 3: The AI-Robot Brain (NVIDIA Isaac™)

This node demonstrates LIDAR processing for environment understanding.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PointStamped
import numpy as np
from collections import deque


class LidarProcessingNode(Node):
    def __init__(self):
        super().__init__('lidar_processing_node')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.obstacle_pub = self.create_publisher(
            String,
            '/lidar/obstacles',
            10
        )

        self.distance_pub = self.create_publisher(
            Float32,
            '/lidar/closest_obstacle',
            10
        )

        self.front_distance_pub = self.create_publisher(
            Float32,
            '/lidar/front_distance',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/lidar/status',
            10
        )

        self.get_logger().info('LIDAR Processing Node initialized')

        # Store recent scan data for filtering
        self.scan_buffer = deque(maxlen=5)

    def scan_callback(self, msg):
        """Process incoming LIDAR scan data"""
        # Add current scan to buffer
        self.scan_buffer.append(msg)

        # Process the scan data
        closest_distance = self._find_closest_obstacle(msg)
        front_distance = self._get_front_distance(msg)

        # Publish results
        closest_msg = Float32()
        closest_msg.data = closest_distance
        self.distance_pub.publish(closest_msg)

        front_msg = Float32()
        front_msg.data = front_distance
        self.front_distance_pub.publish(front_msg)

        # Detect obstacles in specific regions
        obstacles_str = self._detect_obstacles_in_regions(msg)
        obstacles_msg = String()
        obstacles_msg.data = obstacles_str
        self.obstacle_pub.publish(obstacles_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f'Processed scan with {len(msg.ranges)} points. Closest: {closest_distance:.2f}m, Front: {front_distance:.2f}m'
        self.status_pub.publish(status_msg)

        self.get_logger().debug(f'LIDAR processing complete - {obstacles_str}')

    def _find_closest_obstacle(self, scan_msg):
        """Find the closest obstacle in the scan"""
        valid_ranges = [r for r in scan_msg.ranges if r > scan_msg.range_min and r < scan_msg.range_max]
        if valid_ranges:
            return min(valid_ranges)
        else:
            return float('inf')

    def _get_front_distance(self, scan_msg):
        """Get distance to obstacle directly in front (within ±15 degrees)"""
        # Calculate indices for front region (±15 degrees)
        angle_increment = scan_msg.angle_increment
        front_center_idx = int(len(scan_msg.ranges) / 2)
        angle_range = int(15 * np.pi / 180 / angle_increment)  # Convert 15 degrees to indices

        start_idx = max(0, front_center_idx - angle_range)
        end_idx = min(len(scan_msg.ranges), front_center_idx + angle_range)

        front_ranges = scan_msg.ranges[start_idx:end_idx]
        valid_ranges = [r for r in front_ranges if r > scan_msg.range_min and r < scan_msg.range_max]

        if valid_ranges:
            return min(valid_ranges)
        else:
            return float('inf')

    def _detect_obstacles_in_regions(self, scan_msg):
        """Detect obstacles in different regions (front, left, right, back)"""
        num_ranges = len(scan_msg.ranges)
        valid_ranges = [
            r if scan_msg.range_min < r < scan_msg.range_max else float('inf')
            for r in scan_msg.ranges
        ]

        # Define regions
        front_idx = slice(num_ranges//2 - num_ranges//8, num_ranges//2 + num_ranges//8)
        left_idx = slice(num_ranges*3//4 - num_ranges//8, num_ranges*3//4 + num_ranges//8)
        right_idx = slice(num_ranges//4 - num_ranges//8, num_ranges//4 + num_ranges//8)
        back_idx = slice(0, num_ranges//8) + slice(num_ranges - num_ranges//8, num_ranges)

        regions = {
            'front': valid_ranges[num_ranges//2 - num_ranges//8 : num_ranges//2 + num_ranges//8],
            'left': valid_ranges[num_ranges*3//4 - num_ranges//8 : num_ranges*3//4 + num_ranges//8],
            'right': valid_ranges[num_ranges//4 - num_ranges//8 : num_ranges//4 + num_ranges//8],
            'back': valid_ranges[:num_ranges//8] + valid_ranges[num_ranges - num_ranges//8:]
        }

        obstacles = []
        for region_name, region_ranges in regions.items():
            valid_region_ranges = [r for r in region_ranges if r != float('inf')]
            if valid_region_ranges and min(valid_region_ranges) < 1.0:  # Obstacle within 1 meter
                obstacles.append(f"{region_name}")

        if obstacles:
            return f"Obstacles detected in: {', '.join(obstacles)}"
        else:
            return "No obstacles detected in close proximity"

    def get_filtered_scan(self):
        """Get a filtered scan from the buffer"""
        if not self.scan_buffer:
            return None

        # Simple averaging filter
        scans = list(self.scan_buffer)
        if len(scans) == 0:
            return None

        # Average the ranges
        avg_ranges = np.array([scan.ranges for scan in scans])
        filtered_ranges = np.mean(avg_ranges, axis=0)

        # Return the most recent scan with filtered ranges
        result = scans[-1]
        result.ranges = filtered_ranges.tolist()
        return result


def main(args=None):
    rclpy.init(args=args)

    lidar_node = LidarProcessingNode()

    try:
        rclpy.spin(lidar_node)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()