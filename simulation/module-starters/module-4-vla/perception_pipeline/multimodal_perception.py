#!/usr/bin/env python3

"""
Multimodal Perception Pipeline for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module demonstrates multimodal perception combining vision, language, and action.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import json


class MultimodalPerceptionNode(Node):
    def __init__(self):
        super().__init__('multimodal_perception_node')

        # Create CvBridge for image conversion
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/vla/command',
            self.command_callback,
            10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/detections',
            10
        )

        self.scene_description_pub = self.create_publisher(
            String,
            '/perception/scene_description',
            10
        )

        self.object_query_pub = self.create_publisher(
            String,
            '/perception/object_query_response',
            10
        )

        self.get_logger().info('Multimodal Perception Node initialized')

        # Store data
        self.camera_info = None
        self.latest_image = None
        self.latest_scan = None
        self.pending_command = None

        # Object detection parameters
        self.colors_to_detect = {
            'red': ([0, 0, 100], [50, 50, 255]),
            'green': ([0, 100, 0], [50, 255, 50]),
            'blue': ([100, 0, 0], [255, 50, 50]),
            'yellow': ([0, 100, 100], [50, 255, 255]),
            'orange': ([0, 100, 150], [50, 255, 255]),
            'purple': ([100, 0, 100], [255, 50, 255])
        }

        # Detected objects database
        self.detected_objects = []

    def camera_info_callback(self, msg):
        """Store camera information"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process incoming image for multimodal perception"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {str(e)}')
            return

        # Perform object detection
        detection_array = self.detect_objects(cv_image, msg.header)

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Update detected objects list
        self.update_detected_objects(detection_array.detections)

        # Process pending command if available
        if self.pending_command:
            self.process_command(self.pending_command, detection_array)
            self.pending_command = None

        # Create scene description
        scene_description = self.describe_scene(detection_array.detections)
        scene_msg = String()
        scene_msg.data = scene_description
        self.scene_description_pub.publish(scene_msg)

        self.get_logger().debug(f'Multimodal perception complete - {len(detection_array.detections)} objects detected')

    def scan_callback(self, msg):
        """Process LIDAR scan data"""
        self.latest_scan = msg

    def command_callback(self, msg):
        """Process incoming command"""
        self.pending_command = msg.data
        self.get_logger().info(f'Received command: {msg.data}')

    def detect_objects(self, cv_image, header):
        """Detect objects in the image using color-based detection"""
        detection_array = Detection2DArray()
        detection_array.header = header

        # Detect objects by color
        for color_name, (lower, upper) in self.colors_to_detect.items():
            # Create mask for the current color
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(cv_image, lower, upper)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour
            for contour in contours:
                # Filter by area to avoid small noise
                area = cv2.contourArea(contour)
                if area > 300:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Create detection
                    detection = Detection2D()
                    detection.header = header

                    # Set the bounding box
                    detection.bbox.center.x = x + w / 2
                    detection.bbox.center.y = y + h / 2
                    detection.bbox.size_x = w
                    detection.bbox.size_y = h

                    # Create hypothesis with the detected object
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = f'{color_name}_object'
                    hypothesis.hypothesis.score = min(0.9, 0.5 + (area / 10000))  # Confidence based on size
                    detection.results.append(hypothesis)

                    detection_array.detections.append(detection)

        return detection_array

    def update_detected_objects(self, detections):
        """Update the list of detected objects"""
        self.detected_objects = []
        for detection in detections:
            if detection.results:
                obj_info = {
                    'class': detection.results[0].hypothesis.class_id,
                    'confidence': detection.results[0].hypothesis.score,
                    'position': (detection.bbox.center.x, detection.bbox.center.y),
                    'size': (detection.bbox.size_x, detection.bbox.size_y)
                }
                self.detected_objects.append(obj_info)

    def describe_scene(self, detections):
        """Create a textual description of the scene"""
        if not detections:
            return "The scene appears to be empty or no objects were detected."

        description = f"I can see {len(detections)} objects in the scene: "
        objects_by_color = {}

        for detection in detections:
            if detection.results:
                class_name = detection.results[0].hypothesis.class_id
                color = class_name.split('_')[0]  # Extract color from class name

                if color not in objects_by_color:
                    objects_by_color[color] = 0
                objects_by_color[color] += 1

        color_descriptions = []
        for color, count in objects_by_color.items():
            if count == 1:
                color_descriptions.append(f"one {color} object")
            else:
                color_descriptions.append(f"{count} {color} objects")

        description += ", ".join(color_descriptions) + "."
        return description

    def process_command(self, command, detections):
        """Process a command against the current detections"""
        command_lower = command.lower()

        # Check if command is asking about objects
        if "find" in command_lower or "where" in command_lower or "see" in command_lower:
            response = self.handle_object_query(command_lower, detections)
            response_msg = String()
            response_msg.data = response
            self.object_query_pub.publish(response_msg)

    def handle_object_query(self, command, detections):
        """Handle object-related queries"""
        # Extract object type from command
        object_types = ["red", "green", "blue", "yellow", "orange", "purple"]
        found_objects = []

        for obj_type in object_types:
            if obj_type in command:
                for detection in detections.detections:
                    if detection.results and obj_type in detection.results[0].hypothesis.class_id:
                        found_objects.append({
                            'type': detection.results[0].hypothesis.class_id,
                            'position': (detection.bbox.center.x, detection.bbox.center.y),
                            'confidence': detection.results[0].hypothesis.score
                        })

        if found_objects:
            response = f"I found {len(found_objects)} {obj_type} object(s). "
            positions = [f"at position ({obj['position'][0]:.0f}, {obj['position'][1]:.0f})" for obj in found_objects]
            response += "They are located " + ", ".join(positions) + "."
        else:
            response = f"I couldn't find any {obj_type} objects in the current view."

        return response

    def get_object_location(self, object_class):
        """Get the location of a specific object class"""
        for obj in self.detected_objects:
            if object_class in obj['class']:
                return obj['position']
        return None


def main(args=None):
    rclpy.init(args=args)

    perception_node = MultimodalPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()