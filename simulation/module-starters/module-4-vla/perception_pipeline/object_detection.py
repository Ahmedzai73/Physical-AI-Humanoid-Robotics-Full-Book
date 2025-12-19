#!/usr/bin/env python3

"""
Object Detection Module for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module demonstrates object detection capabilities for the VLA system.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud2, PointField
import struct


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

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

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/object_detection/detections',
            10
        )

        self.detection_image_pub = self.create_publisher(
            Image,
            '/object_detection/annotated_image',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/object_detection/status',
            10
        )

        self.get_logger().info('Object Detection Node initialized')

        # Store camera info
        self.camera_info = None

        # Object detection parameters
        self.colors_to_detect = {
            'red': ([0, 0, 100], [50, 50, 255]),
            'green': ([0, 100, 0], [50, 255, 50]),
            'blue': ([100, 0, 0], [255, 50, 50]),
            'yellow': ([0, 100, 100], [50, 255, 255]),
            'orange': ([0, 100, 150], [50, 255, 255]),
            'purple': ([100, 0, 100], [255, 50, 255]),
            'cyan': ([100, 100, 0], [255, 255, 50])
        }

        # Color boundaries for annotation
        self.color_boundaries = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'cyan': (255, 255, 0)
        }

    def camera_info_callback(self, msg):
        """Store camera information"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process incoming image for object detection"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {str(e)}')
            return

        # Perform object detection
        detection_array = self.detect_objects(cv_image, msg.header)

        # Annotate image with detections
        annotated_image = self.annotate_image(cv_image, detection_array.detections)

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.detection_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Could not convert annotated image: {str(e)}')

        # Publish status
        status_msg = String()
        status_msg.data = f'Processed image with {len(detection_array.detections)} detections'
        self.status_pub.publish(status_msg)

        self.get_logger().debug(f'Object detection complete - {len(detection_array.detections)} objects detected')

    def detect_objects(self, cv_image, header):
        """Detect objects in the image using multiple methods"""
        detection_array = Detection2DArray()
        detection_array.header = header

        # Method 1: Color-based detection
        color_detections = self.detect_by_color(cv_image, header)
        detection_array.detections.extend(color_detections)

        # Method 2: Shape-based detection could be added here
        # shape_detections = self.detect_by_shape(cv_image, header)
        # detection_array.detections.extend(shape_detections)

        return detection_array

    def detect_by_color(self, cv_image, header):
        """Detect objects based on color"""
        detections = []

        for color_name, (lower, upper) in self.colors_to_detect.items():
            # Create mask for the current color
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(cv_image, lower, upper)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour
            for contour in contours:
                # Filter by area to avoid small noise
                area = cv2.contourArea(contour)
                if area > 300:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center and dimensions
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # Create detection
                    detection = Detection2D()
                    detection.header = header

                    # Set the bounding box
                    detection.bbox.center.x = float(center_x)
                    detection.bbox.center.y = float(center_y)
                    detection.bbox.size_x = float(w)
                    detection.bbox.size_y = float(h)

                    # Create hypothesis with the detected object
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = f'{color_name}_object'
                    # Confidence based on area and fill ratio
                    fill_ratio = area / (w * h) if w * h > 0 else 0
                    confidence = min(0.95, 0.3 + (area / 10000) + fill_ratio * 0.3)
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)

                    # Add geometric properties as additional results
                    size_hypothesis = ObjectHypothesisWithPose()
                    size_hypothesis.hypothesis.class_id = f'size_{w}x{h}'
                    size_hypothesis.hypothesis.score = 1.0
                    detection.results.append(size_hypothesis)

                    detections.append(detection)

        return detections

    def annotate_image(self, cv_image, detections):
        """Annotate the image with detection results"""
        annotated_image = cv_image.copy()

        for detection in detections:
            if detection.results:
                # Get the primary result
                primary_result = detection.results[0]
                class_name = primary_result.hypothesis.class_id
                confidence = primary_result.hypothesis.score

                # Extract color name from class
                color_name = class_name.split('_')[0] if '_' in class_name else 'unknown'
                color = self.color_boundaries.get(color_name, (255, 255, 255))

                # Draw bounding box
                center_x = int(detection.bbox.center.x)
                center_y = int(detection.bbox.center.y)
                size_x = int(detection.bbox.size_x / 2)
                size_y = int(detection.bbox.size_y / 2)

                top_left = (center_x - size_x, center_y - size_y)
                bottom_right = (center_x + size_x, center_y + size_y)

                cv2.rectangle(annotated_image, top_left, bottom_right, color, 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_rect = (top_left[0], top_left[1] - label_size[1] - 10,
                             label_size[0], label_size[1] + 10)

                cv2.rectangle(annotated_image,
                             (label_rect[0], label_rect[1]),
                             (label_rect[0] + label_rect[2], label_rect[1] + label_rect[3]),
                             color, -1)

                cv2.putText(annotated_image, label,
                           (top_left[0], top_left[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated_image


def main(args=None):
    rclpy.init(args=args)

    detection_node = ObjectDetectionNode()

    try:
        rclpy.spin(detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()