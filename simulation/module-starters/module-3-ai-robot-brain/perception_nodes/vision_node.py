#!/usr/bin/env python3

"""
Vision Node for Physical AI & Humanoid Robotics Textbook
Module 3: The AI-Robot Brain (NVIDIA Isaac™)

This node demonstrates GPU-accelerated vision processing using Isaac ROS.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

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
            '/vision/detections',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/vision/status',
            10
        )

        self.get_logger().info('Vision Node initialized for Isaac ROS')

        # Store camera info
        self.camera_info = None

        # Define objects to detect
        self.colors_to_detect = {
            'red': ([0, 0, 100], [50, 50, 255]),
            'green': ([0, 100, 0], [50, 255, 50]),
            'blue': ([100, 0, 0], [255, 50, 50]),
            'yellow': ([0, 100, 100], [50, 255, 255])
        }

    def camera_info_callback(self, msg):
        """Store camera information"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process incoming image and perform vision tasks"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {str(e)}')
            return

        # Create detection array message
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        # Detect objects by color (basic example - in practice, use Isaac ROS vision nodes)
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
                    detection.header = msg.header

                    # Set the bounding box
                    detection.bbox.center.x = x + w / 2
                    detection.bbox.center.y = y + h / 2
                    detection.bbox.size_x = w
                    detection.bbox.size_y = h

                    # Create hypothesis with the detected object
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = color_name
                    hypothesis.hypothesis.score = min(0.9, 0.5 + (area / 10000))  # Confidence based on size
                    detection.results.append(hypothesis)

                    detection_array.detections.append(detection)

        # Publish detections
        self.detection_pub.publish(detection_array)

        # Publish status
        status_msg = String()
        status_msg.data = f'Processed image with {len(detection_array.detections)} detections'
        self.status_pub.publish(status_msg)

        self.get_logger().debug(f'Vision processing complete - {len(detection_array.detections)} objects detected')


def main(args=None):
    rclpy.init(args=args)

    vision_node = VisionNode()

    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()