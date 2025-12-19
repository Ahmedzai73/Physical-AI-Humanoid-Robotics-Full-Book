#!/usr/bin/env python3

"""
Perception-Action Integration for LLM-Controlled Robots
Module 4: Vision-Language-Action (VLA) - Physical AI & Humanoid Robotics Textbook

This module implements perception-action integration for the VLA system,
allowing the AI to perceive the environment and take appropriate actions
based on the perceived information.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2, CameraInfo
from geometry_msgs.msg import Point, Pose, Twist
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import math
from typing import Dict, Any, Optional, List
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class PerceptionActionIntegrationNode(Node):
    """
    Node that integrates perception and action for LLM-controlled robots
    """
    def __init__(self):
        super().__init__('perception_action_integration_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.action_command_pub = self.create_publisher(
            Twist,
            '/vla/action_command',
            10
        )

        self.perception_status_pub = self.create_publisher(
            String,
            '/perception_action/status',
            10
        )

        self.perception_feedback_pub = self.create_publisher(
            String,
            '/perception_action/feedback',
            10
        )

        # Subscribers
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/pointcloud',
            self.pointcloud_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.llm_perception_request_sub = self.create_subscription(
            String,
            '/vla/llm_perception_request',
            self.llm_perception_request_callback,
            10
        )

        # Internal state
        self.latest_image = None
        self.latest_lidar = None
        self.latest_pointcloud = None
        self.perception_active = False
        self.perception_results = {}
        self.object_detection_threshold = 0.5
        self.distance_threshold = 2.0  # meters

        # Perception parameters
        self.detection_model = None  # In a real implementation, this would be a loaded model
        self.tracking_enabled = True

        self.get_logger().info('Perception-Action Integration Node initialized')

    def camera_callback(self, msg: Image):
        """
        Process camera image for visual perception
        """
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image

            # Perform object detection if perception is active
            if self.perception_active:
                objects = self.detect_objects(cv_image)
                self.perception_results['objects'] = objects
                self.get_logger().info(f'Detected {len(objects)} objects in camera image')

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')

    def lidar_callback(self, msg: LaserScan):
        """
        Process LIDAR data for distance perception
        """
        try:
            self.latest_lidar = msg

            # Process LIDAR data for obstacle detection
            if self.perception_active:
                obstacles = self.detect_obstacles_from_lidar(msg)
                self.perception_results['obstacles'] = obstacles
                self.get_logger().info(f'Detected {len(obstacles)} obstacles from LIDAR')

        except Exception as e:
            self.get_logger().error(f'Error processing LIDAR data: {str(e)}')

    def pointcloud_callback(self, msg: PointCloud2):
        """
        Process point cloud data for 3D perception
        """
        try:
            self.latest_pointcloud = msg

            # Process point cloud for 3D object detection
            if self.perception_active:
                objects_3d = self.detect_objects_from_pointcloud(msg)
                self.perception_results['objects_3d'] = objects_3d
                self.get_logger().info(f'Detected {len(objects_3d)} 3D objects from point cloud')

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud data: {str(e)}')

    def llm_perception_request_callback(self, msg: String):
        """
        Process LLM-generated perception requests
        """
        try:
            request_data = json.loads(msg.data)
            request_type = request_data.get('type', 'object_detection')

            if request_type == 'object_detection':
                target_object = request_data.get('target', 'any')
                self.perform_object_detection(target_object)
            elif request_type == 'obstacle_detection':
                self.perform_obstacle_detection()
            elif request_type == 'environment_mapping':
                self.perform_environment_mapping()
            elif request_type == 'object_tracking':
                target_object = request_data.get('target', 'any')
                self.perform_object_tracking(target_object)
            elif request_type == 'action_planning':
                self.perform_action_planning(request_data)
            else:
                self.get_logger().warn(f'Unknown perception request type: {request_type}')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in perception request: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing LLM perception request: {str(e)}')

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the image using computer vision techniques
        """
        try:
            # In a real implementation, this would use a trained object detection model
            # For this example, we'll use simple color-based detection as a placeholder
            objects = []

            # Convert to HSV for color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define color ranges for different objects
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'green': ([50, 50, 50], [70, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'yellow': ([20, 50, 50], [30, 255, 255])
            }

            height, width = image.shape[:2]

            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for the color
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Filter out small areas
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        # Calculate center and relative position
                        center_x = x + w / 2
                        center_y = y + h / 2
                        relative_x = (center_x - width / 2) / (width / 2)  # -1 to 1
                        relative_y = (center_y - height / 2) / (height / 2)  # -1 to 1

                        # Calculate size relative to image
                        size_ratio = (w * h) / (width * height)

                        object_info = {
                            'name': color_name,
                            'x': relative_x,
                            'y': relative_y,
                            'width': w,
                            'height': h,
                            'size_ratio': size_ratio,
                            'confidence': min(0.9, area / 10000)  # Normalize confidence
                        }

                        objects.append(object_info)

            return objects

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
            return []

    def detect_obstacles_from_lidar(self, lidar_msg: LaserScan) -> List[Dict[str, float]]:
        """
        Detect obstacles from LIDAR data
        """
        try:
            obstacles = []
            ranges = lidar_msg.ranges
            angle_min = lidar_msg.angle_min
            angle_increment = lidar_msg.angle_increment

            for i, range_val in enumerate(ranges):
                if not math.isnan(range_val) and range_val < self.distance_threshold:
                    angle = angle_min + i * angle_increment
                    x = range_val * math.cos(angle)
                    y = range_val * math.sin(angle)

                    obstacle = {
                        'x': x,
                        'y': y,
                        'distance': range_val,
                        'angle': angle
                    }

                    obstacles.append(obstacle)

            return obstacles

        except Exception as e:
            self.get_logger().error(f'Error in LIDAR obstacle detection: {str(e)}')
            return []

    def detect_objects_from_pointcloud(self, pointcloud_msg: PointCloud2) -> List[Dict[str, Any]]:
        """
        Detect objects from point cloud data
        """
        try:
            # In a real implementation, this would use point cloud processing libraries
            # like PCL (Point Cloud Library) or Open3D
            # For this example, we'll return a placeholder
            objects = []

            # Placeholder: simulate detection of objects in point cloud
            # In reality, you would process the point cloud data to segment objects
            objects.append({
                'name': 'table',
                'x': 1.0,
                'y': 0.0,
                'z': 0.0,
                'size': [0.8, 0.6, 0.72],  # width, depth, height
                'confidence': 0.85
            })

            objects.append({
                'name': 'object_on_table',
                'x': 1.2,
                'y': 0.1,
                'z': 0.72,
                'size': [0.1, 0.1, 0.1],  # width, depth, height
                'confidence': 0.78
            })

            return objects

        except Exception as e:
            self.get_logger().error(f'Error in point cloud object detection: {str(e)}')
            return []

    def perform_object_detection(self, target_object: str = 'any'):
        """
        Perform object detection for the specified target
        """
        try:
            self.get_logger().info(f'Performing object detection for: {target_object}')

            # Set perception active
            self.perception_active = True

            # Wait for latest image
            if self.latest_image is None:
                self.get_logger().warn('No image data available for object detection')
                return []

            # Detect objects
            objects = self.detect_objects(self.latest_image)

            # Filter for target object if specified
            if target_object.lower() != 'any':
                objects = [obj for obj in objects if obj['name'].lower() == target_object.lower()]

            # Publish results
            result_msg = String()
            result_msg.data = json.dumps({
                'type': 'object_detection',
                'target': target_object,
                'objects': objects,
                'timestamp': self.get_clock().now().to_msg().sec
            })
            self.perception_feedback_pub.publish(result_msg)

            self.get_logger().info(f'Object detection completed: found {len(objects)} {target_object} objects')
            return objects

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
            return []

    def perform_obstacle_detection(self):
        """
        Perform obstacle detection using LIDAR data
        """
        try:
            self.get_logger().info('Performing obstacle detection')

            # Set perception active
            self.perception_active = True

            # Wait for latest LIDAR data
            if self.latest_lidar is None:
                self.get_logger().warn('No LIDAR data available for obstacle detection')
                return []

            # Detect obstacles
            obstacles = self.detect_obstacles_from_lidar(self.latest_lidar)

            # Publish results
            result_msg = String()
            result_msg.data = json.dumps({
                'type': 'obstacle_detection',
                'obstacles': obstacles,
                'timestamp': self.get_clock().now().to_msg().sec
            })
            self.perception_feedback_pub.publish(result_msg)

            self.get_logger().info(f'Obstacle detection completed: found {len(obstacles)} obstacles')
            return obstacles

        except Exception as e:
            self.get_logger().error(f'Error in obstacle detection: {str(e)}')
            return []

    def perform_environment_mapping(self):
        """
        Perform environment mapping using multiple sensors
        """
        try:
            self.get_logger().info('Performing environment mapping')

            # Set perception active
            self.perception_active = True

            # Collect data from all sensors
            map_data = {
                'timestamp': self.get_clock().now().to_msg().sec,
                'objects': [],
                'obstacles': [],
                'landmarks': []
            }

            # Get object data from camera
            if self.latest_image is not None:
                map_data['objects'] = self.detect_objects(self.latest_image)

            # Get obstacle data from LIDAR
            if self.latest_lidar is not None:
                map_data['obstacles'] = self.detect_obstacles_from_lidar(self.latest_lidar)

            # Get 3D data from point cloud
            if self.latest_pointcloud is not None:
                map_data['landmarks'] = self.detect_objects_from_pointcloud(self.latest_pointcloud)

            # Publish results
            result_msg = String()
            result_msg.data = json.dumps({
                'type': 'environment_mapping',
                'map_data': map_data
            })
            self.perception_feedback_pub.publish(result_msg)

            self.get_logger().info('Environment mapping completed')
            return map_data

        except Exception as e:
            self.get_logger().error(f'Error in environment mapping: {str(e)}')
            return {}

    def perform_object_tracking(self, target_object: str = 'any'):
        """
        Perform object tracking for the specified target
        """
        try:
            self.get_logger().info(f'Performing object tracking for: {target_object}')

            # Set perception active
            self.perception_active = True

            # In a real implementation, this would use tracking algorithms
            # like Kalman filters or deep learning trackers
            # For this example, we'll simulate tracking based on detection

            if self.latest_image is None:
                self.get_logger().warn('No image data available for object tracking')
                return []

            # Detect objects in current frame
            current_objects = self.detect_objects(self.latest_image)

            # Filter for target object
            if target_object.lower() != 'any':
                current_objects = [obj for obj in current_objects if obj['name'].lower() == target_object.lower()]

            # Publish tracking results
            result_msg = String()
            result_msg.data = json.dumps({
                'type': 'object_tracking',
                'target': target_object,
                'tracked_objects': current_objects,
                'timestamp': self.get_clock().now().to_msg().sec
            })
            self.perception_feedback_pub.publish(result_msg)

            self.get_logger().info(f'Object tracking completed: tracking {len(current_objects)} {target_object} objects')
            return current_objects

        except Exception as e:
            self.get_logger().error(f'Error in object tracking: {str(e)}')
            return []

    def perform_action_planning(self, request_data: Dict[str, Any]):
        """
        Plan actions based on perception results
        """
        try:
            self.get_logger().info('Performing action planning based on perception')

            # Get current perception results
            perception_results = self.perception_results.copy()

            # Analyze perception results to determine appropriate action
            action = self.analyze_perception_for_action(perception_results, request_data)

            if action:
                # Publish action command
                action_msg = Twist()
                action_msg.linear.x = action.get('linear_x', 0.0)
                action_msg.linear.y = action.get('linear_y', 0.0)
                action_msg.linear.z = action.get('linear_z', 0.0)
                action_msg.angular.x = action.get('angular_x', 0.0)
                action_msg.angular.y = action.get('angular_y', 0.0)
                action_msg.angular.z = action.get('angular_z', 0.0)

                self.action_command_pub.publish(action_msg)

                # Publish feedback
                feedback_msg = String()
                feedback_msg.data = json.dumps({
                    'type': 'action_planning',
                    'action': action,
                    'perception_results': perception_results,
                    'timestamp': self.get_clock().now().to_msg().sec
                })
                self.perception_feedback_pub.publish(feedback_msg)

                self.get_logger().info(f'Action planned and published: {action}')

        except Exception as e:
            self.get_logger().error(f'Error in action planning: {str(e)}')

    def analyze_perception_for_action(self, perception_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze perception results to determine appropriate action
        """
        try:
            action = {
                'linear_x': 0.0,
                'linear_y': 0.0,
                'linear_z': 0.0,
                'angular_x': 0.0,
                'angular_y': 0.0,
                'angular_z': 0.0
            }

            # Example action planning logic based on perception results
            target_object = request_data.get('target_object', 'any')
            action_type = request_data.get('action_type', 'navigate_to_object')

            if action_type == 'navigate_to_object':
                # Look for target object in detected objects
                if 'objects' in perception_results:
                    target_objects = [
                        obj for obj in perception_results['objects']
                        if obj['name'].lower() == target_object.lower() or target_object.lower() == 'any'
                    ]

                    if target_objects:
                        # Sort by size (largest first) to prioritize closer objects
                        target_objects.sort(key=lambda x: x['size_ratio'], reverse=True)
                        closest_object = target_objects[0]

                        # Calculate movement based on object position
                        obj_x = closest_object['x']  # -1 to 1 relative position
                        obj_y = closest_object['y']  # -1 to 1 relative position

                        # Convert relative position to movement commands
                        # Positive x means object is to the right, so turn right
                        action['angular_z'] = -obj_x * 0.5  # Proportional turning
                        # Move forward if object is in center of view
                        if abs(obj_x) < 0.2 and abs(obj_y) < 0.2:
                            action['linear_x'] = 0.3  # Move forward

            elif action_type == 'avoid_obstacles':
                # Look for obstacles in perception results
                if 'obstacles' in perception_results:
                    obstacles = perception_results['obstacles']

                    # Check for obstacles in front of robot
                    front_obstacles = [obs for obs in obstacles if abs(obs['angle']) < math.pi/4 and obs['distance'] < 0.8]

                    if front_obstacles:
                        # Find the closest obstacle in front
                        closest_front = min(front_obstacles, key=lambda x: x['distance'])

                        # Move away from obstacle
                        if closest_front['angle'] > 0:
                            # Obstacle on the right, turn left
                            action['angular_z'] = 0.5
                        else:
                            # Obstacle on the left, turn right
                            action['angular_z'] = -0.5

            elif action_type == 'follow_object':
                # Implement object following behavior
                if 'objects' in perception_results:
                    target_objects = [
                        obj for obj in perception_results['objects']
                        if obj['name'].lower() == target_object.lower() or target_object.lower() == 'any'
                    ]

                    if target_objects:
                        target_obj = target_objects[0]

                        # Follow if object is visible and not too close
                        if target_obj['size_ratio'] < 0.3:  # Object is far enough
                            action['linear_x'] = 0.2
                        elif target_obj['size_ratio'] > 0.5:  # Object is too close
                            action['linear_x'] = -0.1

                        # Adjust orientation to center the object
                        action['angular_z'] = -target_obj['x'] * 0.5

            return action

        except Exception as e:
            self.get_logger().error(f'Error in action analysis: {str(e)}')
            return {
                'linear_x': 0.0,
                'linear_y': 0.0,
                'linear_z': 0.0,
                'angular_x': 0.0,
                'angular_y': 0.0,
                'angular_z': 0.0
            }


def main(args=None):
    rclpy.init(args=args)

    perception_action_node = PerceptionActionIntegrationNode()

    try:
        rclpy.spin(perception_action_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_action_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()